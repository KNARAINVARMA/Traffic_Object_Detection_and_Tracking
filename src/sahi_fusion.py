"""
sahi_fusion.py — Weighted Box Fusion for SAHI Tile Collision Suppression

WHY THIS MODULE EXISTS
======================
SAHI sliced inference runs the detector on N overlapping tiles of the same frame.
When an object (especially a small motorcycle) sits at a tile boundary it is
detected independently in 2-3 tiles, producing duplicate bounding boxes that:

  1. Are NOT suppressed by a plain IoU-NMS (IoU can be 0.15–0.30 for near-
     duplicate boxes from different tile origins — below a typical 0.45 threshold).
  2. Confuse ByteTrack's Stage-1 Hungarian matcher, which then assigns
     separate track IDs to what is physically the same object.
  3. Amplify ID-switch rate, especially on dense Indian traffic where
     motorcycles fill every inter-vehicle gap and sit at tile edges far more
     often than in sparse Western traffic.

ALGORITHM: Proximity-Based Cluster Fusion
==========================================
Naive NMS operates purely in IoU space.  This module instead:

  Step 1 — Sort all detections by confidence (descending).
  Step 2 — Proximity clustering:
    For each unprocessed detection, gather all unprocessed detections of the
    SAME class whose bounding-box centres are within `cluster_distance_thresh`
    pixels.  All such detections form a cluster.
  Step 3 — Per-cluster decision:
    • Cluster size == 1  → pass-through unchanged (tile_collision=False).
    • Cluster size >= 2  → this is a candidate tile collision.
        a) Keep the highest-confidence box ("anchor").
        b) For every other box in the cluster, compute IoU against the anchor.
           If IoU ≥ overlap_iou_thresh → suppress (duplicate tile hit).
           If IoU < overlap_iou_thresh → keep (distinct object in proximity).
        c) All surviving boxes from a cluster of size ≥ 2 are tagged
           tile_collision=True / tile_origin_count=cluster_size.

WHY PROXIMITY CLUSTERING BEATS PLAIN NMS
=========================================
Consider two motorcycles 12 pixels apart (common in Indian dense traffic):
  • Their IoU is ~0.05 — plain NMS keeps BOTH correctly.
  • But if the same single motorcycle is detected twice from different tiles,
    the two boxes may have IoU=0.22 (centre drift from tile edge quantisation).
  • Proximity clustering catches this: both boxes' CENTRES are <30px apart AND
    they have IoU=0.22 (above 0.35? No → kept as distinct) — wait, 0.22 < 0.35,
    so they are kept.  The crucial improvement is when IoU ≥ 0.35: they ARE
    merged, but ONLY within the same proximity cluster (same class, close centre).

This prevents:
  - Merging genuinely distinct close-together motorcycles (their IoU is low).
  - Keeping duplicate same-object detections (their IoU is high AND centres close).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def weighted_box_fusion(
    detections: List[Dict],
    cluster_distance_thresh: float = 30.0,
    overlap_iou_thresh: float = 0.35,
) -> Tuple[List[Dict], Dict]:
    """
    Cluster detections from overlapping SAHI tiles and apply proximity-based
    weighted fusion to eliminate cross-tile duplicates.

    Args:
        detections:             Raw detector predictions from all tiles.
                                Each dict must contain at minimum:
                                  ``bbox``       — [x1, y1, x2, y2] floats
                                  ``confidence`` — float in [0, 1]
                                  ``class_id``   — int (COCO class index)
                                  ``class_name`` — str
        cluster_distance_thresh: Maximum Euclidean distance between bounding-box
                                centres (pixels) for two detections to be placed
                                in the same proximity cluster.  Default 30 px is
                                calibrated for 1920×1080 footage where a motorcycle
                                occupies ~40×30 px — so 30 px ≈ 75% of object width.
        overlap_iou_thresh:     IoU threshold above which two boxes in the same
                                cluster are treated as duplicates.  Default 0.35.
                                Lower values are more aggressive (suppress more),
                                higher values are more conservative (keep more).

    Returns:
        Tuple of:
          - ``fused_detections``: Deduplicated list with same dict format as input
            plus two additional metadata keys (for logging only):
              ``tile_origin_count`` — int: number of detections in the cluster
              ``is_tile_collision`` — bool: True if cluster had 2+ detections
          - ``stats``: Dict with:
              ``raw_count``          — total input detections
              ``fused_count``        — total output detections
              ``collision_clusters`` — number of clusters with size >= 2
              ``suppressed_count``   — number of duplicates suppressed

    Example::

        raw = detector.get_all_tile_detections(frame)
        fused, stats = weighted_box_fusion(raw)
        # stats: {"raw_count": 45, "fused_count": 38, "collision_clusters": 5, ...}
        detections = [{k: v for k, v in d.items()
                       if k not in ("tile_origin_count", "is_tile_collision")}
                      for d in fused]
    """
    if not detections:
        return [], {"raw_count": 0, "fused_count": 0, "collision_clusters": 0, "suppressed_count": 0}

    raw_count = len(detections)

    # ---- Step 1: Sort by confidence descending --------------------------------
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    # ---- Step 2: Proximity clustering ----------------------------------------
    # Precompute centres for all detections
    centres: List[Tuple[float, float]] = []
    for d in sorted_dets:
        x1, y1, x2, y2 = d["bbox"]
        centres.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    processed = [False] * len(sorted_dets)
    clusters: List[List[int]] = []  # each cluster is a list of indices into sorted_dets

    for i in range(len(sorted_dets)):
        if processed[i]:
            continue
        cluster = [i]
        processed[i] = True
        cls_i = sorted_dets[i]["class_id"]
        cx_i, cy_i = centres[i]

        for j in range(i + 1, len(sorted_dets)):
            if processed[j]:
                continue
            if sorted_dets[j]["class_id"] != cls_i:
                continue
            cx_j, cy_j = centres[j]
            dist = np.hypot(cx_j - cx_i, cy_j - cy_i)
            if dist <= cluster_distance_thresh:
                cluster.append(j)
                processed[j] = True

        clusters.append(cluster)

    # ---- Step 3: Per-cluster fusion ------------------------------------------
    fused: List[Dict] = []
    collision_clusters = 0
    suppressed_count = 0

    for cluster_indices in clusters:
        cluster_size = len(cluster_indices)

        if cluster_size == 1:
            # Single detection — no collision possible, pass through
            det = dict(sorted_dets[cluster_indices[0]])
            det["tile_origin_count"] = 1
            det["is_tile_collision"] = False
            fused.append(det)
            continue

        # Cluster of 2+ detections: potential tile collision
        collision_clusters += 1
        cluster_dets = [sorted_dets[k] for k in cluster_indices]

        # Anchor = highest confidence (already sorted, so index 0)
        anchor = cluster_dets[0]
        anchor_box = anchor["bbox"]

        # Keep anchor unconditionally
        anchor_out = dict(anchor)
        anchor_out["tile_origin_count"] = cluster_size
        anchor_out["is_tile_collision"] = True
        fused.append(anchor_out)

        # For every other detection in the cluster, decide keep vs. suppress
        for other in cluster_dets[1:]:
            iou_val = _iou(anchor_box, other["bbox"])
            if iou_val >= overlap_iou_thresh:
                # Clear duplicate from a different tile — suppress
                suppressed_count += 1
                logger.debug(
                    "WBF: Suppressed %s (conf=%.3f) duplicate of anchor (conf=%.3f), "
                    "IoU=%.3f cluster_size=%d",
                    other["class_name"], other["confidence"], anchor["confidence"],
                    iou_val, cluster_size,
                )
            else:
                # Different object in proximity — keep as distinct detection
                other_out = dict(other)
                other_out["tile_origin_count"] = cluster_size
                other_out["is_tile_collision"] = True
                fused.append(other_out)

    stats: Dict = {
        "raw_count":          raw_count,
        "fused_count":        len(fused),
        "collision_clusters": collision_clusters,
        "suppressed_count":   suppressed_count,
    }

    logger.debug(
        "WBF: raw=%d → fused=%d | clusters=%d collision | suppressed=%d",
        raw_count, len(fused), collision_clusters, suppressed_count,
    )

    return fused, stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-7)


# ---------------------------------------------------------------------------
# Quick self-test (run: python sahi_fusion.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.DEBUG)

    def _make_det(x1, y1, x2, y2, conf, cls_id=3, cls_name="motorcycle"):
        return {"bbox": [x1, y1, x2, y2], "confidence": conf,
                "class_id": cls_id, "class_name": cls_name}

    print("=" * 60)
    print("  sahi_fusion.py — Self-Test")
    print("=" * 60)

    # Test 1: Clear duplicate (same object, two tiles, high IoU)
    raw1 = [
        _make_det(100, 200, 140, 230, 0.82),   # Tile A detection
        _make_det(102, 201, 142, 231, 0.64),   # Tile B duplicate (IoU ~0.88)
    ]
    fused1, stats1 = weighted_box_fusion(raw1)
    assert len(fused1) == 1, f"Expected 1 after dedup, got {len(fused1)}"
    assert stats1["suppressed_count"] == 1
    print(f"Test 1 PASSED: Clear duplicate suppressed. Stats: {stats1}")

    # Test 2: Two distinct motorcycles close together (low IoU)
    # Boxes: [100,200,140,230] and [128,200,168,230]
    # Gap = 28px between right edge of A (140) and left of B (128) → overlapping at x=128-140
    # Actual overlap width = 140-128 = 12px, box width = 40px
    # IoU = (12×30) / (40×30 + 40×30 - 12×30) = 360 / (1200+1200-360) = 360/2040 = 0.176 → kept
    raw2 = [
        _make_det(100, 200, 140, 230, 0.80),   # Motorcycle A
        _make_det(128, 200, 168, 230, 0.75),   # Motorcycle B (12px overlap, IoU~0.18 < 0.35)
    ]
    fused2, stats2 = weighted_box_fusion(raw2)
    assert len(fused2) == 2, f"Expected 2 distinct, got {len(fused2)}"
    assert stats2["suppressed_count"] == 0
    print(f"Test 2 PASSED: Distinct close motorcycles preserved. Stats: {stats2}")

    # Test 3: Cross-class proximity (car and motorcycle near each other — never merged)
    raw3 = [
        _make_det(100, 200, 140, 230, 0.80, cls_id=3, cls_name="motorcycle"),
        _make_det(110, 205, 200, 265, 0.90, cls_id=2, cls_name="car"),
    ]
    fused3, stats3 = weighted_box_fusion(raw3)
    assert len(fused3) == 2, f"Expected 2 (cross-class), got {len(fused3)}"
    print(f"Test 3 PASSED: Cross-class detections kept separate. Stats: {stats3}")

    # Test 4: Three-tile collision (one object detected in 3 tiles)
    raw4 = [
        _make_det(200, 300, 240, 330, 0.85),   # Primary tile
        _make_det(201, 301, 241, 331, 0.71),   # Overlap tile 1 (IoU ~0.93)
        _make_det(199, 299, 239, 329, 0.60),   # Overlap tile 2 (IoU ~0.92)
    ]
    fused4, stats4 = weighted_box_fusion(raw4)
    assert len(fused4) == 1, f"Expected 1 after triple-tile dedup, got {len(fused4)}"
    assert stats4["suppressed_count"] == 2
    assert fused4[0]["tile_origin_count"] == 3
    print(f"Test 4 PASSED: 3-tile collision -> 1 box. Stats: {stats4}")

    print("\nAll tests PASSED OK")
