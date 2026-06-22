"""
tune_sahi_dino.py — Hyperparameter Grid Search for SAHI+DINO on Indian Traffic

PURPOSE
=======
Indian traffic is pathologically different from the COCO/KITTI datasets that
RT-DETR and DINO were trained and benchmarked on:
  • Motorcycles: dominant class, often 3-4 riders per bike, 10-30 per frame
  • Inter-vehicle gaps: <0.5 m at intersections
  • Lane discipline: absent (vehicles fill all available space)
  • Velocity: unpredictable stop-and-go mixed with sudden swerves

The default SAHI parameters (512×512 slices, 0.30 overlap) were designed for
COCO-style scenes.  This script exhaustively tests 16 parameter combinations
on a representative video to find the optimal configuration for this domain.

GRID
====
Dimensions:
  slice_height  : [512, 640]
  slice_width   : [512, 640]
  overlap_ratio : [0.25, 0.35]
  conf_threshold: [0.08, 0.12]
Total: 2 × 2 × 2 × 2 = 16 configurations

SCORING (Indian traffic optimised)
====================================
  score = motorcycle_count × 0.5 − id_switches × 0.5

This maximises motorcycle recall (the hardest class in Indian traffic) while
penalising tracking instability (ID switches indicate tile-collision duplicates
are confusing ByteTrack's Stage-1 matcher).

OUTPUT
======
  • tune_results.csv: 16-row CSV with all metrics
  • Printed recommendation: best config + why it was chosen

USAGE
=====
  python src/tune_sahi_dino.py \\
      --video data/video/test1.mp4 \\
      --model rtdetr-x.pt \\
      --device cuda \\
      --max-frames 200 \\
      --output-csv tune_results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Make sure src/ is on the path when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from sahi_dino_detection import SahiDinoDetector
from tracker import BYTETracker, STrack
from utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID-switch counter helper
# ---------------------------------------------------------------------------

def _count_id_switches(prev_ids: set, curr_ids: set) -> int:
    """
    Count IDs that were present last frame but are gone this frame, AND
    IDs that are new this frame but were not present last frame — each pair
    potentially represents an ID switch (one track ended, another began for
    the same physical object).

    Returns:
        Number of new IDs introduced that are NOT in prev_ids.
    """
    new_ids = curr_ids - prev_ids
    return len(new_ids)


# ---------------------------------------------------------------------------
# Single configuration evaluator
# ---------------------------------------------------------------------------

def evaluate_configuration(
    video_path: str,
    model_path: str,
    slice_height: int,
    slice_width: int,
    overlap_ratio: float,
    conf_threshold: float,
    device: str,
    max_frames: int = 200,
    # ByteTrack standard params (fixed across all configs for fair comparison)
    high_thresh: float = 0.50,
    low_thresh: float = 0.10,
    match_thresh: float = 0.80,
    track_buffer: int = 30,
    motorcycle_track_buffer: int = 60,
    motorcycle_match_thresh: float = 0.70,
    min_hits: int = 3,
) -> Dict:
    """
    Run SahiDinoDetector + BYTETracker on the first ``max_frames`` of ``video_path``
    with the given configuration.

    Returns a metrics dict:
        slice_height, slice_width, overlap_ratio, conf_threshold,
        avg_dets_per_frame, motorcycle_count, car_count, person_count,
        tile_collision_rate_pct, suppression_rate_pct,
        total_tracks, id_switches, id_switch_rate_pct,
        fps, total_time_s, frames_processed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # ---- Instantiate detector ------------------------------------------------
    detector = SahiDinoDetector(
        model_path           = model_path,
        slice_height         = slice_height,
        slice_width          = slice_width,
        overlap_height_ratio = overlap_ratio,
        overlap_width_ratio  = overlap_ratio,  # symmetric overlap
        conf                 = conf_threshold,
        device               = device,
    )

    # ---- Instantiate tracker (fixed params for fair comparison) --------------
    STrack.reset_id_counter()
    tracker = BYTETracker(
        high_thresh             = high_thresh,
        low_thresh              = low_thresh,
        match_thresh            = match_thresh,
        track_buffer            = track_buffer,
        min_hits                = min_hits,
        motorcycle_track_buffer = motorcycle_track_buffer,
        motorcycle_match_thresh = motorcycle_match_thresh,
        device                  = device,
    )

    # ---- Per-frame accumulators -----------------------------------------------
    total_dets         = 0
    motorcycle_count   = 0
    car_count          = 0
    person_count       = 0
    total_tracks_seen: set  = set()
    id_switches        = 0
    prev_frame_ids: set = set()

    frame_idx  = 0
    t_start    = time.perf_counter()

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        detections = detector.detect(frame)

        # Track
        tracks = tracker.update(detections, frame)

        # Accumulate
        total_dets += len(detections)
        motorcycle_count += sum(1 for d in detections if d["class_id"] == 3)
        car_count        += sum(1 for d in detections if d["class_id"] == 2)
        person_count     += sum(1 for d in detections if d["class_id"] == 0)

        curr_frame_ids = {t["track_id"] for t in tracks}
        id_switches += _count_id_switches(prev_frame_ids, curr_frame_ids)
        total_tracks_seen.update(curr_frame_ids)
        prev_frame_ids = curr_frame_ids

        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t_start

    # ---- Detector internal stats --------------------------------------------
    det_stats = detector.get_stats()

    # ---- Compute derived metrics --------------------------------------------
    frames_processed  = max(frame_idx, 1)
    avg_dets          = total_dets / frames_processed
    id_switch_rate    = id_switches / max(len(total_tracks_seen), 1) * 100.0
    fps               = frames_processed / max(elapsed, 1e-6)
    tile_coll_rate    = det_stats.get("tile_collision_rate", 0.0)
    suppression_rate  = det_stats.get("reduction_pct", 0.0)

    return {
        # Config
        "slice_height":         slice_height,
        "slice_width":          slice_width,
        "overlap_ratio":        overlap_ratio,
        "conf_threshold":       conf_threshold,
        # Detection metrics
        "avg_dets_per_frame":   round(avg_dets, 2),
        "motorcycle_count":     motorcycle_count,
        "car_count":            car_count,
        "person_count":         person_count,
        "tile_collision_rate_pct": round(tile_coll_rate, 2),
        "suppression_rate_pct": round(suppression_rate, 2),
        # Tracking metrics
        "total_unique_tracks":  len(total_tracks_seen),
        "id_switches":          id_switches,
        "id_switch_rate_pct":   round(id_switch_rate, 2),
        # Speed
        "fps":                  round(fps, 2),
        "total_time_s":         round(elapsed, 2),
        "frames_processed":     frame_idx,
    }


# ---------------------------------------------------------------------------
# Scoring & recommendation
# ---------------------------------------------------------------------------

def score_config(metrics: Dict) -> float:
    """
    Score a configuration for Indian traffic suitability.

    Objective:
      Maximise motorcycle recall (motorcycle_count proxy).
      Minimise ID switches (tracking stability).
      Penalise excessive detection rates (may indicate many false positives).

    Returns a scalar — higher is better.
    """
    mc = metrics["motorcycle_count"]
    ids = metrics["id_switches"]
    # Bonus for keeping tile collisions low
    collision_penalty = metrics["tile_collision_rate_pct"] * 0.1
    return mc * 0.5 - ids * 0.5 - collision_penalty


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def _print_results_table(results: List[Dict]) -> None:
    """Print a formatted results table to stdout."""
    print("\n" + "=" * 100)
    print("  TUNING RESULTS — SAHI+DINO on Indian Traffic")
    print("=" * 100)
    header = (
        f"{'Slice':>12} {'Overlap':>8} {'Conf':>6} "
        f"{'Motos':>7} {'Cars':>6} {'AvgDets':>8} "
        f"{'IDSw':>6} {'IDSw%':>6} "
        f"{'CollRate':>9} {'SuppRate':>9} "
        f"{'FPS':>6}"
    )
    print(header)
    print("-" * 100)
    for r in sorted(results, key=lambda x: score_config(x), reverse=True):
        print(
            f"{r['slice_height']:>5}x{r['slice_width']:<5} "
            f"{r['overlap_ratio']:>8.2f} "
            f"{r['conf_threshold']:>6.2f} "
            f"{r['motorcycle_count']:>7} "
            f"{r['car_count']:>6} "
            f"{r['avg_dets_per_frame']:>8.1f} "
            f"{r['id_switches']:>6} "
            f"{r['id_switch_rate_pct']:>5.1f}% "
            f"{r['tile_collision_rate_pct']:>8.1f}% "
            f"{r['suppression_rate_pct']:>8.1f}% "
            f"{r['fps']:>6.1f}"
        )
    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tune_sahi_dino",
        description="Grid search for SAHI+DINO hyperparameters on Indian traffic footage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",       required=True, help="Input video file path.")
    p.add_argument("--model",       default="rtdetr-x.pt", help="RT-DETRv2 weights file.")
    p.add_argument("--device",      default=None, help="Inference device (cuda/cpu/mps).")
    p.add_argument("--max-frames",  type=int, default=200, help="Frames per config to test.")
    p.add_argument("--output-csv",  default="tune_results.csv", help="Output CSV file path.")
    p.add_argument(
        "--slice-heights", nargs="+", type=int, default=[512, 640],
        help="SAHI slice heights to test.",
    )
    p.add_argument(
        "--slice-widths", nargs="+", type=int, default=[512, 640],
        help="SAHI slice widths to test.",
    )
    p.add_argument(
        "--overlap-ratios", nargs="+", type=float, default=[0.25, 0.35],
        help="SAHI overlap ratios to test (symmetric H+W).",
    )
    p.add_argument(
        "--conf-thresholds", nargs="+", type=float, default=[0.08, 0.12],
        help="Global confidence thresholds to test.",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    video_path = str(Path(args.video))
    if not Path(video_path).exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    # Build grid
    configs: List[Tuple] = [
        (sh, sw, ov, ct)
        for sh in args.slice_heights
        for sw in args.slice_widths
        for ov in args.overlap_ratios
        for ct in args.conf_thresholds
    ]

    n_configs = len(configs)
    logger.info(
        "Tuning SAHI+DINO: %d configurations × %d frames each on '%s'",
        n_configs, args.max_frames, video_path,
    )

    results: List[Dict] = []
    csv_fields: List[str] = []

    for idx, (sh, sw, ov, ct) in enumerate(configs, start=1):
        tag = f"{sh}×{sw} overlap={ov:.2f} conf={ct:.2f}"
        logger.info("[%d/%d] Evaluating: %s", idx, n_configs, tag)

        try:
            metrics = evaluate_configuration(
                video_path      = video_path,
                model_path      = args.model,
                slice_height    = sh,
                slice_width     = sw,
                overlap_ratio   = ov,
                conf_threshold  = ct,
                device          = args.device or "cpu",
                max_frames      = args.max_frames,
            )
        except Exception as exc:
            logger.error("Config %s FAILED: %s", tag, exc, exc_info=True)
            continue

        metrics["score"] = round(score_config(metrics), 3)
        results.append(metrics)
        if not csv_fields:
            csv_fields = list(metrics.keys())

        print(
            f"  ✓ [{idx:>2}/{n_configs}] {tag}: "
            f"motos={metrics['motorcycle_count']} "
            f"id_sw={metrics['id_switches']} "
            f"coll={metrics['tile_collision_rate_pct']:.1f}% "
            f"fps={metrics['fps']:.1f} "
            f"score={metrics['score']:.1f}"
        )

    if not results:
        logger.error("No configurations completed successfully.")
        sys.exit(1)

    # ---- Print table --------------------------------------------------------
    _print_results_table(results)

    # ---- Save CSV -----------------------------------------------------------
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results saved to %s", out_path)

    # ---- Best config recommendation -----------------------------------------
    best = max(results, key=lambda r: r["score"])
    print(f"\n{'='*60}")
    print("  🏆  RECOMMENDED CONFIGURATION FOR INDIAN TRAFFIC")
    print(f"{'='*60}")
    print(f"  Slice size      : {best['slice_height']}×{best['slice_width']} px")
    print(f"  Overlap ratio   : {best['overlap_ratio']:.2f}")
    print(f"  Conf threshold  : {best['conf_threshold']:.2f}")
    print(f"  Score           : {best['score']:.2f}")
    print(f"  Motorcycles     : {best['motorcycle_count']}")
    print(f"  ID switches     : {best['id_switches']}")
    print(f"  Tile collisions : {best['tile_collision_rate_pct']:.1f}%")
    print(f"  Processing FPS  : {best['fps']:.1f}")
    print(f"\n  Usage:")
    print(f"  python src/main.py --input <video> --detector sahi_dino \\")
    print(f"      --slice-height {best['slice_height']} --slice-width {best['slice_width']} \\")
    print(f"      --overlap-height-ratio {best['overlap_ratio']:.2f} \\")
    print(f"      --overlap-width-ratio {best['overlap_ratio']:.2f} \\")
    print(f"      --conf {best['conf_threshold']:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
