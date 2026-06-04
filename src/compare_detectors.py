"""
compare_detectors.py — A/B Benchmark: SAHI+RT-DETR vs SAHI+DINO

PURPOSE
=======
Side-by-side comparison of the old (RT-DETR-L) and new (DINO-style RT-DETRv2-X)
detection backends on the SAME input video using the SAME ByteTrack parameters.

This script isolates the detector as the only variable, so all differences in
the output metrics are attributable solely to the model change — not to any
tracker tuning or postprocessing changes.

METRICS MEASURED
================
Detection Quality:
  • avg_dets_per_frame    — raw detection rate per frame
  • motorcycle_count      — total motorcycle detections (key Indian traffic class)
  • car_count             — total car detections
  • conf_mean, conf_std   — confidence distribution (lower std = more stable)
  • tile_collision_rate   — % of clusters that were tile collisions (DINO only)
  • suppression_rate      — % of raw detections removed by WBF (DINO only)

Tracking Performance:
  • total_unique_tracks   — total track IDs assigned (more = more fragmentation)
  • id_switches           — ID changes between consecutive frames
  • id_switch_rate_pct    — id_switches / total_unique_tracks × 100

Speed:
  • fps                   — processing FPS
  • total_time_s          — wall clock time for the test segment

RECOMMENDATION LOGIC
=====================
  SWITCH TO DINO if:
    • motorcycle_count improvement ≥ 5%   (recall gain)
    • id_switch_rate reduction ≥ 10%      (tracking stability)
  KEEP RT-DETR if:
    • Neither threshold is met, or DINO is >20% slower.

USAGE
=====
  # Basic comparison (500 frames, both detectors):
  python src/compare_detectors.py --video data/video/test1.mp4

  # Quick test (50 frames):
  python src/compare_detectors.py --video data/video/test1.mp4 --max-frames 50

  # Specify models explicitly:
  python src/compare_detectors.py \\
      --video data/video/test1.mp4 \\
      --rtdetr-model models/rtdetr-l.pt \\
      --dino-model rtdetr-x.pt \\
      --device cuda \\
      --max-frames 500 \\
      --output-dir outputs/comparison
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Make sure src/ is on path when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from sahi_rtdetr_detection import SahiRTDetrDetector
from sahi_dino_detection import SahiDinoDetector
from tracker import BYTETracker, STrack
from utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single detector pipeline runner
# ---------------------------------------------------------------------------

def run_detector_pipeline(
    video_path: str,
    detector_name: str,
    detector,
    device: str,
    max_frames: int = 500,
    # ByteTrack params (fixed for fair comparison)
    high_thresh: float = 0.50,
    low_thresh: float = 0.10,
    match_thresh: float = 0.80,
    track_buffer: int = 30,
    motorcycle_track_buffer: int = 60,
    motorcycle_match_thresh: float = 0.70,
    min_hits: int = 3,
) -> Dict:
    """
    Run ``detector`` + BYTETracker on ``video_path`` for up to ``max_frames``.

    Returns a comprehensive metrics dict.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Fresh tracker for each detector run
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

    # Accumulators
    total_dets        = 0
    motorcycle_count  = 0
    car_count         = 0
    person_count      = 0
    bus_count         = 0
    truck_count       = 0
    all_confidences: List[float] = []
    total_tracks_seen: set = set()
    id_switches       = 0
    prev_frame_ids: set = set()
    per_frame_dets: List[int] = []

    frame_idx = 0
    t_start   = time.perf_counter()

    logger.info("Running pipeline: %s (max_frames=%d)", detector_name, max_frames)

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        detections = detector.detect(frame)

        # Track
        tracks = tracker.update(detections, frame)

        # Accumulate detection stats
        n_dets = len(detections)
        total_dets += n_dets
        per_frame_dets.append(n_dets)

        for d in detections:
            all_confidences.append(d["confidence"])
            cid = d["class_id"]
            if cid == 3:   motorcycle_count += 1
            elif cid == 2: car_count        += 1
            elif cid == 0: person_count     += 1
            elif cid == 5: bus_count        += 1
            elif cid == 7: truck_count      += 1

        # Accumulate tracking stats
        curr_frame_ids = {t["track_id"] for t in tracks}
        new_ids = curr_frame_ids - prev_frame_ids
        id_switches += len(new_ids)
        total_tracks_seen.update(curr_frame_ids)
        prev_frame_ids = curr_frame_ids

        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t_start

    # ---- Compute derived metrics --------------------------------------------
    frames_processed  = max(frame_idx, 1)
    avg_dets          = total_dets / frames_processed
    conf_arr          = np.array(all_confidences) if all_confidences else np.array([0.0])
    total_tracks      = len(total_tracks_seen)
    id_switch_rate    = id_switches / max(total_tracks, 1) * 100.0
    fps               = frames_processed / max(elapsed, 1e-6)

    # Per-frame detection stats
    dets_arr = np.array(per_frame_dets) if per_frame_dets else np.array([0])

    # DINO-specific internal stats (only available for SahiDinoDetector)
    det_stats = {}
    if hasattr(detector, "get_stats"):
        det_stats = detector.get_stats()

    return {
        "detector_name":         detector_name,
        "frames_processed":      frame_idx,
        "avg_dets_per_frame":    round(avg_dets, 2),
        "dets_std_per_frame":    round(float(dets_arr.std()), 2),
        "total_detections":      total_dets,
        "motorcycle_count":      motorcycle_count,
        "car_count":             car_count,
        "person_count":          person_count,
        "bus_count":             bus_count,
        "truck_count":           truck_count,
        "conf_mean":             round(float(conf_arr.mean()), 4),
        "conf_std":              round(float(conf_arr.std()), 4),
        "conf_min":              round(float(conf_arr.min()), 4),
        "conf_max":              round(float(conf_arr.max()), 4),
        "total_unique_tracks":   total_tracks,
        "id_switches":           id_switches,
        "id_switch_rate_pct":    round(id_switch_rate, 2),
        "fps":                   round(fps, 2),
        "total_time_s":          round(elapsed, 2),
        # DINO-specific (0 / N/A for RT-DETR)
        "tile_collision_rate_pct": round(det_stats.get("tile_collision_rate", 0.0), 2),
        "suppression_rate_pct":    round(det_stats.get("reduction_pct", 0.0), 2),
        "total_suppressed":        det_stats.get("total_suppressed", "N/A"),
    }


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def get_recommendation(
    rtdetr_metrics: Dict,
    dino_metrics: Dict,
    moto_improvement_threshold: float = 0.05,
    id_switch_improvement_threshold: float = 0.10,
    speed_tolerance: float = 0.20,
) -> str:
    """
    Produce a human-readable recommendation based on metric comparisons.

    Args:
        moto_improvement_threshold:    Min fractional improvement in motorcycle
                                       count required to recommend switching.
        id_switch_improvement_threshold: Min fractional reduction in ID switch
                                         rate required to recommend switching.
        speed_tolerance:               Max fractional speed degradation acceptable.
    """
    moto_rtdetr = max(rtdetr_metrics["motorcycle_count"], 1)
    moto_dino   = dino_metrics["motorcycle_count"]
    moto_gain   = (moto_dino - moto_rtdetr) / moto_rtdetr

    id_sw_rtdetr = max(rtdetr_metrics["id_switch_rate_pct"], 0.01)
    id_sw_dino   = dino_metrics["id_switch_rate_pct"]
    id_sw_gain   = (id_sw_rtdetr - id_sw_dino) / id_sw_rtdetr  # positive = reduction

    fps_rtdetr = max(rtdetr_metrics["fps"], 0.01)
    fps_dino   = dino_metrics["fps"]
    speed_loss = (fps_rtdetr - fps_dino) / fps_rtdetr  # positive = dino slower

    moto_ok  = moto_gain >= moto_improvement_threshold
    id_ok    = id_sw_gain >= id_switch_improvement_threshold
    speed_ok = speed_loss <= speed_tolerance

    lines: List[str] = []

    if moto_ok and id_ok and speed_ok:
        lines.append("✅  RECOMMENDATION: SWITCH TO SAHI+DINO")
        lines.append(f"   • Motorcycle detections: {moto_rtdetr} → {moto_dino} ({moto_gain:+.1%})")
        lines.append(f"   • ID switch rate: {id_sw_rtdetr:.1f}% → {id_sw_dino:.1f}% ({id_sw_gain:+.1%} reduction)")
        lines.append(f"   • Speed: {fps_rtdetr:.1f} → {fps_dino:.1f} FPS ({speed_loss:.1%} degradation — acceptable)")
    elif not speed_ok:
        lines.append("⚠️   RECOMMENDATION: BENCHMARK FURTHER (speed concern)")
        lines.append(f"   • DINO is {speed_loss:.1%} slower than RT-DETR (exceeds {speed_tolerance:.0%} tolerance)")
        if moto_ok or id_ok:
            lines.append("   • Detection quality is better but speed may be unacceptable for your use case")
    elif moto_ok and not id_ok:
        lines.append("🔶  RECOMMENDATION: CONDITIONAL SWITCH")
        lines.append(f"   • Motorcycle recall improved ({moto_gain:+.1%}) but ID switches did not improve enough")
        lines.append("   • Consider tuning tracker or running tune_sahi_dino.py first")
    elif id_ok and not moto_ok:
        lines.append("🔶  RECOMMENDATION: CONDITIONAL SWITCH")
        lines.append(f"   • ID switch rate improved ({id_sw_gain:+.1%}) but motorcycle recall did not")
        lines.append("   • Lower motorcycle_conf_thresh (try 0.12) and re-run")
    else:
        lines.append("❌  RECOMMENDATION: KEEP SAHI+RT-DETR")
        lines.append("   • Neither motorcycle recall nor ID switch rate improved significantly")
        lines.append("   • Try tune_sahi_dino.py to find better DINO parameters before concluding")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _delta(a, b, fmt=".2f") -> str:
    """Format a numeric delta with sign."""
    try:
        d = float(b) - float(a)
        return f"{d:+{fmt}}"
    except (TypeError, ValueError):
        return "N/A"


def print_comparison_report(rtdetr: Dict, dino: Dict) -> None:
    """Print a formatted comparison table to stdout."""
    sep = "─" * 72
    wide = "═" * 72

    print(f"\n{wide}")
    print("  COMPARISON REPORT: SAHI+RT-DETR vs SAHI+DINO (RT-DETRv2-X)")
    print(f"{wide}")
    print(f"  Video frames: {rtdetr['frames_processed']} (RT-DETR) / {dino['frames_processed']} (DINO)")
    print(f"{sep}")

    def row(label, key, fmt=".2f", suffix=""):
        rv, dv = rtdetr.get(key, "N/A"), dino.get(key, "N/A")
        dlt = _delta(rv, dv, fmt)
        rv_s = f"{rv:{fmt}}{suffix}" if isinstance(rv, (int, float)) else str(rv)
        dv_s = f"{dv:{fmt}}{suffix}" if isinstance(dv, (int, float)) else str(dv)
        print(f"  {label:<35} | {rv_s:>10} | {dv_s:>10} | {dlt:>8}")

    print(f"  {'Metric':<35} | {'RT-DETR':>10} | {'DINO':>10} | {'Δ':>8}")
    print(f"  {sep}")
    print("  DETECTION QUALITY")
    row("  Avg detections/frame",    "avg_dets_per_frame")
    row("  Det std (frame-to-frame)","dets_std_per_frame")
    row("  Motorcycle detections",   "motorcycle_count",  ".0f")
    row("  Car detections",          "car_count",         ".0f")
    row("  Confidence mean",         "conf_mean",         ".4f")
    row("  Confidence std (noise)",  "conf_std",          ".4f")
    print(f"  {sep}")
    print("  TRACKING PERFORMANCE")
    row("  Total unique track IDs",  "total_unique_tracks",".0f")
    row("  ID switches",             "id_switches",        ".0f")
    row("  ID switch rate (%)",      "id_switch_rate_pct", ".2f", "%")
    print(f"  {sep}")
    print("  WBF TILE FUSION (DINO only)")
    row("  Tile collision rate (%)", "tile_collision_rate_pct", ".2f", "%")
    row("  Suppression rate (%)",    "suppression_rate_pct",    ".2f", "%")
    print(f"  {sep}")
    print("  SPEED")
    row("  Processing FPS",          "fps",            ".2f")
    row("  Total time (s)",          "total_time_s",   ".1f")
    print(f"{sep}")
    print()
    print(get_recommendation(rtdetr, dino))
    print(f"\n{wide}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_detectors",
        description="Compare SAHI+RT-DETR vs SAHI+DINO on the same video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", "-i", required=True, help="Input video path.")
    p.add_argument("--max-frames",    type=int,   default=500,
                   help="Maximum frames to process per detector.")
    p.add_argument("--rtdetr-model",  default="rtdetr-l.pt",
                   help="RT-DETR-L weights file (current baseline).")
    p.add_argument("--dino-model",    default="rtdetr-x.pt",
                   help="DINO-style RT-DETRv2-X weights file.")
    p.add_argument("--rtdetr-conf",   type=float, default=0.15,
                   help="Global confidence threshold for RT-DETR.")
    p.add_argument("--dino-conf",     type=float, default=0.10,
                   help="Global confidence threshold for DINO.")
    p.add_argument("--rtdetr-slice",  type=int,   default=512,
                   help="Slice height/width for RT-DETR.")
    p.add_argument("--dino-slice",    type=int,   default=640,
                   help="Slice height/width for DINO.")
    p.add_argument("--overlap",       type=float, default=0.30,
                   help="SAHI overlap ratio (same for both detectors).")
    p.add_argument("--device",        default=None,
                   help="Inference device (cuda/cpu/mps). Auto-detected if None.")
    p.add_argument("--output-dir",    default="outputs/comparison",
                   help="Directory for output CSV and logs.")
    p.add_argument("--skip-rtdetr",   action="store_true",
                   help="Skip RT-DETR run (load from previous CSV if available).")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    video_path = str(Path(args.video))
    if not Path(video_path).exists():
        logger.error("Video not found: %s", video_path)
        sys.exit(1)

    device = args.device  # None = auto-detect inside each detector

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  SAHI+RT-DETR vs SAHI+DINO Comparison")
    print(f"{'='*60}")
    print(f"  Video      : {video_path}")
    print(f"  Max frames : {args.max_frames}")
    print(f"  RT-DETR    : {args.rtdetr_model} (slice={args.rtdetr_slice}, conf={args.rtdetr_conf})")
    print(f"  DINO       : {args.dino_model}   (slice={args.dino_slice}, conf={args.dino_conf})")
    print(f"  Output dir : {out_dir}")
    print()

    # ---- Run RT-DETR baseline -----------------------------------------------
    rtdetr_metrics: Dict = {}
    if not args.skip_rtdetr:
        print("▶ Running SAHI+RT-DETR baseline...")
        rtdetr_detector = SahiRTDetrDetector(
            model_path           = args.rtdetr_model,
            slice_height         = args.rtdetr_slice,
            slice_width          = args.rtdetr_slice,
            overlap_height_ratio = args.overlap,
            overlap_width_ratio  = args.overlap,
            conf                 = args.rtdetr_conf,
            device               = device,
        )
        rtdetr_metrics = run_detector_pipeline(
            video_path    = video_path,
            detector_name = "SAHI+RT-DETR-L",
            detector      = rtdetr_detector,
            device        = device or "cpu",
            max_frames    = args.max_frames,
        )
        print(f"  ✓ RT-DETR done: {rtdetr_metrics['motorcycle_count']} motos, "
              f"{rtdetr_metrics['id_switch_rate_pct']:.1f}% ID switches, "
              f"{rtdetr_metrics['fps']:.1f} FPS")
    else:
        logger.warning("--skip-rtdetr set; RT-DETR metrics will be zeros.")
        rtdetr_metrics = {"detector_name": "SAHI+RT-DETR-L (skipped)"}

    # ---- Run DINO -------------------------------------------------------
    print("\n▶ Running SAHI+DINO (RT-DETRv2-X)...")
    dino_detector = SahiDinoDetector(
        model_path           = args.dino_model,
        slice_height         = args.dino_slice,
        slice_width          = args.dino_slice,
        overlap_height_ratio = args.overlap,
        overlap_width_ratio  = args.overlap,
        conf                 = args.dino_conf,
        device               = device,
    )
    dino_metrics = run_detector_pipeline(
        video_path    = video_path,
        detector_name = "SAHI+DINO (RT-DETRv2-X)",
        detector      = dino_detector,
        device        = device or "cpu",
        max_frames    = args.max_frames,
    )
    print(f"  ✓ DINO done: {dino_metrics['motorcycle_count']} motos, "
          f"{dino_metrics['id_switch_rate_pct']:.1f}% ID switches, "
          f"{dino_metrics['fps']:.1f} FPS")

    # ---- Print comparison report --------------------------------------------
    if rtdetr_metrics and not args.skip_rtdetr:
        print_comparison_report(rtdetr_metrics, dino_metrics)
    else:
        print("\nDINO metrics:")
        for k, v in dino_metrics.items():
            print(f"  {k:<35}: {v}")

    # ---- Save CSV -----------------------------------------------------------
    csv_path = out_dir / "comparison.csv"
    all_metrics = []
    if rtdetr_metrics and not args.skip_rtdetr:
        all_metrics.append(rtdetr_metrics)
    all_metrics.append(dino_metrics)

    fieldnames = list(all_metrics[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)

    logger.info("Comparison CSV saved: %s", csv_path)
    print(f"\nResults saved to: {csv_path}\n")


if __name__ == "__main__":
    main()
