"""
Traffic Conflict Safety Analysis
---------------------------------
Usage:
    python -m safety.main_safety \\
        --csv  outputs/csv/DJI_0010_tracks_v2.csv \\
        --video outputs/video/DJI_0010_tracked_v2.mp4 \\
        --output-dir outputs/safety/

Optional tuning flags (all have sensible defaults):
    --delta       lookahead frames        [default: 25]
    --proximity   proximity threshold m   [default: 15.0]
    --min-angle   heading diff threshold° [default: 30.0]
    --preview-zone  draw zone on first frame and exit (for calibration)
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def _preview_zone(video_path: str) -> None:
    from .zone import INTERSECTION_ZONE_PX
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read first frame.")
        return
    pts = np.array(INTERSECTION_ZONE_PX, dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (0, 200, 0))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [pts], True, (0, 200, 0), 3)
    for p in INTERSECTION_ZONE_PX:
        cv2.circle(frame, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
    out_path = os.path.join(os.path.dirname(video_path), "zone_preview.png")
    cv2.imwrite(out_path, frame)
    print(f"Zone preview saved → {out_path}")
    print("Adjust INTERSECTION_ZONE_PX in zone.py if needed, then re-run without --preview-zone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic Conflict Safety Analysis")
    parser.add_argument("--csv", required=True, help="Path to tracking CSV")
    parser.add_argument("--video", required=True, help="Path to tracked video")
    parser.add_argument("--output-dir", default="outputs/safety/", help="Output directory")
    parser.add_argument("--delta", type=int, default=125, help="Lookahead frames (default 125 = 5 sec)")
    parser.add_argument("--proximity", type=float, default=20.0, help="Proximity threshold in metres")
    parser.add_argument("--min-angle", type=float, default=30.0, help="Min heading difference in degrees")
    parser.add_argument("--min-track-length", type=int, default=100,
                        help="Min frames a track must exist to be considered (default 100 = 4 sec)")
    parser.add_argument("--override-min-track-length", type=int, default=None,
                        help="Override per-class minimums and use this value for all classes")
    parser.add_argument("--preview-zone", action="store_true",
                        help="Save a zone overlay image for calibration and exit")
    parser.add_argument("--detect-unsafe-shortcuts", action="store_true",
                        help="Run unsafe roundabout shortcut detection and save violations CSV")
    parser.add_argument("--unsafe-shortcut-output", default=None,
                        help="Path to write unsafe shortcut violations CSV")
    parser.add_argument("--detect-unsafe-overtaking", action="store_true",
                        help="Run unsafe overtaking detection and save violations CSV")
    parser.add_argument("--unsafe-overtaking-output", default=None,
                        help="Path to write unsafe overtaking violations CSV")
    parser.add_argument("--annotate-unsafe-overtaking", action="store_true",
                        help="Annotate unsafe overtaking violations on the output video")
    parser.add_argument("--unsafe-overtaking-video-output", default=None,
                        help="Path to write unsafe overtaking annotated video")
    args = parser.parse_args()

    if args.preview_zone:
        _preview_zone(args.video)
        sys.exit(0)

    from .conflict_detector import load_tracks, detect_conflicts, MIN_TRACK_LENGTH_BY_CLASS
    from .aggressor_classifier import classify_aggressor
    from .safety_reporter import write_conflict_csv, annotate_video, annotate_overtaking_video
    from .unsafe_roundabout_shortcut_rule import detect_unsafe_roundabout_shortcuts
    from .unsafe_overtaking_rule import detect_unsafe_overtaking

    if args.override_min_track_length is not None:
        for k in list(MIN_TRACK_LENGTH_BY_CLASS.keys()):
            MIN_TRACK_LENGTH_BY_CLASS[k] = args.override_min_track_length

    video_stem = os.path.splitext(os.path.basename(args.video))[0]
    csv_out = os.path.join(args.output_dir, f"{video_stem}_conflict_events.csv")
    video_out = os.path.join(args.output_dir, f"{video_stem}_conflict_annotated.mp4")

    print(f"[main] Loading tracks from {args.csv} …")
    tracks = load_tracks(args.csv)
    print(f"[main] Loaded {len(tracks)} unique track IDs")

    print(f"[main] Detecting conflicts  (delta={args.delta} frames, proximity={args.proximity}m, "
          f"min_angle={args.min_angle}°, min_track_length={args.min_track_length} frames) …")
    conflicts = detect_conflicts(tracks, args.delta, args.proximity, args.min_angle, args.min_track_length)
    print(f"[main] {len(conflicts)} conflict events detected")

    if not conflicts:
        print("[main] No conflicts found. Try adjusting --proximity or --delta.")
        sys.exit(0)

    print("[main] Classifying aggressors …")
    results = [classify_aggressor(e, tracks, args.delta) for e in conflicts]

    agreed = sum(1 for r in results if r.methods_agree)
    print(f"[main] Angular + TTC agreement: {agreed}/{len(results)} "
          f"({100*agreed//len(results)}%)")

    write_conflict_csv(results, csv_out)
    annotate_video(results, tracks, args.video, video_out, args.delta)

    if args.detect_unsafe_shortcuts:
        unsafe_out = args.unsafe_shortcut_output or os.path.join(
            args.output_dir, f"{video_stem}_unsafe_shortcut_violations.csv"
        )
        print(f"[main] Running unsafe roundabout shortcut detection → {unsafe_out}")
        detect_unsafe_roundabout_shortcuts(args.csv, unsafe_out)

    if args.detect_unsafe_overtaking:
        overtaking_out = args.unsafe_overtaking_output or os.path.join(
            args.output_dir, f"{video_stem}_unsafe_overtaking_violations.csv"
        )
        print(f"[main] Running unsafe overtaking detection → {overtaking_out}")
        detect_unsafe_overtaking(args.csv, overtaking_out)

        if args.annotate_unsafe_overtaking:
            overtaking_video_out = args.unsafe_overtaking_video_output or os.path.join(
                args.output_dir, f"{video_stem}_unsafe_overtaking_annotated.mp4"
            )
            print(f"[main] Annotating unsafe overtaking video → {overtaking_video_out}")
            annotate_overtaking_video(overtaking_out, tracks, args.video, overtaking_video_out)

    print("\n[main] === Summary ===")
    print(f"  Total conflicts   : {len(results)}")
    print(f"  Methods agree     : {agreed}")
    print(f"  CSV               : {csv_out}")
    print(f"  Video             : {video_out}")
    if args.detect_unsafe_shortcuts:
        print(f"  Unsafe shortcuts  : {unsafe_out}")
    if args.detect_unsafe_overtaking:
        print(f"  Unsafe overtaking: {overtaking_out}")
    if args.detect_unsafe_overtaking and args.annotate_unsafe_overtaking:
        print(f"  Unsafe overtaking video: {overtaking_video_out}")


if __name__ == "__main__":
    main()
