"""
generate_annotations.py — Offline Batch Annotation Generator

This script runs the SAHI + RT-DETR detector on an input video, exports
extracted frames and normalized YOLO-format annotation files, and optionally
saves an annotated preview video.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from sahi_rtdetr_detection import SahiRTDetrDetector, TRAFFIC_CLASSES
from utils import ensure_dir, draw_box_label, CLASS_COLORS, setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_annotations",
        description="Run SAHI + RT-DETR on a video to export frames and YOLO annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--input", "-i", required=True,
        help="Path to the input video file.",
    )
    p.add_argument(
        "--output-dir", "-o", default="data/annotations",
        help="Directory to save YOLO format annotation .txt files.",
    )
    p.add_argument(
        "--frames-dir", "-f", default="data/frames",
        help="Directory to save extracted frame images.",
    )
    p.add_argument(
        "--model", default="rtdetr-l.pt",
        help="RT-DETR model weights file path or name.",
    )
    p.add_argument(
        "--slice-height", type=int, default=640,
        help="SAHI slice height.",
    )
    p.add_argument(
        "--slice-width", type=int, default=640,
        help="SAHI slice width.",
    )
    p.add_argument(
        "--overlap-height-ratio", type=float, default=0.20,
        help="SAHI slice overlap height ratio.",
    )
    p.add_argument(
        "--overlap-width-ratio", type=float, default=0.20,
        help="SAHI slice overlap width ratio.",
    )
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold.",
    )
    p.add_argument(
        "--device", default=None,
        help="Inference device: 'cuda', 'cpu', 'mps', or '0' for GPU index.",
    )
    p.add_argument(
        "--preview-video", default="outputs/video/annotated_preview.mp4",
        help="Path to save the annotated preview video. Set to empty string to skip preview.",
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop processing after this many frames (useful for testing).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )

    # ---- Vehicle Class Postprocessing --------------------------------------
    p.add_argument("--truck-conf-thresh", type=float, default=0.55,
                   help="Confidence threshold for truck detections.")
    p.add_argument("--truck-min-area", type=float, default=8000.0,
                   help="Minimum bounding box area (in pixels) for trucks.")
    p.add_argument("--truck-min-width", type=float, default=120.0,
                   help="Minimum bounding box width (in pixels) for trucks.")
    p.add_argument("--truck-min-height", type=float, default=50.0,
                   help="Minimum bounding box height (in pixels) for trucks.")
    p.add_argument("--bus-conf-thresh", type=float, default=0.30,
                   help="Confidence threshold for bus detections.")
    p.add_argument("--car-conf-thresh", type=float, default=0.25,
                   help="Confidence threshold for car/relabel detections.")
    p.add_argument("--debug-trucks", action="store_true",
                   help="Enable saving cropped truck detections for manual debugging.")

    return p


def run_annotation(args: argparse.Namespace) -> None:
    # ---- Logging ------------------------------------------------------------
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # ---- Resolve Paths ------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input video not found: %s", input_path)
        sys.exit(1)

    annotations_dir = Path(args.output_dir)
    frames_dir = Path(args.frames_dir)
    ensure_dir(str(annotations_dir))
    ensure_dir(str(frames_dir))

    # ---- Open Video ---------------------------------------------------------
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", input_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)

    logger.info(
        "Processing Video: %s | %dx%d @ %.1f fps | %d frames to process",
        input_path.name, width, height, fps, total_frames,
    )

    # ---- Initialize Detector ------------------------------------------------
    detector = SahiRTDetrDetector(
        model_path=args.model,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        conf=args.conf,
        device=args.device,
    )

    # ---- Initialize Preview Video Writer ------------------------------------
    writer = None
    if args.preview_video:
        preview_path = Path(args.preview_video)
        ensure_dir(str(preview_path.parent))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(preview_path),
            fourcc,
            fps,
            (width, height),
        )
        logger.info("Saving preview video to: %s", preview_path)

    t_start = time.perf_counter()
    processed_count = 0

    postprocess_stats = {
        "car_detections": 0,
        "truck_detections": 0,
        "truck_to_car_relabels": 0,
        "truck_filtered_out": 0,
        "raw_truck_predictions": 0,
        "bus_detections": 0,
        "bus_filtered_out": 0,
        "person_detections": 0,
        "motorcycle_detections": 0,
    }

    try:
        with tqdm(total=total_frames, unit="frame", desc="Exporting Annotations") as pbar:
            while processed_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Run inference
                detections = detector.detect(frame)

                # Apply class-aware postprocessing
                from postprocess_vehicle_classes import postprocess_vehicle_detections
                detections = postprocess_vehicle_detections(
                    detections           = detections,
                    frame                = frame,
                    frame_idx            = processed_count,
                    truck_conf_thresh    = args.truck_conf_thresh,
                    truck_min_area       = args.truck_min_area,
                    truck_min_width      = args.truck_min_width,
                    truck_min_height     = args.truck_min_height,
                    bus_conf_thresh      = args.bus_conf_thresh,
                    car_conf_thresh      = args.car_conf_thresh,
                    debug_trucks         = args.debug_trucks,
                    stats                = postprocess_stats,
                )

                # 2. Save frame image
                frame_name = f"frame_{processed_count:06d}"
                frame_file = frames_dir / f"{frame_name}.jpg"
                cv2.imwrite(str(frame_file), frame)

                # 3. Save annotations (YOLO normalized format)
                annotation_file = annotations_dir / f"{frame_name}.txt"
                with open(annotation_file, "w", encoding="utf-8") as f:
                    for det in detections:
                        x1, y1, x2, y2 = det["bbox"]
                        class_id = det["class_id"]

                        # Compute YOLO normalized coordinates
                        x_center = (x1 + x2) / (2.0 * width)
                        y_center = (y1 + y2) / (2.0 * height)
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height

                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

                # 4. Save preview frame if enabled
                if writer is not None:
                    preview_frame = frame.copy()
                    for det in detections:
                        bbox = [int(val) for val in det["bbox"]]
                        class_name = det["class_name"]
                        confidence = det["confidence"]
                        color = CLASS_COLORS.get(class_name, (200, 200, 200))
                        label = f"{class_name} {confidence:.2f}"
                        draw_box_label(preview_frame, bbox, label, color)
                    writer.write(preview_frame)

                processed_count += 1
                pbar.update(1)
                pbar.set_postfix(dets=len(detections))

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - t_start
    
    raw_trucks = postprocess_stats.get("raw_truck_predictions", 0)
    trucks_filtered_out = postprocess_stats.get("truck_filtered_out", 0)
    trucks_relabeled = postprocess_stats.get("truck_to_car_relabels", 0)
    total_trucks_filtered = trucks_filtered_out + trucks_relabeled
    filtered_pct = (total_trucks_filtered / raw_trucks * 100.0) if raw_trucks > 0 else 0.0

    logger.info(
        "Finished exporting annotations in %.2f seconds (%.2f fps). Frames saved to %s, annotations saved to %s.",
        elapsed,
        processed_count / max(elapsed, 1e-6),
        frames_dir,
        annotations_dir,
    )

    print("\n" + "=" * 56)
    print("  POSTPROCESSING SUMMARY")
    print("=" * 56)
    print(f"  Car Detections:             {postprocess_stats.get('car_detections', 0)}")
    print(f"  Truck Detections:           {postprocess_stats.get('truck_detections', 0)}")
    print(f"  Truck->Car Relabels:        {postprocess_stats.get('truck_to_car_relabels', 0)}")
    print(f"  Truck Predictions Filtered: {filtered_pct:.2f}% (raw: {raw_trucks}, filtered: {trucks_filtered_out}, relabeled: {trucks_relabeled})")
    print("=" * 56 + "\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_annotation(args)


if __name__ == "__main__":
    main()
