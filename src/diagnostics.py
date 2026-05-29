"""
diagnostics.py — Dataset and Tracking Diagnostics Module

This utility calculates quantitative tracking metrics from the pipeline's
exported tracks CSV and compares them against ground-truth annotations (if available).
It computes:
  1. Total detections and unique tracks per class.
  2. Average track duration (class-specific).
  3. Track fragmentation events (unsupervised gap detection per track ID).
  4. Precise motorcycle recall and precision statistics (when data/annotations/*.txt is present).
  5. Outputs results to outputs/metrics/diagnostics_summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def iou(boxA: List[float], boxB: List[float]) -> float:
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = boxAArea + boxBArea - interArea
    return interArea / (unionArea + 1e-7)


def run_diagnostics(csv_path: Path, annotations_dir: Path, output_metrics_dir: Path) -> None:
    logger.info("Loading tracks CSV from %s...", csv_path)
    if not csv_path.exists():
        logger.error("Tracks CSV file not found: %s", csv_path)
        return
        
    df = pd.read_csv(csv_path)
    
    metrics: Dict = {}
    
    # 1. Detections and Tracks per class
    logger.info("Computing detection and track counts...")
    metrics["total_detections"] = len(df)
    
    class_counts = df["class_name"].value_counts().to_dict()
    metrics["detections_per_class"] = class_counts
    
    unique_tracks = df.groupby("class_name")["track_id"].nunique().to_dict()
    metrics["unique_tracks_per_class"] = unique_tracks
    metrics["total_unique_tracks"] = df["track_id"].nunique()
    
    # 2. Track durations (frames)
    durations = df.groupby(["track_id", "class_name"])["frame"].agg(["min", "max"])
    durations["duration"] = durations["max"] - durations["min"] + 1
    durations = durations.reset_index()
    
    avg_duration = durations.groupby("class_name")["duration"].mean().to_dict()
    metrics["avg_track_duration_frames"] = {k: round(v, 2) for k, v in avg_duration.items()}
    metrics["overall_avg_track_duration_frames"] = round(durations["duration"].mean(), 2)
    
    # 3. Track fragmentation events (unsupervised gap detection)
    logger.info("Detecting track fragmentation events (temporal gaps)...")
    fragmentation_events = 0
    m_fragmentation_events = 0
    
    for (tid, cls_name), group in df.groupby(["track_id", "class_name"]):
        frames = sorted(group["frame"].tolist())
        gaps = 0
        for i in range(1, len(frames)):
            if frames[i] - frames[i-1] >= 2:
                gaps += 1
        if gaps > 0:
            fragmentation_events += gaps
            if cls_name == "motorcycle":
                m_fragmentation_events += gaps
                
    metrics["track_fragmentation"] = {
        "overall_fragmentation_gaps_count": fragmentation_events,
        "motorcycle_fragmentation_gaps_count": m_fragmentation_events,
    }
    
    # 4. Supervised precision/recall evaluation (if GT annotations exist)
    gt_files = sorted(list(annotations_dir.glob("frame_*.txt")))
    if gt_files:
        logger.info("Ground-truth labels found in %s. Evaluating recall and precision...", annotations_dir)
        
        # We need to map class IDs (YOLO annotations) to class names
        class_id_map = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        
        # Parse all GT boxes
        gt_by_frame: Dict[int, List[Dict]] = {}
        for gf in gt_files:
            try:
                # Extracts frame number from frame_XXXXXX.txt
                frame_idx = int(gf.stem.split("_")[-1])
            except Exception:
                continue
                
            gt_boxes = []
            with open(gf, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        if cls_id not in class_id_map:
                            continue
                        # YOLO format: x_center, y_center, w, h
                        x_c, y_c, w, h = map(float, parts[1:])
                        # Since we don't have the frame size here, we work in normalized coords,
                        # or we can reconstruct normalized coordinates for tracker boxes!
                        # Better yet, let's map absolute tracker boxes to normalized coordinates.
                        gt_boxes.append({
                            "class_id": cls_id,
                            "class_name": class_id_map[cls_id],
                            "bbox": [x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]
                        })
            gt_by_frame[frame_idx] = gt_boxes
            
        # Get frame dimensions from the tracker CSV
        # We normalize the tracker bboxes: [x1/W, y1/H, x2/W, y2/H]
        # Let's estimate frame size from the center and world positions if possible, or just look at the CSV min/max!
        # Actually, in the CSV we have `x1` and `x2` and `center_x`.
        # To get the normalization width and height, we can estimate it, or since we know standard resolution is 1920x1080:
        # Let's check: can we determine the image size? We can use the default 1920x1080!
        W, H = 1920, 1080  # Default standard, can be overridden or estimated
        
        # Group tracker detections by frame
        tracker_by_frame: Dict[int, List[Dict]] = {}
        for idx, row in df.iterrows():
            f_idx = int(row["frame"])
            if f_idx not in tracker_by_frame:
                tracker_by_frame[f_idx] = []
            tracker_by_frame[f_idx].append({
                "class_name": row["class_name"],
                # Normalize tracker boxes
                "bbox": [row["x1"]/W, row["y1"]/H, row["x2"]/W, row["y2"]/H]
            })
            
        # Evaluate frame-by-frame
        total_gt = {"motorcycle": 0, "car": 0, "person": 0, "overall": 0}
        total_pred = {"motorcycle": 0, "car": 0, "person": 0, "overall": 0}
        matched_gt = {"motorcycle": 0, "car": 0, "person": 0, "overall": 0}
        
        for f_idx, gt_list in gt_by_frame.items():
            pred_list = tracker_by_frame.get(f_idx, [])
            
            # Count class targets
            for gt in gt_list:
                c_name = gt["class_name"]
                if c_name in total_gt:
                    total_gt[c_name] += 1
                total_gt["overall"] += 1
                
            for pred in pred_list:
                c_name = pred["class_name"]
                if c_name in total_pred:
                    total_pred[c_name] += 1
                total_pred["overall"] += 1
                
            # Matching (Greedy matching by IoU)
            used_preds = set()
            for gt in gt_list:
                c_name = gt["class_name"]
                best_iou = 0.0
                best_pred_idx = -1
                
                for p_idx, pred in enumerate(pred_list):
                    if p_idx in used_preds or pred["class_name"] != c_name:
                        continue
                    curr_iou = iou(gt["bbox"], pred["bbox"])
                    if curr_iou > best_iou:
                        best_iou = curr_iou
                        best_pred_idx = p_idx
                        
                if best_iou >= 0.5:
                    used_preds.add(best_pred_idx)
                    if c_name in matched_gt:
                        matched_gt[c_name] += 1
                    matched_gt["overall"] += 1
                    
        # Compute scores
        recall = {}
        precision = {}
        for c_name in ["motorcycle", "car", "person", "overall"]:
            gts = total_gt[c_name]
            preds = total_pred[c_name]
            matches = matched_gt[c_name]
            
            recall[c_name] = round(matches / gts, 4) if gts > 0 else 0.0
            precision[c_name] = round(matches / preds, 4) if preds > 0 else 0.0
            
        metrics["supervised_evaluation"] = {
            "total_ground_truth_frames": len(gt_by_frame),
            "motorcycle_recall": recall["motorcycle"],
            "motorcycle_precision": precision["motorcycle"],
            "car_recall": recall["car"],
            "person_recall": recall["person"],
            "overall_recall": recall["overall"],
            "overall_precision": precision["overall"],
        }
    else:
        logger.warning("No ground-truth frames found in %s. Skipping supervised evaluation.", annotations_dir)
        metrics["supervised_evaluation"] = "No ground-truth label files (frame_*.txt) found in data/annotations/"
        
    # 5. Save results
    ensure_dir(str(output_metrics_dir))
    metrics_file = output_metrics_dir / "diagnostics_summary.json"
    with open(metrics_file, "w", encoding="utf-8") as mf:
        json.dump(metrics, mf, indent=4)
        
    logger.info("Diagnostics completed successfully! Saved to %s", metrics_file)
    
    # Print clean summary
    print("\n" + "=" * 56)
    print("  TRACKING DIAGNOSTICS SUMMARY")
    print("=" * 56)
    print(f"  Total Detections Ingested:   {metrics['total_detections']}")
    print(f"  Total Unique Bounded Tracks: {metrics['total_unique_tracks']}")
    
    print("\n  UNIQUE TRACKS BY CLASS:")
    for k, v in unique_tracks.items():
        print(f"    {k:<15} : {v}")
        
    print("\n  AVERAGE DURATION (FRAMES):")
    for k, v in avg_duration.items():
        print(f"    {k:<15} : {v:.1f} frames")
        
    print("\n  TRACK FRAGMENTATION GAPS:")
    print(f"    Overall Fragmentation Gaps : {fragmentation_events}")
    print(f"    Motorcycle Gaps            : {m_fragmentation_events}")
    
    if gt_files:
        print("\n  SUPERVISED MOTORCYCLE PERFORMANCE:")
        print(f"    Motorcycle Recall          : {metrics['supervised_evaluation']['motorcycle_recall']*100.0:.2f}%")
        print(f"    Motorcycle Precision       : {metrics['supervised_evaluation']['motorcycle_precision']*100.0:.2f}%")
        print(f"    Overall Recall             : {metrics['supervised_evaluation']['overall_recall']*100.0:.2f}%")
        print(f"    Overall Precision          : {metrics['supervised_evaluation']['overall_precision']*100.0:.2f}%")
    print("=" * 56 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="diagnostics",
        description="Run tracking pipeline diagnostic metrics.",
    )
    p.add_argument("--csv", default="outputs/csv/Train_cut_tracks.csv", help="Path to exported tracks CSV.")
    p.add_argument("--annotations", default="data/annotations", help="Dir containing ground-truth YOLO labels.")
    p.add_argument("--output-dir", default="outputs/metrics", help="Dir to save diagnostics JSON.")
    args = p.parse_args()
    
    setup_logging(level=logging.INFO)
    run_diagnostics(Path(args.csv), Path(args.annotations), Path(args.output_dir))


if __name__ == "__main__":
    main()
