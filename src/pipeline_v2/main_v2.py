"""
Pipeline V2 — BoT-SORT tracking with ultralytics built-in tracker.

Improvements over V1:
  - BoT-SORT replaces custom ByteTrack: adds global motion compensation
    (sparseOptFlow) which handles drone camera sway and produces longer,
    more stable track IDs.
  - Lower confidence threshold (0.15) catches more small vehicles.
  - Longer track buffer (60 frames) reduces fragmentation.

Usage:
    cd traffic_tracking/src
    python -m pipeline_v2.main_v2 \
        --input  ../../DJI_20250124130311_0010_D.MP4 \
        --output-video ../../outputs_v2/video/DJI_0010_v2_tracked.mp4 \
        --output-csv   ../../outputs_v2/csv/DJI_0010_v2_tracks.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import deque
from pathlib import Path
from typing import Dict, Deque, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

SCALE_MPX: float = 0.05          # metres per pixel (same as V1)
SMOOTH_WINDOW: int = 7           # moving-average window for jitter reduction
TRAJECTORY_LEN: int = 40         # frames of trail to draw per vehicle

TRACKER_CFG = os.path.join(os.path.dirname(__file__), "botsort_drone.yaml")

# colour per class (BGR)
CLASS_COLOURS = {
    "car":        (0, 200, 255),
    "motorcycle": (0, 255, 128),
    "truck":      (255, 100, 0),
    "bus":        (255, 0, 200),
    "person":     (200, 200, 200),
}
DEFAULT_COLOUR = (180, 180, 180)


# ---------------------------------------------------------------------------
# Simple moving-average smoother
# ---------------------------------------------------------------------------

class Smoother:
    def __init__(self, window: int = SMOOTH_WINDOW):
        self.window = window
        self._buf: Dict[int, Deque[Tuple[float, float]]] = {}

    def update(self, tid: int, cx: float, cy: float) -> Tuple[float, float]:
        if tid not in self._buf:
            self._buf[tid] = deque(maxlen=self.window)
        self._buf[tid].append((cx, cy))
        xs = [p[0] for p in self._buf[tid]]
        ys = [p[1] for p in self._buf[tid]]
        return sum(xs) / len(xs), sum(ys) / len(ys)


# ---------------------------------------------------------------------------
# Velocity estimator (m/s from world-coordinate deltas)
# ---------------------------------------------------------------------------

class VelocityEstimator:
    def __init__(self, fps: float):
        self.fps = fps
        self._prev: Dict[int, Tuple[float, float]] = {}

    def update(self, tid: int, wx: float, wy: float) -> float:
        if tid in self._prev:
            px, py = self._prev[tid]
            dist = math.hypot(wx - px, wy - py)
            speed = dist * self.fps
        else:
            speed = 0.0
        self._prev[tid] = (wx, wy)
        return speed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)),   exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    smoother  = Smoother(SMOOTH_WINDOW)
    vel_est   = VelocityEstimator(fps)
    trails: Dict[int, deque] = {}

    csv_rows = []

    results_gen = model.track(
        source      = args.input,
        tracker     = TRACKER_CFG,
        conf        = 0.15,
        iou         = 0.50,
        imgsz       = 1280,
        stream      = True,
        persist     = True,
        verbose     = False,
    )

    frame_idx = 0
    unique_ids = set()

    with tqdm(total=total, unit="frame", desc="V2 Tracking") as pbar:
        for result in results_gen:
            img = result.orig_img.copy()

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids    = boxes.id.cpu().numpy().astype(int)
                xyxys  = boxes.xyxy.cpu().numpy()
                clss   = boxes.cls.cpu().numpy().astype(int)
                confs  = boxes.conf.cpu().numpy()

                for tid, xyxy, cls_idx, conf in zip(ids, xyxys, clss, confs):
                    x1, y1, x2, y2 = xyxy
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    class_name = model.names[cls_idx]

                    s_cx, s_cy = smoother.update(tid, cx, cy)
                    wx = s_cx * SCALE_MPX
                    wy = s_cy * SCALE_MPX
                    speed = vel_est.update(tid, wx, wy)

                    unique_ids.add(tid)

                    csv_rows.append({
                        "frame":      frame_idx,
                        "track_id":   tid,
                        "class_name": class_name,
                        "confidence": round(float(conf), 3),
                        "x1": round(float(x1), 1),
                        "y1": round(float(y1), 1),
                        "x2": round(float(x2), 1),
                        "y2": round(float(y2), 1),
                        "center_x":   round(float(s_cx), 2),
                        "center_y":   round(float(s_cy), 2),
                        "world_x":    round(float(wx), 4),
                        "world_y":    round(float(wy), 4),
                        "velocity_ms": round(float(speed), 4),
                    })

                    # Trail
                    if tid not in trails:
                        trails[tid] = deque(maxlen=TRAJECTORY_LEN)
                    trails[tid].append((int(s_cx), int(s_cy)))

                    # Draw trail
                    colour = CLASS_COLOURS.get(class_name, DEFAULT_COLOUR)
                    pts = list(trails[tid])
                    for k in range(1, len(pts)):
                        alpha = k / len(pts)
                        c = tuple(int(v * alpha) for v in colour)
                        cv2.line(img, pts[k-1], pts[k], c, 1)

                    # Bounding box + label
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
                    label = f"#{tid} {class_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    ty = max(int(y1) - 4, 12)
                    cv2.rectangle(img, (int(x1), ty - th - 2), (int(x1) + tw + 2, ty + 2), colour, -1)
                    cv2.putText(img, label, (int(x1) + 1, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(img, f"frame {frame_idx}  |  ids: {len(unique_ids)}",
                        (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            writer.write(img)

            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix(ids=len(unique_ids))

    writer.release()

    # Write CSV
    fields = ["frame","track_id","class_name","confidence",
              "x1","y1","x2","y2","center_x","center_y",
              "world_x","world_y","velocity_ms"]
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)

    print(f"\n[V2] Done.")
    print(f"[V2] Frames processed : {frame_idx}")
    print(f"[V2] Unique track IDs : {len(unique_ids)}")
    print(f"[V2] CSV rows         : {len(csv_rows)}")
    print(f"[V2] Video            : {args.output_video}")
    print(f"[V2] CSV              : {args.output_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline V2 — BoT-SORT tracking")
    p.add_argument("--input",        required=True)
    p.add_argument("--output-video", default="../../outputs_v2/video/tracked_v2.mp4")
    p.add_argument("--output-csv",   default="../../outputs_v2/csv/tracks_v2.csv")
    p.add_argument("--model",        default="yolov8m.pt")
    run(p.parse_args())


if __name__ == "__main__":
    main()
