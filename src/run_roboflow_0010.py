"""
Roboflow detection + tracking on DJI_20250124130311_0010_D.MP4
Produces:
  - roboflow_DJI_0010_tracked.mp4   (stabilized, same visual style)
  - roboflow_DJI_0010_tracks.csv    (for safety analysis)
"""
from __future__ import annotations

import csv
import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Tuple

import numpy as np
from inference.models.utils import get_roboflow_model
from tqdm import tqdm
import supervision as sv

API_KEY   = "MhcETJfdaYzdKCUFPzjC"
MODEL_ID  = "vehicle-count-in-drone-video/6"
SOURCE    = "/Users/suyashagarwal/Desktop/btp/DJI_20250124130311_0010_D.MP4"
OUT_VIDEO = "/Users/suyashagarwal/Desktop/btp/traffic_tracking/outputs/video/roboflow_DJI_0010_tracked.mp4"
OUT_CSV   = "/Users/suyashagarwal/Desktop/btp/traffic_tracking/outputs/csv/roboflow_DJI_0010_tracks.csv"
SCALE_MPX = 0.05
CONF      = 0.3
IOU       = 0.7
SMOOTH_W  = 7

COLOR = sv.Color.from_hex("#3C76D1")

Path(OUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)


class BoxSmoother:
    def __init__(self, window: int = SMOOTH_W):
        self.window = window
        self._bufs: Dict[int, Deque[Tuple[float,float,float,float]]] = {}

    def update(self, tid: int, x1: float, y1: float, x2: float, y2: float
               ) -> Tuple[float, float, float, float]:
        if tid not in self._bufs:
            self._bufs[tid] = deque(maxlen=self.window)
        self._bufs[tid].append((x1, y1, x2, y2))
        arr = np.array(self._bufs[tid])
        return tuple(arr.mean(axis=0))


model      = get_roboflow_model(model_id=MODEL_ID, api_key=API_KEY)
tracker    = sv.ByteTrack()
smoother   = BoxSmoother(SMOOTH_W)
video_info = sv.VideoInfo.from_video_path(SOURCE)

box_annotator   = sv.BoxAnnotator(color=sv.ColorPalette([COLOR]))
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette([COLOR]), text_color=sv.Color.WHITE
)
trace_annotator = sv.TraceAnnotator(
    color=sv.ColorPalette([COLOR]),
    position=sv.Position.CENTER,
    trace_length=100,
    thickness=2,
)

frame_generator = sv.get_video_frames_generator(source_path=SOURCE)
prev_world: Dict[int, Tuple[float, float]] = {}
csv_rows = []
frame_idx = 0

with sv.VideoSink(OUT_VIDEO, video_info) as sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Roboflow DJI_0010"):
        results    = model.infer(frame, confidence=CONF, iou_threshold=IOU)[0]
        detections = sv.Detections.from_inference(results)

        class_names = []
        if hasattr(results, "predictions"):
            for pred in results.predictions:
                class_names.append(getattr(pred, "class_name", "vehicle").lower())
        if len(class_names) != len(detections):
            class_names = ["vehicle"] * len(detections)

        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            smoothed_xyxy = np.array(detections.xyxy, dtype=float)
            for k, tid in enumerate(detections.tracker_id):
                sx1, sy1, sx2, sy2 = smoother.update(int(tid), *detections.xyxy[k])
                smoothed_xyxy[k] = [sx1, sy1, sx2, sy2]

            for k, tid in enumerate(detections.tracker_id):
                x1, y1, x2, y2 = smoothed_xyxy[k]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                wx = cx * SCALE_MPX
                wy = cy * SCALE_MPX
                vel = 0.0
                if tid in prev_world:
                    pwx, pwy = prev_world[tid]
                    vel = math.hypot(wx - pwx, wy - pwy) * video_info.fps
                prev_world[int(tid)] = (wx, wy)

                cn = class_names[k] if k < len(class_names) else "vehicle"
                csv_rows.append({
                    "frame": frame_idx, "track_id": int(tid),
                    "class_name": cn,
                    "x1": round(x1, 2), "y1": round(y1, 2),
                    "x2": round(x2, 2), "y2": round(y2, 2),
                    "center_x": round(cx, 2), "center_y": round(cy, 2),
                    "world_x": round(wx, 4), "world_y": round(wy, 4),
                    "confidence": round(float(detections.confidence[k])
                                        if detections.confidence is not None else 0.0, 4),
                    "velocity_ms": round(vel, 4),
                })

            draw_det = sv.Detections(
                xyxy=smoothed_xyxy,
                tracker_id=detections.tracker_id,
                confidence=detections.confidence,
                class_id=detections.class_id,
            )
        else:
            draw_det = detections

        labels = [f"#{tid}" for tid in (draw_det.tracker_id if draw_det.tracker_id is not None else [])]
        annotated = frame.copy()
        annotated = trace_annotator.annotate(annotated, draw_det)
        annotated = box_annotator.annotate(annotated, draw_det)
        annotated = label_annotator.annotate(annotated, draw_det, labels)
        sink.write_frame(annotated)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  {frame_idx}/{video_info.total_frames} frames …")

fields = ["frame","track_id","class_name","x1","y1","x2","y2",
          "center_x","center_y","world_x","world_y","confidence","velocity_ms"]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_rows)

unique_ids = len({r["track_id"] for r in csv_rows})
print(f"\nDone.")
print(f"  Video : {OUT_VIDEO}")
print(f"  CSV   : {OUT_CSV}")
print(f"  Frames: {frame_idx}  |  Unique tracks: {unique_ids}  |  Total detections: {len(csv_rows)}")
