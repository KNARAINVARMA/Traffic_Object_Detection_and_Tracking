"""
Export Module — Annotated Video + CSV Output

Responsibilities:
  1. Accumulate per-frame track records into an in-memory list.
  2. Compute per-track velocity estimates from consecutive world positions.
  3. Flush everything to a CSV file at end-of-video.
  4. Write a per-frame annotated BGR image to an output video file.
  5. Optionally overlay each track's recent trajectory as a coloured polyline.

CSV Schema
===========
  frame           — 0-based video frame index
  track_id        — unique persistent track identifier
  class_name      — "person" | "car" | "motorcycle" | "bus" | "truck"
  x1, y1, x2, y2 — bounding box corners in pixel coordinates
  center_x        — smoothed pixel column of box centre
  center_y        — smoothed pixel row of box centre
  world_x         — centre_x converted to metres (CoordinateMapper)
  world_y         — centre_y converted to metres
  confidence      — YOLO detection confidence (last matched detection)
  velocity_ms     — estimated speed in m/s (from consecutive world positions × fps)

Video Annotations
==================
  • Colour-coded bounding boxes (class-specific palette; see utils.py).
  • Label: "ID:<n> <class>" rendered above the box with a matching background.
  • Optional trajectory polyline showing the last N smoothed positions.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import CLASS_COLORS, draw_box_label

logger = logging.getLogger(__name__)

# CSV header matches the schema documented above
_CSV_HEADER = [
    "frame", "track_id", "class_name",
    "x1", "y1", "x2", "y2",
    "center_x", "center_y",
    "world_x", "world_y",
    "confidence", "velocity_ms",
]


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class Exporter:
    """
    Stateful exporter that ingests per-frame track data and writes outputs.

    Usage::

        exporter = Exporter(fps=25, output_video_path="out.mp4",
                            output_csv_path="out.csv", frame_size=(1920, 1080))
        for frame_idx, (frame, tracks) in enumerate(pipeline):
            annotated = exporter.process_frame(frame_idx, frame, tracks)
            # (annotated already written to video internally)
        exporter.close()
    """

    def __init__(
        self,
        fps:               float,
        output_video_path: Optional[str],
        output_csv_path:   str,
        frame_size:        Tuple[int, int],   # (width, height)
        draw_trajectories: bool  = True,
        trajectory_length: int   = 30,        # frames of history to display
    ) -> None:
        self.fps               = fps
        self.draw_trajectories = draw_trajectories
        self.trajectory_length = trajectory_length

        # ------- CSV writer --------------------------------------------------
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._csv_file   = open(output_csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_HEADER)
        self._csv_writer.writeheader()
        logger.info("CSV output: %s", output_csv_path)

        # ------- Video writer ------------------------------------------------
        self._video_writer: Optional[cv2.VideoWriter] = None
        if output_video_path:
            Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, frame_size
            )
            logger.info("Video output: %s", output_video_path)

        # ------- Per-track state for velocity and trajectory -----------------
        # {track_id: (prev_world_x, prev_world_y)}
        self._prev_world: Dict[int, Tuple[float, float]] = {}
        # {track_id: deque[(cx, cy)]}  — smoothed pixel positions for trajectory
        self._traj_buffers: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=trajectory_length)
        )

        # Summary counters
        self._total_rows = 0

    # ---- main per-frame API -------------------------------------------------

    def process_frame(
        self,
        frame_idx: int,
        frame:     np.ndarray,
        tracks:    List[Dict],
    ) -> np.ndarray:
        """
        Write CSV rows and produce an annotated copy of *frame*.

        Args:
            frame_idx: 0-based index of the current video frame.
            frame:     Raw BGR image (H × W × 3).
            tracks:    List of track dicts from BYTETracker.update() after
                       smoothing / coordinate mapping is applied.
                       Required keys:
                         track_id, bbox, center, class_name,
                         confidence, world_x, world_y, smoothed_cx, smoothed_cy

        Returns:
            Annotated BGR image (same size as *frame*).
        """
        annotated = frame.copy()

        for t in tracks:
            tid        = t["track_id"]
            x1, y1, x2, y2 = t["bbox"]
            s_cx, s_cy = t["smoothed_cx"], t["smoothed_cy"]
            wx, wy     = t["world_x"], t["world_y"]
            conf       = t["confidence"]
            cls_name   = t["class_name"]

            # ---- velocity ---------------------------------------------------
            vel_ms = 0.0
            if tid in self._prev_world:
                pwx, pwy = self._prev_world[tid]
                dist = np.hypot(wx - pwx, wy - pwy)
                vel_ms = float(dist * self.fps)
            self._prev_world[tid] = (wx, wy)

            # ---- CSV row ----------------------------------------------------
            row = {
                "frame":      frame_idx,
                "track_id":   tid,
                "class_name": cls_name,
                "x1":         round(x1, 2),
                "y1":         round(y1, 2),
                "x2":         round(x2, 2),
                "y2":         round(y2, 2),
                "center_x":   round(s_cx, 2),
                "center_y":   round(s_cy, 2),
                "world_x":    round(wx, 4),
                "world_y":    round(wy, 4),
                "confidence": round(conf, 4),
                "velocity_ms": round(vel_ms, 4),
            }
            self._csv_writer.writerow(row)
            self._total_rows += 1

            # ---- trajectory buffer ------------------------------------------
            self._traj_buffers[tid].append((int(s_cx), int(s_cy)))

            # ---- draw trajectory polyline -----------------------------------
            if self.draw_trajectories:
                pts = list(self._traj_buffers[tid])
                color = CLASS_COLORS.get(cls_name, (200, 200, 200))
                for k in range(1, len(pts)):
                    alpha = k / len(pts)  # fade older points
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(annotated, pts[k - 1], pts[k], faded, 1, cv2.LINE_AA)

            # ---- draw bounding box and label --------------------------------
            draw_box_label(
                annotated,
                bbox      = [int(x1), int(y1), int(x2), int(y2)],
                label     = f"ID:{tid} {cls_name}",
                color     = CLASS_COLORS.get(cls_name, (200, 200, 200)),
            )

        # Write annotated frame to video
        if self._video_writer is not None:
            self._video_writer.write(annotated)

        return annotated

    # ---- cleanup ------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all open output handles."""
        self._csv_file.flush()
        self._csv_file.close()
        logger.info("CSV closed — %d rows written.", self._total_rows)

        if self._video_writer is not None:
            self._video_writer.release()
            logger.info("Video writer released.")

    def __enter__(self) -> "Exporter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
