"""
Trajectory Smoothing Module

WHY SMOOTHING IS CRITICAL FOR SAFETY ANALYSIS
===============================================
Even after Kalman-filter based tracking, the *output* coordinates still carry
frame-to-frame jitter caused by:
  • Detection bounding-box instability (YOLO's output varies slightly per frame
    even for a stationary object).
  • Quantisation noise from integer pixel rounding.
  • Brief partial occlusions that shift the box by a few pixels.

For safety-rule logic, jitter translates directly into:
  1. Spurious velocity spikes — an object that is actually stationary may appear
     to jump 5 pixels between frames, which at a scale of 0.05 m/px × 25 fps
     gives a phantom velocity of 6.25 m/s (~22 km/h).  This would trigger false
     "speeding" or "sudden acceleration" alarms.
  2. Erratic trajectory curves — the direction-of-travel vector becomes noisy,
     making lane-change or wrong-way detection unreliable.
  3. Incorrect TTC estimates — Time-To-Collision computed from jittery
     positions produces wide confidence intervals.

Smoothing replaces each coordinate with a local temporal average, substantially
reducing high-frequency noise while preserving the genuine low-frequency motion
of vehicles and pedestrians.

TWO IMPLEMENTATIONS
====================
MovingAverageSmoother (default)
  Simple O(1) online update using a fixed-length ring buffer (deque) per track.
  Window size 5–9 frames is sufficient for 25–30 fps video.
  Pro: zero parameters to tune, extremely fast.
  Con: introduces a half-window lag (2–4 frames), acceptable for safety analysis.

KalmanSmoother (alternative)
  1-D Kalman filter applied independently to cx and cy.
  Pro: adapts to measurement noise; smoother on long straight segments.
  Con: requires two noise parameters (process_noise, measurement_noise) to tune.
  Use when you need minimal lag and still want filtering.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Moving-average smoother (recommended default)
# ---------------------------------------------------------------------------

class MovingAverageSmoother:
    """
    Per-track simple moving-average (SMA) smoother.

    Maintains a separate position buffer for each track ID.
    Expired tracks (not seen for `max_age` frames) are automatically purged
    to prevent unbounded memory growth.

    Args:
        window:   Number of frames to average (odd values give symmetric lag).
        max_age:  Frames since last update before a track's buffer is dropped.
    """

    def __init__(self, window: int = 7, max_age: int = 60) -> None:
        self.window  = window
        self.max_age = max_age

        # {track_id: deque of (cx, cy)}
        self._buffers:       Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        # {track_id: frames_since_last_update}
        self._last_seen:     Dict[int, int]   = {}
        self._global_frame:  int = 0

    def update(
        self,
        track_id: int,
        cx:       float,
        cy:       float,
    ) -> Tuple[float, float]:
        """
        Push a new (cx, cy) observation and return the smoothed estimate.

        Args:
            track_id: Unique track identifier.
            cx, cy:   Raw centre coordinates from the tracker.

        Returns:
            Smoothed (cx, cy) — the mean over the buffer.
        """
        buf = self._buffers[track_id]
        buf.append((cx, cy))
        self._last_seen[track_id] = self._global_frame

        arr = np.array(buf)
        s_cx = float(np.mean(arr[:, 0]))
        s_cy = float(np.mean(arr[:, 1]))
        return s_cx, s_cy

    def tick(self) -> None:
        """
        Advance the internal frame counter and prune stale buffers.
        Call once per video frame (regardless of whether any tracks are active).
        """
        self._global_frame += 1
        stale = [
            tid for tid, last in self._last_seen.items()
            if (self._global_frame - last) > self.max_age
        ]
        for tid in stale:
            del self._buffers[tid]
            del self._last_seen[tid]
        if stale:
            logger.debug("Pruned %d stale smoother buffers.", len(stale))

    def reset(self) -> None:
        """Clear all state (e.g. between video clips)."""
        self._buffers.clear()
        self._last_seen.clear()
        self._global_frame = 0


# ---------------------------------------------------------------------------
# 1-D Kalman smoother (alternative)
# ---------------------------------------------------------------------------

class _Kalman1D:
    """Scalar constant-velocity Kalman filter for a single coordinate."""

    def __init__(self, process_noise: float, measurement_noise: float) -> None:
        # State [x, v], Process model: x' = x + v, v' = v
        self.F  = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H  = np.array([[1.0, 0.0]])
        self.Q  = np.diag([process_noise, process_noise * 0.1])
        self.R  = np.array([[measurement_noise]])

        self.x  = np.zeros((2, 1))          # state
        self.P  = np.eye(2) * 100.0         # covariance (high initial uncertainty)
        self._initialised = False

    def update(self, z: float) -> float:
        if not self._initialised:
            self.x[0, 0] = z
            self._initialised = True
            return z

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S[0, 0]
        self.x = self.x + K * (z - (self.H @ self.x)[0, 0])
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return float(self.x[0, 0])


class KalmanSmoother:
    """
    Per-track Kalman smoother applied independently to cx and cy.

    Args:
        process_noise:      Controls how quickly the filter adapts to genuine
                            motion changes. Higher → more responsive but less smooth.
        measurement_noise:  Expected pixel-level jitter in raw detections.
                            Higher → more smoothing but more lag.
        max_age:            Frames without update before purging track state.
    """

    def __init__(
        self,
        process_noise:    float = 0.5,
        measurement_noise: float = 5.0,
        max_age:          int   = 60,
    ) -> None:
        self.process_noise    = process_noise
        self.measurement_noise = measurement_noise
        self.max_age          = max_age

        self._filters_x:  Dict[int, _Kalman1D] = {}
        self._filters_y:  Dict[int, _Kalman1D] = {}
        self._last_seen:  Dict[int, int]        = {}
        self._global_frame = 0

    def _get_or_create(self, track_id: int) -> Tuple[_Kalman1D, _Kalman1D]:
        if track_id not in self._filters_x:
            self._filters_x[track_id] = _Kalman1D(self.process_noise, self.measurement_noise)
            self._filters_y[track_id] = _Kalman1D(self.process_noise, self.measurement_noise)
        return self._filters_x[track_id], self._filters_y[track_id]

    def update(
        self,
        track_id: int,
        cx:       float,
        cy:       float,
    ) -> Tuple[float, float]:
        """
        Feed a raw position observation and return the Kalman-smoothed estimate.
        """
        kx, ky = self._get_or_create(track_id)
        self._last_seen[track_id] = self._global_frame
        return kx.update(cx), ky.update(cy)

    def tick(self) -> None:
        """Advance frame counter and prune stale tracks."""
        self._global_frame += 1
        stale = [
            tid for tid, last in self._last_seen.items()
            if (self._global_frame - last) > self.max_age
        ]
        for tid in stale:
            del self._filters_x[tid]
            del self._filters_y[tid]
            del self._last_seen[tid]

    def reset(self) -> None:
        self._filters_x.clear()
        self._filters_y.clear()
        self._last_seen.clear()
        self._global_frame = 0
