from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .zone import SCALE_MPX, in_intersection_zone

DELTA_FRAMES: int = 125         # lookahead window (5 sec at 25 fps)
PROXIMITY_M: float = 20.0       # max world distance between pair (metres)
MIN_HEADING_DIFF_DEG: float = 50.0  # skip pairs going same direction
TRACKED_CLASSES: set = {
    # YOLOv8 names
    "car", "motorcycle",
    # Roboflow model names
    "sedan", "hatchback", "suv", "lcv", "hcv",
    "two wheeler", "three wheeler",
}
# Per-class minimum track length
MIN_TRACK_LENGTH_BY_CLASS: Dict[str, int] = {
    "car":          100,
    "motorcycle":    25,
    "sedan":        100,
    "hatchback":    100,
    "suv":          100,
    "lcv":           50,
    "hcv":           50,
    "two wheeler":   25,
    "three wheeler": 25,
}
MIN_TRACK_LENGTH: int = 25   # fallback for any class not in the dict above


@dataclass
class ConflictEvent:
    frame: int
    v1_id: int
    v1_class: str
    v2_id: int
    v2_class: str
    conflict_world: Tuple[float, float]   # (x, y) metres
    conflict_px: Tuple[float, float]      # (x, y) pixels
    v1_speed: float                        # m/s at detection frame
    v2_speed: float
    v1_world_pos: Tuple[float, float]
    v2_world_pos: Tuple[float, float]
    v1_future_world: Tuple[float, float]  # position at frame + DELTA
    v2_future_world: Tuple[float, float]


TrackDB = Dict[int, Dict[int, dict]]  # track_id -> frame -> row


def load_tracks(csv_path: str) -> TrackDB:
    tracks: TrackDB = defaultdict(dict)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            tid = int(row["track_id"])
            frame = int(row["frame"])
            tracks[tid][frame] = {
                "class_name": row["class_name"],
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "world_x": float(row["world_x"]),
                "world_y": float(row["world_y"]),
                "velocity_ms": float(row["velocity_ms"]),
            }
    return tracks


MAX_TTC_SEC: float = 7.0        # only flag conflicts where both vehicles arrive within 7 s
MAX_TTC_DIFF_SEC: float = 1.0   # both vehicles must arrive at P within 1 s of each other
MIN_DISPLACEMENT_M: float = 3.0 # vehicle must travel at least 3 m over the lookahead window


def _segment_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """
    Bounded segment intersection for P1P2 and P3P4.
    Returns (x, y) if the two bounded segments cross, else None.
    """
    d1x = p2[0] - p1[0]
    d1y = p2[1] - p1[1]
    d2x = p4[0] - p3[0]
    d2y = p4[1] - p3[1]

    cross = d1x * d2y - d1y * d2x
    if abs(cross) < 1e-10:
        return None

    t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / cross
    s = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / cross

    if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
        return p1[0] + t * d1x, p1[1] + t * d1y
    return None


def _ray_intersection(
    pos1: Tuple[float, float],
    dir1: Tuple[float, float],
    pos2: Tuple[float, float],
    dir2: Tuple[float, float],
    s1: float,
    s2: float,
    max_ttc: float = MAX_TTC_SEC,
    max_ttc_diff: float = MAX_TTC_DIFF_SEC,
) -> Optional[Tuple[float, float]]:
    """
    Find where the two heading RAYS meet.
    Returns conflict point P only if:
      - P is ahead of both vehicles
      - Both TTCs are < max_ttc (neither vehicle is too far away)
      - |TTC1 - TTC2| < max_ttc_diff (both arrive at roughly the same time —
        eliminates 'comfortable lead' cases where one is already past P)
    """
    d1x, d1y = dir1
    d2x, d2y = dir2

    cross = d1x * d2y - d1y * d2x
    if abs(cross) < 1e-10:
        return None

    t = ((pos2[0] - pos1[0]) * d2y - (pos2[1] - pos1[1]) * d2x) / cross
    u = ((pos2[0] - pos1[0]) * d1y - (pos2[1] - pos1[1]) * d1x) / cross

    if t < 0 or u < 0:
        return None

    px = pos1[0] + t * d1x
    py = pos1[1] + t * d1y

    d1 = math.hypot(px - pos1[0], py - pos1[1])
    d2 = math.hypot(px - pos2[0], py - pos2[1])

    ttc1 = d1 / s1 if s1 > STOPPED_SPEED_MS else float("inf")
    ttc2 = d2 / s2 if s2 > STOPPED_SPEED_MS else float("inf")

    if ttc1 > max_ttc or ttc2 > max_ttc:
        return None

    # Reject if one vehicle arrives much earlier — it will be long gone
    if abs(ttc1 - ttc2) > max_ttc_diff:
        return None

    return px, py


STOPPED_SPEED_MS: float = 0.5


MIN_LOOKAHEAD: int = 25  # need at least 1 sec of real future data for direction to be meaningful


def _heading_angle(pos: Tuple[float, float], future: Tuple[float, float]) -> float:
    return math.atan2(future[1] - pos[1], future[0] - pos[0])


def _angle_diff_deg(a1: float, a2: float) -> float:
    diff = abs(math.degrees(a1 - a2)) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _best_future_frame(
    track_frames: dict, t: int, delta: int, min_lookahead: int = MIN_LOOKAHEAD
) -> Optional[int]:
    """
    Return the furthest available frame in [t+min_lookahead, t+delta].
    Handles tracking gaps: uses the furthest real position we have rather
    than requiring data at exactly t+delta.
    """
    for future in range(t + delta, t + min_lookahead - 1, -1):
        if future in track_frames:
            return future
    return None


def detect_conflicts(
    tracks: TrackDB,
    delta: int = DELTA_FRAMES,
    proximity_m: float = PROXIMITY_M,
    min_angle_deg: float = MIN_HEADING_DIFF_DEG,
    min_track_length: int = MIN_TRACK_LENGTH,
    fps: float = 30.0,
) -> List[ConflictEvent]:
    # Drop short-lived tracks using per-class thresholds
    def _min_len(frames: dict) -> int:
        cls = next(iter(frames.values()))["class_name"]
        return MIN_TRACK_LENGTH_BY_CLASS.get(cls, min_track_length)

    long_tracks = {tid: frames for tid, frames in tracks.items()
                   if len(frames) >= _min_len(frames)}

    # Build frame -> [track_ids] index
    frame_to_ids: Dict[int, List[int]] = defaultdict(list)
    for tid, frames in long_tracks.items():
        for frame in frames:
            frame_to_ids[frame].append(tid)

    seen_pairs: set = set()
    conflicts: List[ConflictEvent] = []

    for frame in sorted(frame_to_ids.keys()):
        # Vehicles inside zone with at least MIN_LOOKAHEAD frames of future data
        in_zone: List[int] = []
        future_map: Dict[int, int] = {}   # tid -> best available future frame
        for tid in frame_to_ids[frame]:
            rec = long_tracks[tid][frame]
            if rec["class_name"] not in TRACKED_CLASSES:
                continue
            if not in_intersection_zone(rec["center_x"], rec["center_y"]):
                continue
            ff = _best_future_frame(long_tracks[tid], frame, delta)
            if ff is not None:
                in_zone.append(tid)
                future_map[tid] = ff

        for i in range(len(in_zone)):
            for j in range(i + 1, len(in_zone)):
                v1_id, v2_id = in_zone[i], in_zone[j]
                pair_key = (min(v1_id, v2_id), max(v1_id, v2_id))

                if pair_key in seen_pairs:
                    continue

                v1 = long_tracks[v1_id][frame]
                v2 = long_tracks[v2_id][frame]
                v1_f = long_tracks[v1_id][future_map[v1_id]]
                v2_f = long_tracks[v2_id][future_map[v2_id]]

                # --- Filter 0: both vehicles must be genuinely moving ---
                # Use actual displacement over the lookahead window, NOT instantaneous
                # velocity (which is noisy frame-to-frame). A parked car with jittery
                # detections can show high instantaneous speed but near-zero displacement.
                v1_pos = (v1["world_x"], v1["world_y"])
                v2_pos = (v2["world_x"], v2["world_y"])
                v1_fpos = (v1_f["world_x"], v1_f["world_y"])
                v2_fpos = (v2_f["world_x"], v2_f["world_y"])

                disp1 = math.hypot(v1_fpos[0] - v1_pos[0], v1_fpos[1] - v1_pos[1])
                disp2 = math.hypot(v2_fpos[0] - v2_pos[0], v2_fpos[1] - v2_pos[1])

                if disp1 < MIN_DISPLACEMENT_M or disp2 < MIN_DISPLACEMENT_M:
                    continue

                # Effective speed = displacement / actual elapsed time (stable, not noisy)
                elapsed1 = (future_map[v1_id] - frame) / fps
                elapsed2 = (future_map[v2_id] - frame) / fps
                eff_speed1 = disp1 / elapsed1 if elapsed1 > 0 else 0.0
                eff_speed2 = disp2 / elapsed2 if elapsed2 > 0 else 0.0

                # --- Filter 1: proximity ---
                dist = math.hypot(
                    v1["world_x"] - v2["world_x"],
                    v1["world_y"] - v2["world_y"],
                )
                if dist > proximity_m:
                    continue

                # --- Filter 2: heading angle (skip same-direction pairs) ---
                h1 = _heading_angle(
                    (v1["world_x"], v1["world_y"]),
                    (v1_f["world_x"], v1_f["world_y"]),
                )
                h2 = _heading_angle(
                    (v2["world_x"], v2["world_y"]),
                    (v2_f["world_x"], v2_f["world_y"]),
                )
                if _angle_diff_deg(h1, h2) < min_angle_deg:
                    continue

                # --- Filter 3: TTC-based conflict detection ---
                dir1 = (v1_fpos[0] - v1_pos[0], v1_fpos[1] - v1_pos[1])
                dir2 = (v2_fpos[0] - v2_pos[0], v2_fpos[1] - v2_pos[1])

                pt = _ray_intersection(
                    v1_pos, dir1, v2_pos, dir2,
                    eff_speed1, eff_speed2,
                )
                if pt is None:
                    continue

                seen_pairs.add(pair_key)
                conflicts.append(
                    ConflictEvent(
                        frame=frame,
                        v1_id=v1_id,
                        v1_class=v1["class_name"],
                        v2_id=v2_id,
                        v2_class=v2["class_name"],
                        conflict_world=pt,
                        conflict_px=(pt[0] / SCALE_MPX, pt[1] / SCALE_MPX),
                        v1_speed=eff_speed1,
                        v2_speed=eff_speed2,
                        v1_world_pos=v1_pos,
                        v2_world_pos=v2_pos,
                        v1_future_world=v1_fpos,
                        v2_future_world=v2_fpos,
                    )
                )

    return conflicts
