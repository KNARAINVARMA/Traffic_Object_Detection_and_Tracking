from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .conflict_detector import (
    TrackDB, TRACKED_CLASSES, MIN_TRACK_LENGTH_BY_CLASS, MIN_TRACK_LENGTH,
    _best_future_frame, _heading_angle, _angle_diff_deg,
)

# ── Stop-and-wait constants ──────────────────────────────────────────────────
SW_STOP_SPEED_MS: float   = 1.0    # below this = stopped
SW_MOVING_SPEED_MS: float = 1.5    # must have been moving at this speed before
SW_WINDOW: int            = 50     # frames to look back for prior motion (~2 sec)
SW_CAUSE_PROX_M: float    = 25.0   # max distance to cause vehicle
SW_CAUSE_CONE_DEG: float  = 60.0   # max angle between heading and direction to cause
SW_COOLDOWN: int          = 75     # frames between re-flagging the same vehicle

# ── Sharp-turn constants (angular-velocity formula) ──────────────────────────
# Angular velocity = heading_change_deg / elapsed_sec
# Short window catches abrupt direction changes before they are diluted.
ST_WINDOW: int            = 10     # ~0.4 sec at 25 fps — small window
ST_MIN_ANG_VEL_DEG_S: float = 40.0 # must turn at ≥ 40 deg/sec
ST_MIN_SPEED_MS: float    = 1.0    # vehicle must be moving
ST_MIN_DISP_M: float      = 1.0    # must travel at least 1 m over window
ST_COOLDOWN: int          = 75     # frames between re-flagging the same vehicle


def _get_class(frames: dict) -> str:
    return next(iter(frames.values()))["class_name"]


def _disp_speed(frames: dict, t0: int, t1: int, fps: float) -> float:
    """Net displacement / elapsed time between two frames."""
    if t0 not in frames or t1 not in frames or t0 == t1:
        return 0.0
    dx = frames[t1]["world_x"] - frames[t0]["world_x"]
    dy = frames[t1]["world_y"] - frames[t0]["world_y"]
    return math.hypot(dx, dy) / ((t1 - t0) / fps)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class StopWaitEvent:
    frame: int
    stopped_id: int
    stopped_class: str
    cause_id: int
    cause_class: str
    stopped_px: Tuple[float, float]
    cause_px: Tuple[float, float]


@dataclass
class SharpTurnEvent:
    frame: int
    vehicle_id: int
    vehicle_class: str
    curvature_deg: float
    px_pos: Tuple[float, float]


@dataclass
class OvertakeEvent:
    frame: int
    overtaker_id: int
    overtaker_class: str
    overtaken_id: int
    overtaken_class: str
    overtaker_px: Tuple[float, float]
    overtaken_px: Tuple[float, float]


# ── Detector 1: Stop-and-Wait ────────────────────────────────────────────────

def detect_stop_wait(tracks: TrackDB, fps: float = 25.0) -> List[StopWaitEvent]:
    long_tracks = {
        tid: frames for tid, frames in tracks.items()
        if _get_class(frames) in TRACKED_CLASSES
        and len(frames) >= MIN_TRACK_LENGTH_BY_CLASS.get(_get_class(frames), MIN_TRACK_LENGTH)
    }

    # frame → list of track IDs present in that frame
    frame_to_ids: Dict[int, List[int]] = defaultdict(list)
    for tid, frames in long_tracks.items():
        for f in frames:
            frame_to_ids[f].append(tid)

    events: List[StopWaitEvent] = []
    last_flagged: Dict[int, int] = {}   # tid → last frame it was flagged

    for frame in sorted(frame_to_ids.keys()):
        for tid in frame_to_ids[frame]:
            # Cooldown
            if tid in last_flagged and frame - last_flagged[tid] < SW_COOLDOWN:
                continue

            tf = long_tracks[tid]
            sorted_f = sorted(tf.keys())

            # Current speed: displacement over last 10 frames
            recent = [f for f in sorted_f if frame - 10 <= f <= frame]
            if len(recent) < 2:
                continue
            cur_speed = _disp_speed(tf, recent[0], recent[-1], fps)
            if cur_speed > SW_STOP_SPEED_MS:
                continue  # not stopped

            # Prior speed: displacement SW_WINDOW frames ago
            t_prev_candidates = [f for f in sorted_f if f <= frame - SW_WINDOW]
            if not t_prev_candidates:
                continue
            t_prev = max(t_prev_candidates)
            if frame - t_prev < 25:
                continue
            t_mid = (t_prev + frame) // 2
            prior_w = [f for f in sorted_f if t_prev <= f <= t_mid]
            if len(prior_w) < 2:
                continue
            prior_speed = _disp_speed(tf, prior_w[0], prior_w[-1], fps)
            if prior_speed < SW_MOVING_SPEED_MS:
                continue  # wasn't moving before

            # Heading when moving (from t_prev sub-window)
            heading = math.atan2(
                tf[prior_w[-1]]["world_y"] - tf[prior_w[0]]["world_y"],
                tf[prior_w[-1]]["world_x"] - tf[prior_w[0]]["world_x"],
            )

            rec = tf[frame]
            pos = (rec["world_x"], rec["world_y"])

            # Find a cause vehicle: nearby, in the heading cone
            for cid in frame_to_ids[frame]:
                if cid == tid:
                    continue
                crec = long_tracks[cid][frame]
                cpos = (crec["world_x"], crec["world_y"])

                dist = math.hypot(cpos[0] - pos[0], cpos[1] - pos[1])
                if dist > SW_CAUSE_PROX_M:
                    continue

                bearing = math.atan2(cpos[1] - pos[1], cpos[0] - pos[0])
                diff = abs(math.degrees(heading - bearing)) % 360
                if diff > 180:
                    diff = 360 - diff
                if diff > SW_CAUSE_CONE_DEG:
                    continue

                last_flagged[tid] = frame
                events.append(StopWaitEvent(
                    frame=frame,
                    stopped_id=tid,
                    stopped_class=rec["class_name"],
                    cause_id=cid,
                    cause_class=crec["class_name"],
                    stopped_px=(rec["center_x"], rec["center_y"]),
                    cause_px=(crec["center_x"], crec["center_y"]),
                ))
                break  # one cause per stop event

    return events


# ── Detector 2: Sharp Turns ──────────────────────────────────────────────────

def detect_sharp_turns(tracks: TrackDB, fps: float = 25.0) -> List[SharpTurnEvent]:
    long_tracks = {
        tid: frames for tid, frames in tracks.items()
        if _get_class(frames) in TRACKED_CLASSES
        and len(frames) >= MIN_TRACK_LENGTH_BY_CLASS.get(_get_class(frames), MIN_TRACK_LENGTH)
    }

    events: List[SharpTurnEvent] = []
    last_flagged: Dict[int, int] = {}

    for tid, tf in long_tracks.items():
        sorted_f = sorted(tf.keys())

        for t in sorted_f:
            if tid in last_flagged and t - last_flagged[tid] < ST_COOLDOWN:
                continue

            # Need data ST_WINDOW frames back (short window)
            t0_candidates = [f for f in sorted_f if f <= t - ST_WINDOW]
            if not t0_candidates:
                continue
            t0 = max(t0_candidates)
            if t - t0 < 4:
                continue

            elapsed = (t - t0) / fps

            # Displacement + speed check
            dx = tf[t]["world_x"] - tf[t0]["world_x"]
            dy = tf[t]["world_y"] - tf[t0]["world_y"]
            disp = math.hypot(dx, dy)
            if disp < ST_MIN_DISP_M:
                continue
            if disp / elapsed < ST_MIN_SPEED_MS:
                continue

            # Heading at start and end of the short window
            h_start = math.atan2(
                tf[t0]["world_y"] - tf[sorted_f[max(0, sorted_f.index(t0) - 1)]]["world_y"],
                tf[t0]["world_x"] - tf[sorted_f[max(0, sorted_f.index(t0) - 1)]]["world_x"],
            ) if sorted_f.index(t0) > 0 else math.atan2(dy, dx)

            h_end = math.atan2(
                tf[t]["world_y"] - tf[t0]["world_y"],
                tf[t]["world_x"] - tf[t0]["world_x"],
            )

            # Angular velocity in deg/sec
            curvature = _angle_diff_deg(h_start, h_end)
            ang_vel = curvature / elapsed
            if ang_vel < ST_MIN_ANG_VEL_DEG_S:
                continue

            last_flagged[tid] = t
            rec = tf[t]
            events.append(SharpTurnEvent(
                frame=t,
                vehicle_id=tid,
                vehicle_class=rec["class_name"],
                curvature_deg=round(ang_vel, 1),   # store angular velocity (deg/s)
                px_pos=(rec["center_x"], rec["center_y"]),
            ))

    return events


# ── Detector 3: Overtaking (Q3 → Q4 transition in δ space) ──────────────────

def _delta_v2_to_v1(
    v2_pos: Tuple[float, float],
    v2_future: Tuple[float, float],
    v1_pos: Tuple[float, float],
) -> float:
    """
    δ = heading(V2) − bearing(V2 → V1).
    Q3 (cos<0, sin<0): V1 is behind-right of V2.
    Q4 (cos>0, sin<0): V1 is ahead-right of V2  ← overtake complete.
    """
    heading = math.atan2(v2_future[1] - v2_pos[1], v2_future[0] - v2_pos[0])
    bearing = math.atan2(v1_pos[1] - v2_pos[1], v1_pos[0] - v2_pos[0])
    return heading - bearing


def detect_overtakes(tracks: TrackDB, fps: float = 25.0) -> List[OvertakeEvent]:
    long_tracks = {
        tid: frames for tid, frames in tracks.items()
        if _get_class(frames) in TRACKED_CLASSES
        and len(frames) >= MIN_TRACK_LENGTH_BY_CLASS.get(_get_class(frames), MIN_TRACK_LENGTH)
    }

    frame_to_ids: Dict[int, List[int]] = defaultdict(list)
    for tid, frames in long_tracks.items():
        for f in frames:
            frame_to_ids[f].append(tid)

    # (v2_id, v1_id) → [(frame, cos_d, sin_d)]  — accumulate delta history
    delta_history: Dict[Tuple[int, int], List[Tuple[int, float, float]]] = defaultdict(list)
    seen_pairs: set = set()
    events: List[OvertakeEvent] = []

    for frame in sorted(frame_to_ids.keys()):
        tids = frame_to_ids[frame]

        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                v1id, v2id = tids[i], tids[j]
                pair = (min(v1id, v2id), max(v1id, v2id))
                if pair in seen_pairs:
                    continue

                v1f = long_tracks[v1id][frame]
                v2f = long_tracks[v2id][frame]

                # Proximity check
                dist = math.hypot(v1f["world_x"] - v2f["world_x"],
                                   v1f["world_y"] - v2f["world_y"])
                if dist > OT_PROX_M:
                    continue

                # Both must be moving
                v1_spd = v1f.get("velocity_ms", 0.0)
                v2_spd = v2f.get("velocity_ms", 0.0)
                if v1_spd < OT_MIN_SPEED_MS and v2_spd < OT_MIN_SPEED_MS:
                    continue

                # Get short-horizon future for heading
                ff1 = _best_future_frame(long_tracks[v1id], frame, 20, 8)
                ff2 = _best_future_frame(long_tracks[v2id], frame, 20, 8)
                if ff1 is None or ff2 is None:
                    continue

                v1fut = long_tracks[v1id][ff1]
                v2fut = long_tracks[v2id][ff2]

                h1 = _heading_angle((v1f["world_x"], v1f["world_y"]),
                                     (v1fut["world_x"], v1fut["world_y"]))
                h2 = _heading_angle((v2f["world_x"], v2f["world_y"]),
                                     (v2fut["world_x"], v2fut["world_y"]))

                if _angle_diff_deg(h1, h2) > OT_HEADING_DIFF_DEG:
                    continue  # not going same direction

                # Compute δ from V2's perspective toward V1 (V1 is potential overtaker)
                d = _delta_v2_to_v1(
                    (v2f["world_x"], v2f["world_y"]),
                    (v2fut["world_x"], v2fut["world_y"]),
                    (v1f["world_x"], v1f["world_y"]),
                )
                cos_d = math.cos(d)
                sin_d = math.sin(d)

                key = (v2id, v1id)
                history = delta_history[key]
                history.append((frame, cos_d, sin_d))
                # Trim to OT_WINDOW
                history = [(f, c, s) for f, c, s in history if frame - f <= OT_WINDOW]
                delta_history[key] = history

                if len(history) < 12:
                    continue

                recent = history[-6:]
                older  = history[:-6]
                if len(older) < 6:
                    continue

                # Q3 → Q4: cos goes negative → positive, sin stays negative
                older_q3  = sum(1 for _, c, s in older[-8:] if c < 0 and s < 0)
                recent_q4 = sum(1 for _, c, s in recent   if c > 0 and s < 0)

                if older_q3 >= 4 and recent_q4 >= 4:
                    seen_pairs.add(pair)
                    rec1 = long_tracks[v1id].get(frame, v1f)
                    rec2 = long_tracks[v2id].get(frame, v2f)
                    events.append(OvertakeEvent(
                        frame=frame,
                        overtaker_id=v1id,
                        overtaker_class=v1f["class_name"],
                        overtaken_id=v2id,
                        overtaken_class=v2f["class_name"],
                        overtaker_px=(rec1["center_x"], rec1["center_y"]),
                        overtaken_px=(rec2["center_x"], rec2["center_y"]),
                    ))

    return events
