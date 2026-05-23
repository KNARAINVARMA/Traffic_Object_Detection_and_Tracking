from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .conflict_detector import ConflictEvent, TrackDB, DELTA_FRAMES, _best_future_frame

STOPPED_SPEED_MS: float = 0.5   # vehicles slower than this cannot be aggressors
ANGULAR_WINDOW: int = 5         # frames either side of conflict frame for δ tracking


@dataclass
class ClassificationResult:
    event: ConflictEvent
    aggressor_id: int
    passive_id: int
    angular_says: str     # 'v1' | 'v2' | 'uncertain'
    ttc_says: str         # 'v1' | 'v2' | 'uncertain'
    methods_agree: bool
    ttc1: float           # seconds (inf if vehicle is stopped)
    ttc2: float


# ---------------------------------------------------------------------------
# Angular δ method  (primary)
# ---------------------------------------------------------------------------

def _compute_delta(
    v1_pos: Tuple[float, float],
    v1_future: Tuple[float, float],
    v2_pos: Tuple[float, float],
) -> float:
    """
    δ(t) = atan2(V1(t+Δ)_y − V1(t)_y,  V1(t+Δ)_x − V1(t)_x)
           − atan2(V2(t)_y  − V1(t)_y,  V2(t)_x  − V1(t)_x)

    heading of V1  minus  bearing from V1 to V2.
    """
    heading = math.atan2(v1_future[1] - v1_pos[1], v1_future[0] - v1_pos[0])
    bearing = math.atan2(v2_pos[1] - v1_pos[1], v2_pos[0] - v1_pos[0])
    return heading - bearing


def _classify_by_angular(
    event: ConflictEvent,
    tracks: TrackDB,
    delta: int = DELTA_FRAMES,
    window: int = ANGULAR_WINDOW,
) -> str:
    """
    Compute δ(t) over a window of frames around the conflict frame.
    cos δ: +ve → −ve  ⟹  V1 crosses first → 'v1'
    sin δ: any sign flip  ⟹  V2 crosses first → 'v2'
    Returns 'v1', 'v2', or 'uncertain'.
    """
    cos_vals: List[float] = []
    sin_vals: List[float] = []

    t_start = max(0, event.frame - window)
    t_end = event.frame + window

    for t in range(t_start, t_end + 1):
        future_t = t + delta
        v1_frames = tracks.get(event.v1_id, {})
        v2_frames = tracks.get(event.v2_id, {})
        if t not in v1_frames or t not in v2_frames:
            continue

        v1 = v1_frames[t]
        v2 = v2_frames[t]
        actual_future = _best_future_frame(v1_frames, t, delta)
        if actual_future is None:
            continue
        v1_f = v1_frames[actual_future]

        d = _compute_delta(
            (v1["world_x"], v1["world_y"]),
            (v1_f["world_x"], v1_f["world_y"]),
            (v2["world_x"], v2["world_y"]),
        )
        cos_vals.append(math.cos(d))
        sin_vals.append(math.sin(d))

    if len(cos_vals) < 2:
        return "uncertain"

    # cos δ: + → −  means V1 crosses first
    cos_pos_to_neg = any(
        cos_vals[k] > 0 and cos_vals[k + 1] < 0
        for k in range(len(cos_vals) - 1)
    )

    # sin δ: any sign flip means V2 crosses first
    sin_sign_flip = any(
        sin_vals[k] * sin_vals[k + 1] < 0
        for k in range(len(sin_vals) - 1)
    )

    if cos_pos_to_neg and not sin_sign_flip:
        return "v1"
    if sin_sign_flip and not cos_pos_to_neg:
        return "v2"
    if cos_pos_to_neg:       # both triggered — cos takes priority per notes ordering
        return "v1"
    if sin_sign_flip:
        return "v2"
    return "uncertain"


# ---------------------------------------------------------------------------
# TTC method  (secondary / confirmation)
# ---------------------------------------------------------------------------

def _classify_by_ttc(
    event: ConflictEvent,
) -> Tuple[str, float, float]:
    """
    TTC₁ = d₁ / s₁,  TTC₂ = d₂ / s₂
    Smaller TTC → aggressor.
    Returns (decision, ttc1, ttc2) where decision is 'v1', 'v2', or 'uncertain'.
    """
    d1 = math.hypot(
        event.v1_world_pos[0] - event.conflict_world[0],
        event.v1_world_pos[1] - event.conflict_world[1],
    )
    d2 = math.hypot(
        event.v2_world_pos[0] - event.conflict_world[0],
        event.v2_world_pos[1] - event.conflict_world[1],
    )

    ttc1 = d1 / event.v1_speed if event.v1_speed > STOPPED_SPEED_MS else float("inf")
    ttc2 = d2 / event.v2_speed if event.v2_speed > STOPPED_SPEED_MS else float("inf")

    if ttc1 == ttc2 == float("inf"):
        return "uncertain", ttc1, ttc2
    if ttc1 < ttc2:
        return "v1", ttc1, ttc2
    if ttc2 < ttc1:
        return "v2", ttc1, ttc2
    return "uncertain", ttc1, ttc2


# ---------------------------------------------------------------------------
# Combined classifier
# ---------------------------------------------------------------------------

def classify_aggressor(
    event: ConflictEvent,
    tracks: TrackDB,
    delta: int = DELTA_FRAMES,
) -> ClassificationResult:
    angular = _classify_by_angular(event, tracks, delta)
    ttc_decision, ttc1, ttc2 = _classify_by_ttc(event)

    # Angular is primary; fall back to TTC only when angular is uncertain
    if angular != "uncertain":
        final = angular
        agree = ttc_decision == "uncertain" or ttc_decision == angular
    else:
        final = ttc_decision
        agree = True  # only one source available

    if final == "v1":
        aggressor_id, passive_id = event.v1_id, event.v2_id
    elif final == "v2":
        aggressor_id, passive_id = event.v2_id, event.v1_id
    else:
        # Both methods uncertain — label lower-ID as aggressor as placeholder
        aggressor_id, passive_id = min(event.v1_id, event.v2_id), max(event.v1_id, event.v2_id)

    return ClassificationResult(
        event=event,
        aggressor_id=aggressor_id,
        passive_id=passive_id,
        angular_says=angular,
        ttc_says=ttc_decision,
        methods_agree=agree,
        ttc1=ttc1,
        ttc2=ttc2,
    )
