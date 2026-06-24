from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

from .aggressor_classifier import ClassificationResult
from .conflict_detector import TrackDB, DELTA_FRAMES
from .event_detector import StopWaitEvent, SharpTurnEvent, OvertakeEvent
from .zone import INTERSECTION_ZONE_PX

# BGR
RED    = (0,   0,   255)
YELLOW = (0,   220, 255)
WHITE  = (255, 255, 255)
GREEN  = (0,   200, 0)
BLACK  = (0,   0,   0)
ORANGE = (0,   140, 255)
CYAN   = (255, 255, 0)    # BGR cyan
MAGENTA = (255, 0, 255)   # BGR magenta

PAIR_COLOURS = [
    (0,   0,   255),   # red
    (0,   165, 255),   # orange
    (255, 0,   255),   # magenta
    (255, 220, 0),     # cyan
    (0,   255, 128),   # spring green
    (128, 0,   255),   # violet
    (0,   200, 200),   # teal
    (200, 80,  0),     # dark blue
]

CONFLICT_DISPLAY_FRAMES: int = 75   # show each conflict for 3 seconds


def write_conflict_csv(results: List[ClassificationResult], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fields = [
        "frame", "v1_id", "v1_class", "v2_id", "v2_class",
        "conflict_world_x", "conflict_world_y",
        "conflict_px_x", "conflict_px_y",
        "aggressor_id", "passive_id",
        "angular_says", "ttc_says", "methods_agree",
        "ttc1_sec", "ttc2_sec",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            e = r.event
            writer.writerow({
                "frame": e.frame,
                "v1_id": e.v1_id,
                "v1_class": e.v1_class,
                "v2_id": e.v2_id,
                "v2_class": e.v2_class,
                "conflict_world_x": round(e.conflict_world[0], 4),
                "conflict_world_y": round(e.conflict_world[1], 4),
                "conflict_px_x": round(e.conflict_px[0], 1),
                "conflict_px_y": round(e.conflict_px[1], 1),
                "aggressor_id": r.aggressor_id,
                "passive_id": r.passive_id,
                "angular_says": r.angular_says,
                "ttc_says": r.ttc_says,
                "methods_agree": r.methods_agree,
                "ttc1_sec": round(r.ttc1, 3) if r.ttc1 != float("inf") else "inf",
                "ttc2_sec": round(r.ttc2, 3) if r.ttc2 != float("inf") else "inf",
            })
    print(f"[reporter] Wrote {len(results)} conflict events -> {output_path}")


def _filter_one_per_vehicle(results: List[ClassificationResult]) -> List[ClassificationResult]:
    """
    Each physical vehicle should appear in at most ONE conflict in the video.
    Iterate conflicts in detection order; once a vehicle is assigned to a pair,
    skip any later conflict that re-uses it.
    """
    used: Set[int] = set()
    filtered = []
    for r in results:
        if r.aggressor_id in used or r.passive_id in used:
            continue
        used.add(r.aggressor_id)
        used.add(r.passive_id)
        filtered.append(r)
    return filtered


def _draw_zone(img: np.ndarray) -> None:
    pts = np.array(INTERSECTION_ZONE_PX, dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], GREEN)
    cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)
    cv2.polylines(img, [pts], True, GREEN, 2)


def _centre(rec: dict) -> Tuple[int, int]:
    return int(rec["center_x"]), int(rec["center_y"])


def _draw_vehicle(img: np.ndarray, rec: dict, box_colour: Tuple, label: str) -> None:
    x1, y1, x2, y2 = int(rec["x1"]), int(rec["y1"]), int(rec["x2"]), int(rec["y2"])
    cv2.rectangle(img, (x1, y1), (x2, y2), box_colour, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = max(y1 - 6, 16)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), box_colour, -1)
    cv2.putText(img, label, (x1 + 3, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2, cv2.LINE_AA)


def _draw_stop_wait_events(
    img: np.ndarray,
    sw_events: List[StopWaitEvent],
    tracks: TrackDB,
    frame_idx: int,
    display_frames: int,
) -> None:
    for ev in sw_events:
        if not (ev.frame <= frame_idx < ev.frame + display_frames):
            continue
        # Draw stopped vehicle in cyan
        srec = tracks.get(ev.stopped_id, {}).get(frame_idx)
        crec = tracks.get(ev.cause_id,   {}).get(frame_idx)
        if srec:
            x1, y1, x2, y2 = int(srec["x1"]), int(srec["y1"]), int(srec["x2"]), int(srec["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 3)
            label = f"SW-STOP #{ev.stopped_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            ty = max(y1 - 6, 16)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), CYAN, -1)
            cv2.putText(img, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)
        if crec:
            x1, y1, x2, y2 = int(crec["x1"]), int(crec["y1"]), int(crec["x2"]), int(crec["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), CYAN, 2)
            label = f"SW-CAUSE #{ev.cause_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(y1 - 6, 16)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), CYAN, -1)
            cv2.putText(img, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)
        # Connecting arrow
        if srec and crec:
            sp = (int(srec["center_x"]), int(srec["center_y"]))
            cp = (int(crec["center_x"]), int(crec["center_y"]))
            cv2.arrowedLine(img, cp, sp, CYAN, 2, tipLength=0.12)


def _draw_sharp_turn_events(
    img: np.ndarray,
    st_events: List[SharpTurnEvent],
    tracks: TrackDB,
    frame_idx: int,
    display_frames: int,
) -> None:
    for ev in st_events:
        if not (ev.frame <= frame_idx < ev.frame + display_frames):
            continue
        rec = tracks.get(ev.vehicle_id, {}).get(frame_idx)
        if rec is None:
            continue
        x1, y1, x2, y2 = int(rec["x1"]), int(rec["y1"]), int(rec["x2"]), int(rec["y2"])
        cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE, 3)
        label = f"TURN {ev.curvature_deg:.0f}d/s #{ev.vehicle_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ty = max(y1 - 6, 16)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), ORANGE, -1)
        cv2.putText(img, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)


def _draw_overtake_events(
    img: np.ndarray,
    ot_events: List[OvertakeEvent],
    tracks: TrackDB,
    frame_idx: int,
    display_frames: int,
) -> None:
    for ev in ot_events:
        if not (ev.frame <= frame_idx < ev.frame + display_frames):
            continue
        r_er = tracks.get(ev.overtaker_id, {}).get(frame_idx)
        r_ed = tracks.get(ev.overtaken_id, {}).get(frame_idx)
        for rec, role in [(r_er, "OT-ER"), (r_ed, "OT-ED")]:
            if rec is None:
                continue
            x1, y1, x2, y2 = int(rec["x1"]), int(rec["y1"]), int(rec["x2"]), int(rec["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), MAGENTA, 3)
            tid = ev.overtaker_id if role == "OT-ER" else ev.overtaken_id
            label = f"{role} #{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            ty = max(y1 - 6, 16)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), MAGENTA, -1)
            cv2.putText(img, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2, cv2.LINE_AA)
        if r_er and r_ed:
            p1 = (int(r_er["center_x"]), int(r_er["center_y"]))
            p2 = (int(r_ed["center_x"]), int(r_ed["center_y"]))
            cv2.line(img, p1, p2, MAGENTA, 2, cv2.LINE_AA)


def _load_overtaking_csv(csv_path: str) -> List[Dict[str, Any]]:
    violations: List[Dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "track_id" not in row or "start_frame" not in row:
                continue
            violations.append({
                "track_id": int(row["track_id"]),
                "overtaken_vehicle_id": int(row.get("overtaken_vehicle_id", -1)),
                "class_name": row.get("class_name", ""),
                "start_frame": int(row["start_frame"]),
                "location": row.get("location", ""),
                "violation_type": row.get("violation_type", "Unsafe Overtaking"),
                "reason": row.get("reason", ""),
            })
    return violations


def _draw_unsafe_overtaking_violations(
    img: np.ndarray,
    violations: List[Dict[str, Any]],
    tracks: TrackDB,
    frame_idx: int,
) -> None:
    for violation in violations:
        if frame_idx < violation["start_frame"]:
            continue
        track_id = violation["track_id"]
        rec = tracks.get(track_id, {}).get(frame_idx)
        if rec is None:
            continue
        x1, y1, x2, y2 = int(rec["x1"]), int(rec["y1"]), int(rec["x2"]), int(rec["y2"])
        label = f"OVT#{track_id}"
        cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 6, 16)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), ORANGE, -1)
        cv2.putText(img, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2, cv2.LINE_AA)


def annotate_overtaking_video(
    csv_path: str,
    tracks: TrackDB,
    input_video_path: str,
    output_video_path: str,
) -> None:
    print(f"[reporter] Loading unsafe overtaking violations from: {csv_path}")
    violations = _load_overtaking_csv(csv_path)
    if not violations:
        print("[reporter] No unsafe overtaking violations to annotate.")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        _draw_unsafe_overtaking_violations(img, violations, tracks, frame_idx)
        cv2.putText(img, f"frame {frame_idx}", (width - 110, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        out.write(img)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[reporter] overtaking annotated {frame_idx}/{total} frames ...")

    cap.release()
    out.release()
    print(f"[reporter] Overtaking video -> {output_video_path}")


def annotate_video(
    results: List[ClassificationResult],
    tracks: TrackDB,
    input_video_path: str,
    output_video_path: str,
    display_frames: int = CONFLICT_DISPLAY_FRAMES,
    sw_events: List[StopWaitEvent] = None,
    st_events: List[SharpTurnEvent] = None,
    ot_events: List[OvertakeEvent] = None,
) -> None:
    sw_events = sw_events or []
    st_events = st_events or []
    ot_events = ot_events or []
    # Apply one-per-vehicle filter before rendering
    results = _filter_one_per_vehicle(results)
    print(f"[reporter] After one-per-vehicle filter: {len(results)} conflicts to display")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # frame → list of (pair_index, result) active in that frame
    active_map: Dict[int, List[Tuple[int, ClassificationResult]]] = defaultdict(list)
    for pidx, r in enumerate(results):
        for f in range(r.event.frame, r.event.frame + display_frames):
            active_map[f].append((pidx, r))

    # tid → (pair_index, role) — first assignment wins
    tid_role: Dict[int, Tuple[int, str]] = {}
    for pidx, r in enumerate(results):
        if r.aggressor_id not in tid_role:
            tid_role[r.aggressor_id] = (pidx, "AGG")
        if r.passive_id not in tid_role:
            tid_role[r.passive_id] = (pidx, "PAS")

    frame_idx = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        _draw_zone(img)

        active = active_map.get(frame_idx, [])

        # Draw conflict vehicles (their full track duration)
        centres: Dict[int, Tuple[int, int]] = {}
        for tid, (pidx, role) in tid_role.items():
            rec = tracks.get(tid, {}).get(frame_idx)
            if rec is None:
                continue
            box_colour = RED if role == "AGG" else YELLOW
            pair_colour = PAIR_COLOURS[pidx % len(PAIR_COLOURS)]
            label = f"C{pidx+1}-{role} #{tid}"
            _draw_vehicle(img, rec, box_colour, label)
            x1, y1, x2, y2 = int(rec["x1"]), int(rec["y1"]), int(rec["x2"]), int(rec["y2"])
            cv2.rectangle(img, (x1-3, y1-3), (x2+3, y2+3), pair_colour, 1)
            centres[tid] = _centre(rec)

        # Draw connecting line + conflict point for ALL frames in display window
        drawn_pairs: Set[int] = set()
        for pidx, r in active:
            if pidx in drawn_pairs:
                continue
            drawn_pairs.add(pidx)

            e = r.event
            ca = centres.get(r.aggressor_id)
            cp = centres.get(r.passive_id)
            pair_colour = PAIR_COLOURS[pidx % len(PAIR_COLOURS)]

            # Connecting line between the two vehicles
            if ca and cp:
                cv2.line(img, ca, cp, pair_colour, 2, cv2.LINE_AA)

            # Conflict point — visible throughout the whole display window
            cx, cy = int(e.conflict_px[0]), int(e.conflict_px[1])
            cv2.circle(img, (cx, cy), 12, WHITE, -1)
            cv2.circle(img, (cx, cy), 14, pair_colour, 3)
            cv2.putText(img, f"C{pidx+1}", (cx + 16, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

            # Path arrows only on detection frame
            if frame_idx == e.frame:
                v1_rec = tracks.get(e.v1_id, {}).get(e.frame)
                v2_rec = tracks.get(e.v2_id, {}).get(e.frame)
                v1_f   = tracks.get(e.v1_id, {}).get(e.frame + DELTA_FRAMES)
                v2_f   = tracks.get(e.v2_id, {}).get(e.frame + DELTA_FRAMES)
                agg_c  = RED    if e.v1_id == r.aggressor_id else YELLOW
                pas_c  = YELLOW if e.v1_id == r.aggressor_id else RED
                if v1_rec and v1_f:
                    cv2.arrowedLine(img, _centre(v1_rec), _centre(v1_f), agg_c, 2, tipLength=0.15)
                if v2_rec and v2_f:
                    cv2.arrowedLine(img, _centre(v2_rec), _centre(v2_f), pas_c, 2, tipLength=0.15)

        # Compact info panel — only show currently active conflicts, max 4 lines
        if active:
            panel_x = width - 270
            lines = [f"C{pidx+1}: AGG#{r.aggressor_id} vs PAS#{r.passive_id}"
                     for pidx, r in active[:4]]
            ph = 22 + len(lines) * 22
            cv2.rectangle(img, (panel_x - 6, 8), (width - 8, ph), BLACK, -1)
            cv2.rectangle(img, (panel_x - 6, 8), (width - 8, ph), WHITE, 1)
            cv2.putText(img, "CONFLICT", (panel_x, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
            for row, (line, (pidx, _)) in enumerate(zip(lines, active[:4])):
                colour = PAIR_COLOURS[pidx % len(PAIR_COLOURS)]
                cv2.putText(img, line, (panel_x, 26 + (row+1)*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, colour, 1, cv2.LINE_AA)

        # Bottom legend
        cv2.rectangle(img, (8, height - 58), (240, height - 8), BLACK, -1)
        cv2.rectangle(img, (8, height - 58), (240, height - 8), WHITE, 1)
        cv2.putText(img, "RED    = Aggressor (arrives 1st)",
                    (14, height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED, 1, cv2.LINE_AA)
        cv2.putText(img, "YELLOW = Passive  (would be hit)",
                    (14, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1, cv2.LINE_AA)

        cv2.putText(img, f"frame {frame_idx}", (width - 110, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        _draw_stop_wait_events(img, sw_events, tracks, frame_idx, display_frames)
        _draw_sharp_turn_events(img, st_events, tracks, frame_idx, display_frames)
        _draw_overtake_events(img, ot_events, tracks, frame_idx, display_frames)

        out.write(img)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[reporter] annotated {frame_idx}/{total} frames ...")

    cap.release()
    out.release()
    print(f"[reporter] Conflict video -> {output_video_path}")
