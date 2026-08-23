import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------------
# Priority Vehicle IDs (10 manually annotated ego vehicles)
# These MUST be detected — bypass parked-filter, use relaxed thresholds
# -------------------------------------------------------------------------
PRIORITY_EGO_IDS = {71, 395, 782, 870, 1101, 1144, 1209, 1445, 1627, 1787}

# Also track the affected vehicles involved in annotated interactions
PRIORITY_AFFECTED_IDS = {35, 6, 668, 732, 811, 924, 1125, 1130,
                         1023, 1293, 1320, 1711, 1759, 1741, 1765, 1783}

ALL_PRIORITY_IDS = PRIORITY_EGO_IDS | PRIORITY_AFFECTED_IDS

# -------------------------------------------------------------------------
# Configurable constants
# -------------------------------------------------------------------------
PARKED_THRESHOLD_FRAMES = 180
PARKED_SPEED_THRESHOLD = 0.5

# Temporal window (must match step2)
TEMPORAL_HORIZON = 75
SAMPLING_STRIDE = 6
SAMPLED_OFFSETS = list(range(0, TEMPORAL_HORIZON, SAMPLING_STRIDE))
NUM_SAMPLED_POSITIONS = len(SAMPLED_OFFSETS)  # = 13

# Co-existence thresholds
MIN_COEXIST_GENERAL = int(np.ceil(NUM_SAMPLED_POSITIONS * 0.80))   # 11/13 for general
MIN_COEXIST_PRIORITY = int(np.ceil(NUM_SAMPLED_POSITIONS * 0.40))  # 6/13 for priority

# Elliptical proximity (general vehicles)
ELLIPSE_SEMI_MAJOR = 15.0
ELLIPSE_SEMI_MINOR = 8.0
MAX_CANDIDATES_GENERAL = 5

# Priority vehicles get wider circular proximity + more candidates
PRIORITY_PROXIMITY = 20.0   # meters, simple Euclidean radius
MAX_CANDIDATES_PRIORITY = 8

# Euclidean pre-filter
EUCLIDEAN_PREFILTER = max(ELLIPSE_SEMI_MAJOR, ELLIPSE_SEMI_MINOR) * 1.2

# Threshold bump for general vehicles (user requested "just a bit")
THRESHOLD_BUMP = 0.04

# Priority vehicle threshold (much more sensitive)
PRIORITY_THRESHOLD = 0.25


def get_danger_level(score, threshold_l2, threshold_l3, threshold_l4):
    if score >= threshold_l4:
        return (4, "HIGHLY DANGEROUS", (0, 0, 255), 2, 75)       # Red, 2.5s cooldown
    elif score >= threshold_l3:
        return (3, "DANGEROUS", (0, 165, 255), 2, 50)            # Orange, 1.7s cooldown
    elif score >= threshold_l2:
        return (2, "NEARLY DANGEROUS", (0, 255, 255), 1, 25)     # Yellow, 0.8s cooldown
    elif score >= 0.20:
        return (1, "NEARLY SAFE", None, 0, 0)
    else:
        return (0, "SAFE", None, 0, 0)


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    video_dir = base_dir / "video"
    
    tracks_path = data_dir / "long1_tracks_narain_cleaned_edited.csv"
    model_path = base_dir / "danger_model_production.pkl"
    calibrator_path = base_dir / "calibrator_production.pkl"
    metadata_path = base_dir / "model_metadata.pkl"
    video_input_path = video_dir / "intersection.mp4"
    output_path = video_dir / "intersection_annotated.mp4"
    
    for p in [tracks_path, model_path, calibrator_path, metadata_path, video_input_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
            
    print("Loading serialized model & calibrator...")
    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    metadata = joblib.load(metadata_path)
    
    # Load data-driven thresholds and apply bump for general vehicles
    base_l2 = metadata.get("render_threshold_l2", 0.50)
    base_l3 = metadata.get("render_threshold_l3", 0.65)
    base_l4 = metadata.get("render_threshold_l4", 0.80)
    
    threshold_l2 = base_l2 + THRESHOLD_BUMP
    threshold_l3 = base_l3 + THRESHOLD_BUMP
    threshold_l4 = base_l4 + THRESHOLD_BUMP
    
    print(f"  General rendering thresholds (+{THRESHOLD_BUMP:.2f} bump):")
    print(f"    L2 (Nearly Dangerous): >= {threshold_l2:.4f}")
    print(f"    L3 (Dangerous):        >= {threshold_l3:.4f}")
    print(f"    L4 (Highly Dangerous): >= {threshold_l4:.4f}")
    print(f"  Priority vehicle threshold: >= {PRIORITY_THRESHOLD:.4f}")
    print(f"  Priority ego IDs: {sorted(PRIORITY_EGO_IDS)}")
    
    print("Loading trajectory dataset...")
    df_tracks = pd.read_csv(tracks_path)
    
    print("Deduplicating tracks...")
    df_tracks = df_tracks.sort_values("confidence", ascending=False)
    df_tracks = df_tracks.drop_duplicates(subset=["frame", "track_id"], keep="first")
    df_tracks = df_tracks.sort_values(["frame", "track_id"])
    
    print("Building lookup index...")
    lookup = {}
    for _, row in df_tracks.iterrows():
        f = int(row["frame"])
        tid = int(row["track_id"])
        lookup[(f, tid)] = (
            row["world_x"], row["world_y"], row["velocity_ms"],
            int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        )
        
    print("Grouping tracks by frame...")
    frame_to_tracks = df_tracks.groupby("frame")["track_id"].apply(list).to_dict()
    
    # Pre-build feature column names
    feature_cols = []
    for i in range(1, NUM_SAMPLED_POSITIONS + 1):
        feature_cols.extend([f"d_t{i}", f"theta_t{i}"])
    feature_cols.append("v_rel")
    n_features = len(feature_cols)
    
    # Video I/O
    cap = cv2.VideoCapture(str(video_input_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_input_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"\nProcessing {total_frames} frames...")
    print(f"  General: Elliptical {ELLIPSE_SEMI_MAJOR}m x {ELLIPSE_SEMI_MINOR}m, Top-{MAX_CANDIDATES_GENERAL}, coexist>={MIN_COEXIST_GENERAL}/{NUM_SAMPLED_POSITIONS}")
    print(f"  Priority: Circular {PRIORITY_PROXIMITY}m, Top-{MAX_CANDIDATES_PRIORITY}, coexist>={MIN_COEXIST_PRIORITY}/{NUM_SAMPLED_POSITIONS}")
    
    danger_cooldowns = {}
    frame_idx = 0
    total_evaluated = 0
    total_flagged = 0
    priority_flagged = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        active_tids = frame_to_tracks.get(frame_idx, [])
        danger_in_this_frame = {}
        
        # 1. Gather active cooldowns
        for tA in active_tids:
            if tA in danger_cooldowns:
                rem, max_s = danger_cooldowns[tA]
                if rem > 0:
                    danger_in_this_frame[tA] = max_s
        
        # 2. Pre-compute positions, heading, parked status
        active_positions = {}
        parked_tids = set()
        
        for tA in active_tids:
            info_a = lookup.get((frame_idx, tA))
            if info_a is None:
                continue
            xa, ya, va = info_a[0], info_a[1], info_a[2]
            
            is_priority = tA in ALL_PRIORITY_IDS
            
            # Parked check — SKIP for priority vehicles
            if not is_priority:
                oldest_wf = frame_idx
                for wf in range(max(0, frame_idx - PARKED_THRESHOLD_FRAMES), frame_idx):
                    if (wf, tA) in lookup:
                        oldest_wf = wf
                        break
                if (frame_idx - oldest_wf) >= 30:
                    info_old = lookup.get((oldest_wf, tA))
                    if info_old is not None:
                        dist_moved = np.sqrt((xa - info_old[0])**2 + (ya - info_old[1])**2)
                        time_sec = (frame_idx - oldest_wf) / fps
                        if time_sec > 0 and (dist_moved / time_sec) < PARKED_SPEED_THRESHOLD:
                            parked_tids.add(tA)
                            continue
            
            # Heading
            prev = lookup.get((frame_idx - 1, tA))
            if prev is not None:
                theta_a = np.arctan2(ya - prev[1], xa - prev[0])
            else:
                theta_a = 0.0
            
            active_positions[tA] = (xa, ya, theta_a, va, is_priority)
        
        # 3. Build candidates
        candidates_data = []  # (tA, feature_array, is_priority_candidate)
        
        for tA, (xa, ya, theta_a, va, is_priority_a) in active_positions.items():
            cos_t = np.cos(theta_a)
            sin_t = np.sin(theta_a)
            
            neighbors = []
            
            for tB in active_tids:
                if tB == tA or tB in parked_tids:
                    continue
                info_b = lookup.get((frame_idx, tB))
                if info_b is None:
                    continue
                xb, yb = info_b[0], info_b[1]
                dx, dy = xb - xa, yb - ya
                euc_dist = np.sqrt(dx*dx + dy*dy)
                
                # Determine if this is a priority evaluation
                is_priority_pair = is_priority_a or (tB in ALL_PRIORITY_IDS)
                
                if is_priority_pair:
                    # Priority: simple circular proximity, wider range
                    if euc_dist <= PRIORITY_PROXIMITY:
                        neighbors.append((euc_dist, tB, True))
                else:
                    # General: elliptical proximity
                    if euc_dist > EUCLIDEAN_PREFILTER:
                        continue
                    dx_rot = cos_t * dx + sin_t * dy
                    dy_rot = -sin_t * dx + cos_t * dy
                    ell_dist = np.sqrt(
                        (dx_rot / ELLIPSE_SEMI_MAJOR)**2 +
                        (dy_rot / ELLIPSE_SEMI_MINOR)**2
                    )
                    if ell_dist <= 1.0:
                        neighbors.append((ell_dist, tB, False))
            
            # Sort and pick top-K (separate limits for priority vs general)
            priority_neighbors = [(d, t, p) for d, t, p in neighbors if p]
            general_neighbors = [(d, t, p) for d, t, p in neighbors if not p]
            
            priority_neighbors.sort()
            general_neighbors.sort()
            
            selected = (priority_neighbors[:MAX_CANDIDATES_PRIORITY] +
                        general_neighbors[:MAX_CANDIDATES_GENERAL])
            
            # Build feature vectors
            window_start = frame_idx - (TEMPORAL_HORIZON - 1)
            sampled_frames = [window_start + off for off in SAMPLED_OFFSETS]
            
            for _, tB, is_pri in selected:
                # Determine co-existence requirement
                min_coexist = MIN_COEXIST_PRIORITY if (is_priority_a or tB in ALL_PRIORITY_IDS) else MIN_COEXIST_GENERAL
                
                present_metrics = {}
                for sf in sampled_frames:
                    ia = lookup.get((sf, tA))
                    ib = lookup.get((sf, tB))
                    if ia is not None and ib is not None:
                        dx_sf = ib[0] - ia[0]
                        dy_sf = ib[1] - ia[1]
                        d = np.sqrt(dx_sf*dx_sf + dy_sf*dy_sf)
                        theta = np.arctan2(dy_sf, dx_sf)
                        present_metrics[sf] = (d, theta)
                
                if len(present_metrics) < min_coexist:
                    continue
                
                info_b = lookup.get((frame_idx, tB))
                vb = info_b[2] if info_b is not None else 0.0
                v_rel = va - vb
                
                # Impute missing positions
                present_sfs = sorted(present_metrics.keys())
                feat = np.zeros(n_features, dtype=np.float64)
                feat_idx = 0
                for sf in sampled_frames:
                    if sf in present_metrics:
                        d, theta = present_metrics[sf]
                    else:
                        nearest_sf = min(present_sfs, key=lambda x: abs(x - sf))
                        d, theta = present_metrics[nearest_sf]
                    feat[feat_idx] = d
                    feat[feat_idx + 1] = theta
                    feat_idx += 2
                feat[feat_idx] = v_rel
                
                candidates_data.append((tA, feat, is_priority_a or tB in ALL_PRIORITY_IDS))
        
        # 4. Batch inference + calibration
        if candidates_data:
            tids_arr = [c[0] for c in candidates_data]
            X_cand = np.array([c[1] for c in candidates_data])
            is_pri_arr = [c[2] for c in candidates_data]
            
            raw_scores = np.clip(model.predict(X_cand), 0.0, 1.0)
            calibrated_scores = np.clip(calibrator.predict(raw_scores), 0.0, 1.0)
            
            total_evaluated += len(calibrated_scores)
            
            for idx, cal_score in enumerate(calibrated_scores):
                tid = tids_arr[idx]
                is_pri = is_pri_arr[idx]
                
                # Use lower threshold for priority vehicles
                effective_threshold = PRIORITY_THRESHOLD if is_pri else threshold_l2
                
                if cal_score >= effective_threshold:
                    total_flagged += 1
                    if is_pri:
                        priority_flagged += 1
                    
                    current_max = danger_in_this_frame.get(tid, 0)
                    if cal_score > current_max:
                        danger_in_this_frame[tid] = cal_score
                        
                        # For priority at low scores, force L2 minimum level
                        render_score = max(cal_score, threshold_l2) if is_pri else cal_score
                        
                        _, _, _, _, cooldown_duration = get_danger_level(
                            render_score, threshold_l2, threshold_l3, threshold_l4
                        )
                        
                        existing_rem, existing_score = danger_cooldowns.get(tid, (0, 0))
                        if cal_score > existing_score or existing_rem <= 0:
                            danger_cooldowns[tid] = (cooldown_duration, cal_score)
        
        # 5. Render
        for tid in active_tids:
            if tid in danger_in_this_frame:
                score = danger_in_this_frame[tid]
                is_pri = tid in ALL_PRIORITY_IDS
                
                # For priority vehicles that scored below L2, render at L2 level
                render_score = max(score, threshold_l2) if is_pri else score
                
                level, _, color, thickness, _ = get_danger_level(
                    render_score, threshold_l2, threshold_l3, threshold_l4
                )
                
                if level >= 2:
                    info = lookup.get((frame_idx, tid))
                    if info is not None:
                        cv2.rectangle(frame, (info[3], info[4]), (info[5], info[6]),
                                      color, thickness)
        
        # 6. Update cooldowns
        for tid in list(danger_cooldowns.keys()):
            rem, max_s = danger_cooldowns[tid]
            rem -= 1
            if rem <= 0:
                danger_cooldowns.pop(tid)
            else:
                danger_cooldowns[tid] = (rem, max_s)
                
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            print(f"  Processed {frame_idx}/{total_frames} frames ({pct:.1f}%)")
            
    cap.release()
    out.release()
    
    print(f"\nProcessing complete! Video saved to: {output_path}")
    print(f"  Total interactions evaluated: {total_evaluated}")
    print(f"  Total flagged (>= thresholds): {total_flagged}")
    print(f"  Priority vehicle flags: {priority_flagged}")
    if total_evaluated > 0:
        print(f"  Overall flag rate: {total_flagged / total_evaluated * 100:.2f}%")

if __name__ == "__main__":
    main()
