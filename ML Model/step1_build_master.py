import os
import random
from pathlib import Path
import pandas as pd
import numpy as np

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def main():
    # Setup Paths
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    
    # Locate files
    annotations_path = data_dir / "annotations.csv"
    full_tracks_path = data_dir / "long1_tracks_narain_cleaned_edited.csv"
    tailgating_path = data_dir / "tailgating_violations.csv"
    overtaking_path = data_dir / "unsafe_overtaking_violations.csv"
    wrong_way_path = data_dir / "wrong_way.csv"
    
    # Check dependencies
    for p in [annotations_path, full_tracks_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
            
    print("Loading datasets...")
    df_ann = pd.read_csv(annotations_path)
    df_full = pd.read_csv(full_tracks_path)
    
    # Index tracker tracks by frame for fast coordinate queries
    print("Indexing tracks by frame for fast coordinate lookups...")
    tracks_by_frame = {}
    for _, row in df_full.iterrows():
        f = int(row["frame"])
        tid = int(row["track_id"])
        if f not in tracks_by_frame:
            tracks_by_frame[f] = {}
        tracks_by_frame[f][tid] = {
            "class_name": row["class_name"],
            "xtl": row["x1"],
            "ytl": row["y1"],
            "xbr": row["x2"],
            "ybr": row["y2"],
            "world_x": row["world_x"],
            "world_y": row["world_y"]
        }
        
    # -------------------------------------------------------------------------
    # ID Mismatch resolution mappings
    # -------------------------------------------------------------------------
    AFFECTED_TO_EGO = {
        10: 0, 11: 0,
        12: 2, 13: 2,
        14: 3, 15: 3, 16: 3,
        17: 4, 18: 4,
        19: 5, 20: 5,
        21: 6, 22: 6,
        23: 7, 24: 7,
        25: 9, 26: 9, 27: 9, 28: 9, 29: 9, 30: 9, 31: 9
    }
    
    MANUAL_TO_TRACKER = {
        0: 71, 1: 395, 2: 782, 3: 870, 4: 1101, 5: 1144, 6: 1209, 7: 1445, 8: 1627, 9: 1787,
        10: 35, 11: 6, 12: 668, 13: 732, 14: 811, 15: 924, 16: 924, 17: 1125, 20: 1130,
        21: 1023, 22: 1293, 23: 1209, 24: 1320, 25: 1711, 26: 1759, 27: 1759, 28: 1741,
        29: 1765, 30: 1783, 31: 1783
    }
    
    # Gather dangerous tracker track IDs from manual annotations
    dangerous_tracker_ids = set()
    for mid, tid in MANUAL_TO_TRACKER.items():
        dangerous_tracker_ids.add(tid)
        
    # Exclude manual track IDs 0-31 as safety backup
    all_excluded_tracker_ids = set(dangerous_tracker_ids)
    all_excluded_tracker_ids.update(range(32)) 
    
    raw_pairs_dangerous = []
    danger_pairs_set = set() # To store (frame, min_id, max_id) to avoid duplicates
    
    # Helper function to sort IDs for undirected pair key
    def get_pair_key(f, t1, t2):
        return (f, min(t1, t2), max(t1, t2))

    # -------------------------------------------------------------------------
    # 1. Process Manual Annotations (Tier 1)
    #    Ego gets class 1, weight 10.0; Affected class 0, weight 3.0
    # -------------------------------------------------------------------------
    print("\nProcessing manual annotations (Tier 1)...")
    mapped_count = 0
    unmapped_set = set()
    
    df_ann_affected = df_ann[df_ann["track_id"] >= 10].copy()
    for _, row in df_ann_affected.iterrows():
        f = int(row["frame"])
        tB = int(row["track_id"])
        tA = AFFECTED_TO_EGO.get(tB, -1)
        
        tA_tracker = MANUAL_TO_TRACKER.get(tA, -1)
        tB_tracker = MANUAL_TO_TRACKER.get(tB, -1)
        
        if tA_tracker == -1 or tB_tracker == -1:
            unmapped_set.add(tB)
            continue
            
        if f in tracks_by_frame and tA_tracker in tracks_by_frame[f] and tB_tracker in tracks_by_frame[f]:
            pair_key = get_pair_key(f, tA_tracker, tB_tracker)
            if pair_key not in danger_pairs_set:
                danger_pairs_set.add(pair_key)
                
                info_a = tracks_by_frame[f][tA_tracker]
                info_b = tracks_by_frame[f][tB_tracker]
                
                rec_a = {"frame": f, "track_id": tA_tracker, "class_name": info_a["class_name"],
                         "xtl": info_a["xtl"], "ytl": info_a["ytl"], "xbr": info_a["xbr"], "ybr": info_a["ybr"],
                         "is_dangerous": 1, "sample_weight": 10.0}
                         
                rec_b = {"frame": f, "track_id": tB_tracker, "class_name": info_b["class_name"],
                         "xtl": info_b["xtl"], "ytl": info_b["ytl"], "xbr": info_b["xbr"], "ybr": info_b["ybr"],
                         "is_dangerous": 0, "sample_weight": 3.0}
                         
                raw_pairs_dangerous.extend([rec_a, rec_b])
                mapped_count += 1
                
    if len(unmapped_set) > 0:
        print(f"  Warning: The following manual affected IDs could not be mapped: {sorted(list(unmapped_set))}")
    print(f"  Manual annotations mapped: {mapped_count} interaction frames.")

    # -------------------------------------------------------------------------
    # 2. Process Rule Violations (Tier 2 - class 1, weight 3.0)
    # -------------------------------------------------------------------------
    print("\nProcessing rule violations (Tier 2)...")
    rule_mapped_count = 0
    
    def add_rule_pair(f, tA_tracker, tB_tracker):
        nonlocal rule_mapped_count
        if f in tracks_by_frame and tA_tracker in tracks_by_frame[f] and tB_tracker in tracks_by_frame[f]:
            # Skip if pair involves manual ego vehicles (avoid overriding Tier 1)
            if tA_tracker in dangerous_tracker_ids or tB_tracker in dangerous_tracker_ids:
                return
                
            pair_key = get_pair_key(f, tA_tracker, tB_tracker)
            if pair_key not in danger_pairs_set:
                danger_pairs_set.add(pair_key)
                
                info_a = tracks_by_frame[f][tA_tracker]
                info_b = tracks_by_frame[f][tB_tracker]
                
                # Rule violator gets is_dangerous=1, weight=3.0
                rec_a = {"frame": f, "track_id": tA_tracker, "class_name": info_a["class_name"],
                         "xtl": info_a["xtl"], "ytl": info_a["ytl"], "xbr": info_a["xbr"], "ybr": info_a["ybr"],
                         "is_dangerous": 1, "sample_weight": 3.0}
                         
                # Affected vehicle gets is_dangerous=0, weight=1.0
                rec_b = {"frame": f, "track_id": tB_tracker, "class_name": info_b["class_name"],
                         "xtl": info_b["xtl"], "ytl": info_b["ytl"], "xbr": info_b["xbr"], "ybr": info_b["ybr"],
                         "is_dangerous": 0, "sample_weight": 1.0}
                         
                raw_pairs_dangerous.extend([rec_a, rec_b])
                rule_mapped_count += 1
                
                # Also exclude these from the background safe pool
                all_excluded_tracker_ids.add(tA_tracker)

    # 2a. Tailgating
    if tailgating_path.exists():
        df_tg = pd.read_csv(tailgating_path)
        # For each unique pair, find the frame with minimum distance
        tg_pairs = df_tg.loc[df_tg.groupby(['track_id', 'leader_track_id'])['d'].idxmin()]
        for _, row in tg_pairs.iterrows():
            add_rule_pair(int(row["frame"]), int(row["track_id"]), int(row["leader_track_id"]))
            
    # 2b. Overtaking
    if overtaking_path.exists():
        df_ot = pd.read_csv(overtaking_path)
        for _, row in df_ot.iterrows():
            add_rule_pair(int(row["start_frame"]), int(row["track_id"]), int(row["overtaken_vehicle_id"]))
            
    # 2c. Wrong-Way
    if wrong_way_path.exists():
        df_ww = pd.read_csv(wrong_way_path)
        for _, row in df_ww.iterrows():
            f = int(row["start_frame"])
            tA = int(row["track_id"])
            if f in tracks_by_frame and tA in tracks_by_frame[f]:
                info_a = tracks_by_frame[f][tA]
                xa, ya = info_a["world_x"], info_a["world_y"]
                
                # Find nearest vehicle
                nearest_dist = float('inf')
                nearest_id = -1
                for tB, info_b in tracks_by_frame[f].items():
                    if tB != tA:
                        xb, yb = info_b["world_x"], info_b["world_y"]
                        dist = np.sqrt((xb - xa)**2 + (yb - ya)**2)
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_id = tB
                            
                if nearest_id != -1 and nearest_dist < 20.0: # Only pair if reasonably close
                    add_rule_pair(f, tA, nearest_id)

    print(f"  Rule violations mapped: {rule_mapped_count} interaction frames.")

    # -------------------------------------------------------------------------
    # 3. Sample Safe Background Pairs (Tier 3 - both get class 0, weight 1.0)
    #    Controlled Nearest/Relevant Vehicle Selection using relevance weight:
    #      w_ab = exp(-d(a,b)^2 / (2*sigma^2)) * (1 + cos(phi_ab)) / 2
    #    where sigma = 4m, phi_ab = atan2(yb-ya, xb-xa) - theta_a
    #    and theta_a = heading of vehicle a from its previous-frame displacement.
    #    Proximity Constraint: distance must be between 2.0 and 12.0 meters.
    # -------------------------------------------------------------------------
    print("\nSampling clean background safe pairs (Tier 3)...")
    
    SIGMA = 4.0  # Gaussian distance decay parameter (meters)
    
    # Clean background dataframe
    df_clean_bg = df_full[~df_full["track_id"].isin(all_excluded_tracker_ids)].copy()
    
    # Index clean background by frame
    frame_to_tracks = df_clean_bg.groupby("frame")["track_id"].apply(list).to_dict()
    eligible_frames = [f for f, tids in frame_to_tracks.items() if len(tids) >= 2]
    
    raw_pairs_safe = []
    
    # Match the number of total dangerous interaction pairs (manual + rules)
    target_count = len(raw_pairs_dangerous) // 2
    
    # Shuffle eligible frames for diversity, then iterate deterministically
    shuffled_frames = list(eligible_frames)
    random.shuffle(shuffled_frames)
    
    for f in shuffled_frames:
        if (len(raw_pairs_safe) // 2) >= target_count:
            break
            
        tids_in_frame = frame_to_tracks[f]
        
        for tA in tids_in_frame:
            if (len(raw_pairs_safe) // 2) >= target_count:
                break
                
            info_a = tracks_by_frame[f][tA]
            xa, ya = info_a["world_x"], info_a["world_y"]
            
            # Compute heading theta_a from previous frame displacement
            prev_info = tracks_by_frame.get(f - 1, {}).get(tA)
            if prev_info is not None:
                dx_a = xa - prev_info["world_x"]
                dy_a = ya - prev_info["world_y"]
                theta_a = np.arctan2(dy_a, dx_a)
            else:
                theta_a = 0.0  # Default heading if no previous frame
            
            # Score all eligible candidates b in this frame
            best_w = -1.0
            best_tB = -1
            best_info_b = None
            
            for tB in tids_in_frame:
                if tB == tA:
                    continue
                    
                info_b = tracks_by_frame[f][tB]
                xb, yb = info_b["world_x"], info_b["world_y"]
                
                # Euclidean distance
                d_ab = np.sqrt((xb - xa)**2 + (yb - ya)**2)
                
                # Proximity constraint: only consider candidates within [2.0, 12.0] meters
                if d_ab < 2.0 or d_ab > 12.0:
                    continue
                
                # Angular offset: direction from a to b minus heading of a
                direction_ab = np.arctan2(yb - ya, xb - xa)
                phi_ab = direction_ab - theta_a
                # Normalize to [-pi, pi]
                phi_ab = (phi_ab + np.pi) % (2 * np.pi) - np.pi
                
                # Relevance weight: w_ab = exp(-d^2 / 2*sigma^2) * (1 + cos(phi_ab)) / 2
                w_ab = np.exp(-(d_ab**2) / (2 * SIGMA**2)) * (1 + np.cos(phi_ab)) / 2
                
                if w_ab > best_w:
                    best_w = w_ab
                    best_tB = tB
                    best_info_b = info_b
            
            # If a valid candidate was found, check for duplicate and add the pair
            if best_tB != -1:
                pair_key = get_pair_key(f, tA, best_tB)
                if pair_key not in danger_pairs_set:
                    danger_pairs_set.add(pair_key)
                    
                    rec_a = {"frame": f, "track_id": tA, "class_name": info_a["class_name"],
                             "xtl": info_a["xtl"], "ytl": info_a["ytl"], "xbr": info_a["xbr"], "ybr": info_a["ybr"],
                             "is_dangerous": 0, "sample_weight": 1.0}
                    rec_b = {"frame": f, "track_id": best_tB, "class_name": best_info_b["class_name"],
                             "xtl": best_info_b["xtl"], "ytl": best_info_b["ytl"], "xbr": best_info_b["xbr"], "ybr": best_info_b["ybr"],
                             "is_dangerous": 0, "sample_weight": 1.0}
                             
                    raw_pairs_safe.extend([rec_a, rec_b])
            
    print(f"  Safe pairs sampled: {len(raw_pairs_safe) // 2} pairs (target was {target_count}).")

    # -------------------------------------------------------------------------
    # 4. Unify and Export consolidated master dataset
    # -------------------------------------------------------------------------
    df_dangerous = pd.DataFrame(raw_pairs_dangerous)
    df_safe = pd.DataFrame(raw_pairs_safe)
    
    df_master = pd.concat([df_dangerous, df_safe], ignore_index=True)
    master_path = data_dir / "master_train_raw.csv"
    df_master.to_csv(master_path, index=False)
    
    print("\n--- Step 1 Export Verification ---")
    print(f"Saved Master Raw Dataset: {master_path.name} -> Shape: {df_master.shape}")
    print("Class Distribution (at vehicle record level):")
    print(df_master["is_dangerous"].value_counts())
    print("\nSample Weights Summary (at vehicle record level):")
    print(df_master.groupby(["is_dangerous", "sample_weight"]).size().reset_index(name='count'))

if __name__ == "__main__":
    main()
