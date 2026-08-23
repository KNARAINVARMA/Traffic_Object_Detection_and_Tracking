import os
from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# Temporal window configuration
# -------------------------------------------------------------------------
TEMPORAL_HORIZON = 75        # Total frames of history (~2.5 seconds at 30 FPS)
SAMPLING_STRIDE = 6          # Sample every 6th frame within the window

# Sampled offsets within the window [0, 6, 12, ..., 72]
SAMPLED_OFFSETS = list(range(0, TEMPORAL_HORIZON, SAMPLING_STRIDE))
NUM_SAMPLED_POSITIONS = len(SAMPLED_OFFSETS)  # = 13
NUM_FEATURES = 2 * NUM_SAMPLED_POSITIONS + 1  # = 27 (13 distances + 13 angles + 1 v_rel)

# Minimum co-existence ratio: require at least 80% of sampled positions
MIN_COEXIST_RATIO = 0.80
MIN_COEXIST_POSITIONS = int(np.ceil(NUM_SAMPLED_POSITIONS * MIN_COEXIST_RATIO))

def main():
    # Setup Paths
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    
    master_raw_path = data_dir / "master_train_raw.csv"
    full_tracks_path = data_dir / "long1_tracks_narain_cleaned_edited.csv"
    
    if not master_raw_path.exists():
        raise FileNotFoundError(f"Missing master raw dataset: {master_raw_path}. Run step1_build_master.py first.")
    if not full_tracks_path.exists():
        raise FileNotFoundError(f"Missing tracker file: {full_tracks_path}")
        
    print("Loading datasets...")
    df_master_raw = pd.read_csv(master_raw_path)
    df_full = pd.read_csv(full_tracks_path)
    
    # -------------------------------------------------------------------------
    # Deduplicate Tracker Tracks by keeping the highest confidence detection
    # -------------------------------------------------------------------------
    print("Deduplicating tracker tracks by confidence...")
    initial_shape = df_full.shape
    df_full = df_full.sort_values("confidence", ascending=False)
    df_full = df_full.drop_duplicates(subset=["frame", "track_id"], keep="first")
    df_full = df_full.sort_values(["frame", "track_id"])
    print(f"  Tracker shape changed from {initial_shape} to {df_full.shape}")
    
    # -------------------------------------------------------------------------
    # Setup Lookup Dictionary for Fast Coordinates & Velocity Queries
    # -------------------------------------------------------------------------
    print("Building track details lookup dictionary using world coordinates...")
    lookup = df_full.set_index(["frame", "track_id"])[["world_x", "world_y", "velocity_ms"]].to_dict("index")
    
    def get_track_info(frame, track_id):
        key = (int(frame), int(track_id))
        if key in lookup:
            val = lookup[key]
            return val["world_x"], val["world_y"], val["velocity_ms"]
        return None
        
    # -------------------------------------------------------------------------
    # Temporal feature extraction with 75-frame horizon, stride 6
    #
    # Window convention:
    #   For a pair observed at frame f, the temporal window spans [f-74, f].
    #   Within this 75-frame window, we sample at offsets [0, 6, 12, ..., 72]
    #   from the window start, yielding frames:
    #     [f-74, f-68, f-62, ..., f-2]
    #   This produces N=13 sampled temporal positions.
    #
    # Co-existence relaxation:
    #   Instead of requiring both vehicles at ALL 13 positions (strict),
    #   we require at least MIN_COEXIST_POSITIONS (11) out of 13.
    #   Missing positions are imputed via nearest-neighbor interpolation
    #   from the closest available sampled frame.
    #
    # Features per sample:
    #   d_t1 ... d_t13       (Euclidean distance at each sampled position)
    #   theta_t1 ... theta_t13  (Relative angle at each sampled position)
    #   v_rel                (Relative velocity at current frame f)
    #   Total = 2*13 + 1 = 27 features
    # -------------------------------------------------------------------------
    def process_split(raw_df):
        print(f"Extracting sliding window kinematics features "
              f"({TEMPORAL_HORIZON}-frame horizon, stride {SAMPLING_STRIDE}, "
              f"{NUM_SAMPLED_POSITIONS} sampled positions, {NUM_FEATURES} features)...")
        print(f"  Minimum co-existence: {MIN_COEXIST_POSITIONS}/{NUM_SAMPLED_POSITIONS} "
              f"({MIN_COEXIST_RATIO*100:.0f}%)")
        
        windows = []
        skipped_coexist = 0
        skipped_velocity = 0
        imputed_count = 0
        
        # Pairs are in adjacent rows: Row 0/1 are Pair 1, Row 2/3 are Pair 2, etc.
        row1 = raw_df.iloc[0::2]
        row2 = raw_df.iloc[1::2]
        
        for i in range(min(len(row1), len(row2))):
            r1 = row1.iloc[i]
            r2 = row2.iloc[i]
            
            f = int(r1["frame"])
            tA = int(r1["track_id"])
            tB = int(r2["track_id"])
            is_dang = int(r1["is_dangerous"])
            
            # Carry forward sample weight (take max of pair)
            wA = float(r1.get("sample_weight", 1.0))
            wB = float(r2.get("sample_weight", 1.0))
            weight = max(wA, wB)
            
            # Compute the sampled frame indices within the 75-frame window
            window_start = f - (TEMPORAL_HORIZON - 1)  # = f - 74
            sampled_frames = [window_start + offset for offset in SAMPLED_OFFSETS]
            
            # Collect metrics for all available sampled positions
            present_metrics = {}
            for sf in sampled_frames:
                info_a = get_track_info(sf, tA)
                info_b = get_track_info(sf, tB)
                if info_a is not None and info_b is not None:
                    xa, ya, _ = info_a
                    xb, yb, _ = info_b
                    d = np.sqrt((xb - xa)**2 + (yb - ya)**2)
                    theta = np.arctan2(yb - ya, xb - xa)
                    present_metrics[sf] = (d, theta)
            
            # Check minimum co-existence threshold
            if len(present_metrics) < MIN_COEXIST_POSITIONS:
                skipped_coexist += 1
                continue
            
            # Also need current-frame velocity for v_rel
            info_a_current = get_track_info(f, tA)
            info_b_current = get_track_info(f, tB)
            if info_a_current is None or info_b_current is None:
                skipped_velocity += 1
                continue
                
            v_rel = info_a_current[2] - info_b_current[2]
            
            # Impute missing positions using nearest-neighbor interpolation
            present_sfs = sorted(present_metrics.keys())
            all_metrics = {}
            needs_imputation = False
            
            for sf in sampled_frames:
                if sf in present_metrics:
                    all_metrics[sf] = present_metrics[sf]
                else:
                    # Find nearest available sampled frame
                    nearest_sf = min(present_sfs, key=lambda x: abs(x - sf))
                    all_metrics[sf] = present_metrics[nearest_sf]
                    needs_imputation = True
            
            if needs_imputation:
                imputed_count += 1
            
            # Build feature dictionary
            cand_dict = {
                "is_dangerous": is_dang,
                "sample_weight": weight
            }
            for idx, sf in enumerate(sampled_frames):
                suffix = idx + 1
                d, theta = all_metrics[sf]
                cand_dict[f"d_t{suffix}"] = d
                cand_dict[f"theta_t{suffix}"] = theta
            
            cand_dict["v_rel"] = v_rel
            
            windows.append(cand_dict)
                
        df_out = pd.DataFrame(windows)
        
        print(f"\n  Samples retained: {len(windows)}")
        print(f"  Skipped (co-existence < {MIN_COEXIST_RATIO*100:.0f}%): {skipped_coexist}")
        print(f"  Skipped (no current velocity): {skipped_velocity}")
        print(f"  Samples with imputed positions: {imputed_count}")
        
        return df_out

    # Process dataset
    df_features = process_split(df_master_raw)
    
    # -------------------------------------------------------------------------
    # Export Final Production Feature Matrices
    # -------------------------------------------------------------------------
    feature_cols = []
    for i in range(1, NUM_SAMPLED_POSITIONS + 1):
        feature_cols.extend([f"d_t{i}", f"theta_t{i}"])
    feature_cols.append("v_rel")
    
    assert len(feature_cols) == NUM_FEATURES, \
        f"Expected {NUM_FEATURES} features, got {len(feature_cols)}"
    
    X_train_final = df_features[feature_cols]
    y_train_final = df_features[["is_dangerous"]]
    weights_train_final = df_features[["sample_weight"]]
    
    X_out_path = data_dir / "X_train_final.csv"
    y_out_path = data_dir / "y_train_final.csv"
    w_out_path = data_dir / "weights_train_final.csv"
    
    X_train_final.to_csv(X_out_path, index=False)
    y_train_final.to_csv(y_out_path, index=False)
    weights_train_final.to_csv(w_out_path, index=False)
    
    print("\n--- Step 2 Export Verification ---")
    print(f"Saved X_train_final: {X_out_path.name} -> Shape: {X_train_final.shape}")
    print(f"Saved y_train_final: {y_out_path.name} -> Shape: {y_train_final.shape}")
    print(f"Saved weights_train_final: {w_out_path.name} -> Shape: {weights_train_final.shape}")
    print(f"Feature count: {X_train_final.shape[1]} (expected {NUM_FEATURES})")
    
    print("\ny_train_final class distribution:")
    print(y_train_final["is_dangerous"].value_counts())
    print("\nweights_train_final summary:")
    print(weights_train_final["sample_weight"].value_counts())

if __name__ == "__main__":
    main()
