import argparse
import os
import pandas as pd
import numpy as np
import joblib

def assign_lane(r):
    if 6.0 <= r < 10.0:
        return 'Inner'
    elif 10.0 <= r <= 14.0:
        return 'Outer'
    else:
        return 'None'

def extract_features(df):
    # Sort dataset for sequential computations
    df = df.sort_values(by=['track_id', 'frame']).reset_index(drop=True)
    
    # Parameters
    X_c = 43.5
    Y_c = 28.5
    fps = 30
    
    # Coordinate Conversion & Lane Assignment
    df['r'] = np.sqrt((df['world_x'] - X_c)**2 + (df['world_y'] - Y_c)**2)
    df['theta'] = np.arctan2(df['world_y'] - Y_c, df['world_x'] - X_c)
    df['lane'] = df['r'].apply(assign_lane)
    df['is_in_ring'] = (df['lane'] != 'None')
    
    # Compute dynamics
    df['delta_theta'] = 0.0
    df['omega'] = 0.0
    df['accel'] = 0.0
    df['disp_30'] = 0.0
    df['disp_90'] = 0.0
    df['mean_vel_30'] = 0.0
    df['mean_vel_90'] = 0.0
    
    for track_id, group in df.groupby('track_id'):
        indices = group.index
        theta = group['theta'].values
        vel = group['velocity_ms'].values
        wx = group['world_x'].values
        wy = group['world_y'].values
        
        delta_theta = np.zeros(len(group))
        if len(group) > 1:
            delta_theta[1:] = np.arctan2(np.sin(theta[1:] - theta[:-1]), np.cos(theta[1:] - theta[:-1]))
        df.loc[indices, 'delta_theta'] = delta_theta
        df.loc[indices, 'omega'] = delta_theta * fps
        
        smooth_vel = pd.Series(vel).rolling(window=7, min_periods=1).mean()
        accel = smooth_vel.diff() * fps
        df.loc[indices, 'accel'] = accel.fillna(0.0).values
        
        disp_30 = np.zeros(len(group))
        mean_vel_30 = np.zeros(len(group))
        for k in range(len(group)):
            start_k = max(0, k - 30)
            disp_30[k] = np.hypot(wx[k] - wx[start_k], wy[k] - wy[start_k])
            mean_vel_30[k] = np.mean(vel[start_k:k+1])
        df.loc[indices, 'disp_30'] = disp_30
        df.loc[indices, 'mean_vel_30'] = mean_vel_30
        
        disp_90 = np.zeros(len(group))
        mean_vel_90 = np.zeros(len(group))
        for k in range(len(group)):
            start_k = max(0, k - 90)
            disp_90[k] = np.hypot(wx[k] - wx[start_k], wy[k] - wy[start_k])
            mean_vel_90[k] = np.mean(vel[start_k:k+1])
        df.loc[indices, 'disp_90'] = disp_90
        df.loc[indices, 'mean_vel_90'] = mean_vel_90

    # Compute interaction features
    df['dist_to_leader'] = 100.0
    df['leader_vel_diff'] = 0.0
    df['leader_omega_diff'] = 0.0
    df['leader_r_diff'] = 0.0
    df['leader_track_id'] = -1
    
    for (frame, lane), group in df[df['is_in_ring']].groupby(['frame', 'lane']):
        if len(group) < 2:
            continue
        
        sorted_group = group.sort_values('theta')
        n = len(sorted_group)
        records = sorted_group.to_dict('records')
        indices = sorted_group.index.tolist()
        
        for i in range(n):
            follower = records[i]
            leader = records[(i + 1) % n]
            
            delta_th = (leader['theta'] - follower['theta']) % (2 * np.pi)
            d = ((leader['r'] + follower['r']) / 2) * delta_th
            
            fol_idx = indices[i]
            df.at[fol_idx, 'dist_to_leader'] = float(d)
            df.at[fol_idx, 'leader_vel_diff'] = float(leader['velocity_ms'] - follower['velocity_ms'])
            df.at[fol_idx, 'leader_omega_diff'] = float(leader['omega'] - follower['omega'])
            df.at[fol_idx, 'leader_r_diff'] = float(leader['r'] - follower['r'])
            df.at[fol_idx, 'leader_track_id'] = int(leader['track_id'])
            
    return df

def run_inference(csv_path, model_path, output_path):
    # Load model
    print(f"Loading ML model from {model_path}...")
    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    label_cols = model_data['label_cols']
    
    # Load tracks
    print(f"Loading and processing tracks from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    df = extract_features(df_raw)
    
    df['is_in_ring'] = df['is_in_ring'].astype(int)
    
    # Handle one-hot encoding for classes and lanes
    df_encoded = pd.get_dummies(df, columns=['class_name', 'lane'], drop_first=False)
    
    # Add any missing feature columns from training features list
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0.0
            
    # Keep only the exact feature columns in correct order
    X = df_encoded[feature_cols].fillna(0.0)
    
    # Predict
    print("Running predictions with the ML model...")
    preds = model.predict(X)
    
    # Store predictions back in df
    for i, label_name in enumerate(label_cols):
        # Mapping label column name to short predictive name
        short_name = label_name.replace('label_', 'pred_')
        df[short_name] = preds[:, i]
        
    print("Post-processing predictions...")
    
    straddling_violations = []
    tailgating_records = []
    overtaking_records = []
    braking_records = []
    stoppage_records = []
    wrong_way_violations = {}
    
    # 1. Lane Straddling: track-level aggregation
    # Flag the track if it has >= 30 frames predicted as straddling
    for track_id, group in df.groupby('track_id'):
        straddle_frames = group['pred_straddling'].sum()
        if straddle_frames >= 30:
            class_name = group['class_name'].iloc[0]
            straddling_violations.append({
                'track_id': track_id,
                'class_name': class_name,
                'violation_type': 'Lane Straddling',
                'frame': np.nan,
                'leader_track_id': np.nan,
                'lane': np.nan,
                'd': np.nan
            })
            
    # 2. Tailgating: frame-level export
    tg_mask = (df['pred_tailgating'] == 1) & (df['is_in_ring'])
    for idx in df[tg_mask].index:
        row = df.loc[idx]
        tailgating_records.append({
            'violation_type': 'Tailgating',
            'frame': int(row['frame']),
            'track_id': int(row['track_id']),
            'leader_track_id': int(row['leader_track_id']) if row['leader_track_id'] != -1 else np.nan,
            'class_name': row['class_name'],
            'lane': row['lane'],
            'd': float(row['dist_to_leader'])
        })
        
    # 3. Unsafe Overtaking: group by track, identify blocks of predictions, choosing middle frame of block
    for track_id, group in df.groupby('track_id'):
        group = group.sort_values('frame')
        pred_ot = group['pred_overtaking'].values
        frames = group['frame'].values
        leader_ids = group['leader_track_id'].values
        lanes = group['lane'].values
        r_vals = group['r'].values
        dist_to_leaders = group['dist_to_leader'].values
        class_name = group['class_name'].iloc[0]
        
        # Find contiguous blocks of pred_ot == 1
        in_block = False
        block_start = -1
        for i, val in enumerate(pred_ot):
            if val == 1 and not in_block:
                in_block = True
                block_start = i
            elif val == 0 and in_block:
                in_block = False
                # End of block, take middle index
                mid_idx = (block_start + i - 1) // 2
                f_mid = frames[mid_idx]
                ld_mid = leader_ids[mid_idx]
                r_diff_mid = abs(r_vals[mid_idx] - 10.0) # radial difference representation
                
                overtaking_records.append({
                    'violation_type': 'Unsafe Overtaking',
                    'frame': int(f_mid),
                    'track_id': int(track_id),
                    'leader_track_id': int(ld_mid) if ld_mid != -1 else np.nan,
                    'class_name': class_name,
                    'lane': lanes[mid_idx],
                    'd': float(dist_to_leaders[mid_idx])
                })
        # If block goes until the end of the track
        if in_block:
            mid_idx = (block_start + len(pred_ot) - 1) // 2
            f_mid = frames[mid_idx]
            ld_mid = leader_ids[mid_idx]
            overtaking_records.append({
                'violation_type': 'Unsafe Overtaking',
                'frame': int(f_mid),
                'track_id': int(track_id),
                'leader_track_id': int(ld_mid) if ld_mid != -1 else np.nan,
                'class_name': class_name,
                'lane': lanes[mid_idx],
                'd': float(dist_to_leaders[mid_idx])
            })
            
    # 4. Sudden Braking: frame-level export
    brk_mask = (df['pred_braking'] == 1) & (df['is_in_ring'])
    for idx in df[brk_mask].index:
        row = df.loc[idx]
        braking_records.append({
            'violation_type': 'Sudden Braking',
            'frame': int(row['frame']),
            'track_id': int(row['track_id']),
            'leader_track_id': np.nan,
            'class_name': row['class_name'],
            'lane': row['lane'],
            'd': float(row['accel'])
        })
        
    # 5. Vehicle Stoppage: frame-level export
    stp_mask = (df['pred_stoppage'] == 1) & (df['is_in_ring'])
    for idx in df[stp_mask].index:
        row = df.loc[idx]
        stoppage_records.append({
            'violation_type': 'Vehicle Stoppage',
            'frame': int(row['frame']),
            'track_id': int(row['track_id']),
            'leader_track_id': np.nan,
            'class_name': row['class_name'],
            'lane': row['lane'],
            'd': float(row['disp_90'])
        })

    # 6. Wrong Way: print summary & export
    # Find start frames of wrong way violations for the console printout
    for track_id, group in df.groupby('track_id'):
        group = group.sort_values('frame')
        pred_ww = group['pred_wrong_way'].values
        frames = group['frame'].values
        class_name = group['class_name'].iloc[0]
        
        in_block = False
        for i, val in enumerate(pred_ww):
            if val == 1 and not in_block:
                in_block = True
                wrong_way_violations[track_id] = {
                    'class_name': class_name,
                    'start_frame': int(frames[i])
                }
            elif val == 0:
                in_block = False
                
    # Combine exports
    straddling_export = pd.DataFrame(straddling_violations)
    tailgating_export = pd.DataFrame(tailgating_records)
    overtaking_export = pd.DataFrame(overtaking_records)
    braking_export = pd.DataFrame(braking_records)
    stoppage_export = pd.DataFrame(stoppage_records)
    
    combined_df = pd.concat([
        straddling_export,
        tailgating_export,
        overtaking_export,
        braking_export,
        stoppage_export
    ], ignore_index=True)
    
    # Reorder and format columns
    columns_order = ['violation_type', 'frame', 'track_id', 'leader_track_id', 'class_name', 'lane', 'd']
    if not combined_df.empty:
        combined_df = combined_df[columns_order]
    else:
        combined_df = pd.DataFrame(columns=columns_order)
        
    combined_df.to_csv(output_path, index=False)
    print(f"\nAll violations successfully written to: {output_path}")
    
    # Print Summaries
    print("\n" + "="*50)
    print("ML PREDICTED SAFETY VIOLATIONS SUMMARY")
    print("="*50)
    
    print(f"Total unique track IDs violating Lane Straddling: {len(straddling_violations)}")
    print(f"Total Tailgating events: {len(tailgating_records)} (unique vehicles: {len(tailgating_export['track_id'].unique()) if not tailgating_export.empty else 0})")
    print(f"Total Unsafe Overtaking events: {len(overtaking_records)}")
    print(f"Total Sudden Braking events: {len(braking_records)}")
    print(f"Total Vehicle Stoppage events: {len(stoppage_records)}")
    
    if wrong_way_violations:
        print("\n--- Wrong-Way Driving Violations Detected (ML) ---")
        class_counts = {}
        for vid, info in wrong_way_violations.items():
            c_name = info['class_name']
            class_counts[c_name] = class_counts.get(c_name, 0) + 1
        for c_name, count in class_counts.items():
            print(f"- {c_name}: {count}")
        print("\nSpecific track IDs flagged:")
        for track_id, info in wrong_way_violations.items():
            print(f"Track ID {track_id} (Class: {info['class_name']}) - Violation started at frame: {info['start_frame']}")
    else:
        print("\nNo Wrong-Way Driving Violations detected by ML.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ML Safety Rules on Tracks CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to input tracks CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model joblib file")
    parser.add_argument("--output", type=str, default="rule_ml.csv", help="Path to save output violations CSV")
    args = parser.parse_args()
    
    run_inference(args.csv, args.model, args.output)
