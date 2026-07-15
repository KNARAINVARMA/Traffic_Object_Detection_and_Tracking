import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict

def assign_lane(r):
    if 6.0 <= r < 10.0:
        return 'Inner'
    elif 10.0 <= r <= 14.0:
        return 'Outer'
    else:
        return 'None'

def prep_dataset(csv_path, output_path):
    print(f"Loading tracks from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Parameters
    X_c = 43.5
    Y_c = 28.5
    fps = 30
    dt = 1/30
    
    # Sort dataset for sequential computations
    df = df.sort_values(by=['track_id', 'frame']).reset_index(drop=True)
    
    # Step 1: Coordinate Conversion & Lane Assignment
    df['r'] = np.sqrt((df['world_x'] - X_c)**2 + (df['world_y'] - Y_c)**2)
    df['theta'] = np.arctan2(df['world_y'] - Y_c, df['world_x'] - X_c)
    df['lane'] = df['r'].apply(assign_lane)
    
    # Define Circulating Ring mask
    df['is_in_ring'] = (df['lane'] != 'None')
    
    # Compute single-vehicle motion dynamics (acceleration, angular velocity, displacement)
    print("Computing motion dynamics...")
    df['delta_theta'] = 0.0
    df['omega'] = 0.0
    df['accel'] = 0.0
    df['disp_30'] = 0.0
    df['disp_90'] = 0.0
    df['mean_vel_30'] = 0.0
    df['mean_vel_90'] = 0.0
    
    # Group by track_id to compute shifts and rollings
    for track_id, group in df.groupby('track_id'):
        indices = group.index
        theta = group['theta'].values
        vel = group['velocity_ms'].values
        wx = group['world_x'].values
        wy = group['world_y'].values
        
        # Shortest angular change
        delta_theta = np.zeros(len(group))
        if len(group) > 1:
            delta_theta[1:] = np.arctan2(np.sin(theta[1:] - theta[:-1]), np.cos(theta[1:] - theta[:-1]))
        df.loc[indices, 'delta_theta'] = delta_theta
        df.loc[indices, 'omega'] = delta_theta * fps
        
        # 7-frame rolling average velocity for acceleration
        smooth_vel = pd.Series(vel).rolling(window=7, min_periods=1).mean()
        accel = smooth_vel.diff() * fps
        df.loc[indices, 'accel'] = accel.fillna(0.0).values
        
        # Displacement and mean velocity windows
        # 30-frame window
        disp_30 = np.zeros(len(group))
        mean_vel_30 = np.zeros(len(group))
        for k in range(len(group)):
            start_k = max(0, k - 30)
            disp_30[k] = np.hypot(wx[k] - wx[start_k], wy[k] - wy[start_k])
            mean_vel_30[k] = np.mean(vel[start_k:k+1])
        df.loc[indices, 'disp_30'] = disp_30
        df.loc[indices, 'mean_vel_30'] = mean_vel_30
        
        # 90-frame window
        disp_90 = np.zeros(len(group))
        mean_vel_90 = np.zeros(len(group))
        for k in range(len(group)):
            start_k = max(0, k - 90)
            disp_90[k] = np.hypot(wx[k] - wx[start_k], wy[k] - wy[start_k])
            mean_vel_90[k] = np.mean(vel[start_k:k+1])
        df.loc[indices, 'disp_90'] = disp_90
        df.loc[indices, 'mean_vel_90'] = mean_vel_90

    # Step 2: Compute interaction features (leader-follower metrics)
    print("Computing interaction features...")
    df['dist_to_leader'] = 100.0
    df['leader_vel_diff'] = 0.0
    df['leader_omega_diff'] = 0.0
    df['leader_r_diff'] = 0.0
    df['leader_track_id'] = -1
    
    # Store track-frame index for fast lookup
    # Group frame-by-frame and lane-by-lane to compute relative leader/follower features
    for (frame, lane), group in df[df['is_in_ring']].groupby(['frame', 'lane']):
        if len(group) < 2:
            continue
        
        # Sort by theta
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

    # Step 3: Run mathematical rules to create ground-truth labels
    print("Generating labels based on mathematical safety rules...")
    
    # Initialize label columns
    df['label_straddling'] = 0
    df['label_tailgating'] = 0
    df['label_overtaking'] = 0
    df['label_braking'] = 0
    df['label_stoppage'] = 0
    df['label_wrong_way'] = 0
    
    # 3.1: Lane Straddling Violation Label
    df['is_straddling'] = (np.abs(df['r'] - 10.0) <= 0.5) & df['is_in_ring']
    for track_id, group in df.groupby('track_id'):
        is_strad = group['is_straddling'].values
        cumsum = np.cumsum(~is_strad)
        max_consecutive = 0
        if is_strad.any():
            counts = pd.Series(cumsum[is_strad]).value_counts()
            if not counts.empty:
                max_consecutive = counts.max()
        if max_consecutive >= 30:
            df.loc[group.index, 'label_straddling'] = np.where(group['is_in_ring'], 1, 0)
            
    # 3.2: Tailgating Violation Label
    # follower in same lane, d < 4.0m and follower speed > 1.0 m/s
    tailgating_mask = (df['is_in_ring']) & (df['dist_to_leader'] < 4.0) & (df['velocity_ms'] > 1.0)
    df.loc[tailgating_mask, 'label_tailgating'] = 1
    
    # 3.3: Unsafe Overtaking Violation Label
    # Crossover: sign change in angular difference between pair. Let's run the exact rule on the dataframe.
    pair_states = {}
    overtaking_keys = set() # set of (frame, track_id)
    
    for frame, group in df.groupby('frame'):
        for lane in ['Inner', 'Outer']:
            lane_vehicles = group[group['lane'] == lane].to_dict('records')
            n = len(lane_vehicles)
            if n < 2:
                continue
            
            for i in range(n):
                for j in range(i + 1, n):
                    v1 = lane_vehicles[i]
                    v2 = lane_vehicles[j]
                    
                    track_a = min(v1['track_id'], v2['track_id'])
                    track_b = max(v1['track_id'], v2['track_id'])
                    
                    rec_a = v1 if v1['track_id'] == track_a else v2
                    rec_b = v2 if v1['track_id'] == track_a else v1
                    
                    diff = np.arctan2(np.sin(rec_b['theta'] - rec_a['theta']), np.cos(rec_b['theta'] - rec_a['theta']))
                    pair_key = (track_a, track_b)
                    
                    if pair_key in pair_states:
                        prev_frame, prev_lane, prev_diff = pair_states[pair_key]
                        if prev_frame == frame - 1 and prev_lane == lane:
                            if prev_diff * diff < 0:
                                ang_change = np.abs(np.arctan2(np.sin(diff - prev_diff), np.cos(diff - prev_diff)))
                                dist = np.hypot(rec_a['world_x'] - rec_b['world_x'], rec_a['world_y'] - rec_b['world_y'])
                                
                                if ang_change < 1.0 and dist < 10.0:
                                    overtaker_id = rec_a['track_id'] if prev_diff > 0 else rec_b['track_id']
                                    overtaker_rec = rec_a if prev_diff > 0 else rec_b
                                    
                                    if overtaker_rec['velocity_ms'] > 1.0:
                                        overtaking_keys.add((frame, overtaker_id))
                                        
                    pair_states[pair_key] = (frame, lane, diff)
                    
    # Map overtaking keys to label (we label a 7-frame window around the crossover to let the model learn transition states)
    for frame, tid in overtaking_keys:
        window_indices = df[(df['track_id'] == tid) & (df['frame'] >= frame - 3) & (df['frame'] <= frame + 3)].index
        df.loc[window_indices, 'label_overtaking'] = 1
        
    # 3.4: Sudden Braking Violation Label
    # Compute rolling average velocity and acceleration check
    for track_id, group in df.groupby('track_id'):
        smooth_vel = group['velocity_ms'].rolling(window=7, min_periods=1).mean()
        accel = smooth_vel.diff() * fps
        prev_smooth_vel = smooth_vel.shift(1)
        braking_mask = (accel < -6.0) & (prev_smooth_vel > 3.0)
        df.loc[group[braking_mask].index, 'label_braking'] = 1
        
    # 3.5: Vehicle Stoppage Violation Label
    for track_id, group in df.groupby('track_id'):
        if len(group) < 90:
            continue
        prev_x = group['world_x'].shift(90)
        prev_y = group['world_y'].shift(90)
        disp = np.hypot(group['world_x'] - prev_x, group['world_y'] - prev_y)
        window_mean_vel = group['velocity_ms'].rolling(90).mean()
        stoppage_mask = (disp < 1.0) & (window_mean_vel < 0.8)
        df.loc[group[stoppage_mask].index, 'label_stoppage'] = 1

    # 3.6: Wrong Way Violation Label
    # omega < -0.1 inside circulating ring for >= 15 consecutive frames
    for track_id, group in df.groupby('track_id'):
        is_wrong = (group['omega'] < -0.1) & group['is_in_ring']
        cumsum = np.cumsum(~is_wrong)
        if is_wrong.any():
            counts = is_wrong.groupby(cumsum).transform('size')
            wrong_way_mask = is_wrong & (counts >= 15)
            df.loc[group[wrong_way_mask].index, 'label_wrong_way'] = 1

    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the feature dataframe
    # Drop intermediate columns that are not features or metadata
    cols_to_save = [
        'frame', 'track_id', 'class_name', 'world_x', 'world_y', 'velocity_ms',
        'r', 'theta', 'lane', 'is_in_ring', 'delta_theta', 'omega', 'accel',
        'disp_30', 'disp_90', 'mean_vel_30', 'mean_vel_90',
        'dist_to_leader', 'leader_vel_diff', 'leader_omega_diff', 'leader_r_diff', 'leader_track_id',
        'label_straddling', 'label_tailgating', 'label_overtaking', 'label_braking', 'label_stoppage', 'label_wrong_way'
    ]
    df_save = df[cols_to_save]
    df_save.to_csv(output_path, index=False)
    print(f"Dataset successfully prepared and saved to: {output_path}")
    print(f"Total samples: {len(df_save)}")
    for lbl in ['straddling', 'tailgating', 'overtaking', 'braking', 'stoppage', 'wrong_way']:
        count = df_save[f'label_{lbl}'].sum()
        pct = 100 * count / len(df_save)
        print(f" - {lbl} violation frames: {count} ({pct:.3f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for ML safety violations classifier")
    parser.add_argument("--csv", type=str, required=True, help="Path to input tracks CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to save output features and labels CSV")
    args = parser.parse_args()
    
    prep_dataset(args.csv, args.output)
