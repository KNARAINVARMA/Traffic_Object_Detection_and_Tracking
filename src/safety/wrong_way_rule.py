import pandas as pd
import numpy as np
from pathlib import Path

def detect_wrong_way_violations(csv_file, output_csv_path=None):
    # Load dataset
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    if output_csv_path is None:
        output_csv_path = Path(__file__).resolve().parent / 'wrong_way.csv'
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Constants
    X_c = 43.5
    Y_c = 28.5
    FPS = 30.0
    R_MIN = 6.0
    R_MAX = 14.0
    OMEGA_THRESHOLD = -0.1
    CONSECUTIVE_FRAMES_THRESHOLD = 15

    # 1. Calculate Polar Radius 'r' and Angle 'theta'
    # Assuming the coordinates to use are world_x and world_y
    df['dx'] = df['world_x'] - X_c
    df['dy'] = df['world_y'] - Y_c
    df['r'] = np.sqrt(df['dx']**2 + df['dy']**2)
    df['theta'] = np.arctan2(df['dy'], df['dx'])

    # 2. Group by 'track_id' and sort by 'frame'
    df = df.sort_values(by=['track_id', 'frame'])

    # Flag to keep track of violations
    violations = {}

    for track_id, group in df.groupby('track_id'):
        group = group.copy()
        
        # 3. Compute shortest angular change
        theta_shift = group['theta'].shift(1)
        group['delta_theta'] = np.arctan2(np.sin(group['theta'] - theta_shift), np.cos(group['theta'] - theta_shift))
        
        # 4. Calculate angular velocity
        group['omega'] = group['delta_theta'] * FPS
        
        # 5. Check if vehicle is inside the circulating ring and traveling wrong way
        group['is_in_ring'] = (group['r'] >= R_MIN) & (group['r'] <= R_MAX)
        group['is_wrong_way'] = (group['omega'] < OMEGA_THRESHOLD) & group['is_in_ring']
        
        # 6. Track continuous frames
        # We can find consecutive True values in 'is_wrong_way' by grouping by consecutive blocks
        group['consecutive_group'] = (group['is_wrong_way'] != group['is_wrong_way'].shift()).cumsum()
        
        # Filter only the wrong way frames
        wrong_way_frames = group[group['is_wrong_way']]
        
        # Count consecutive frames
        if not wrong_way_frames.empty:
            counts = wrong_way_frames.groupby('consecutive_group').size()
            valid_groups = counts[counts >= CONSECUTIVE_FRAMES_THRESHOLD].index
            
            if not valid_groups.empty:
                # Find the first frame of the first valid sequence
                first_valid_group = valid_groups[0]
                violation_start_frame = wrong_way_frames[wrong_way_frames['consecutive_group'] == first_valid_group]['frame'].iloc[0]
                class_name = group['class_name'].iloc[0]
                
                violations[track_id] = {
                    'class_name': class_name,
                    'start_frame': int(violation_start_frame)
                }

    # Prepare output rows
    output_rows = []
    for track_id, info in violations.items():
        output_rows.append({
            'track_id': int(track_id),
            'class_name': info['class_name'],
            'start_frame': int(info['start_frame']),
            'violation_type': 'Wrong-Way'
        })

    output_df = pd.DataFrame(output_rows, columns=['track_id', 'class_name', 'start_frame', 'violation_type'])
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved wrong-way violation records to: {output_csv_path}")

    if output_df.empty:
        print("No Wrong-Way Driving Violations detected.")
        return output_csv_path

    print("--- Wrong-Way Driving Violations Summary ---")
    
    # Broken down by class_name
    class_counts = {}
    for vid, info in violations.items():
        c_name = info['class_name']
        class_counts[c_name] = class_counts.get(c_name, 0) + 1
        
    print("\nTotal number of unique track IDs flagged by class:")
    for c_name, count in class_counts.items():
        print(f"- {c_name}: {count}")

    print("\nSpecific track IDs caught violating the rule:")
    for track_id, info in violations.items():
        print(f"Track ID {track_id} (Class: {info['class_name']}) - Violation started at frame: {info['start_frame']}")

    return output_csv_path

if __name__ == "__main__":
    # The default execution processes the dataset specified
    dataset_file = r"D:\btp\narain_data\test1.csv"
    output_file = Path(__file__).resolve().parent / 'wrong_way.csv'
    detect_wrong_way_violations(dataset_file, output_file)
