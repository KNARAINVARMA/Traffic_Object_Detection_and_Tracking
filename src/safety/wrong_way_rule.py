import pandas as pd
import numpy as np

def detect_wrong_way_violations(csv_file):
    # Load dataset
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return

    # Determine scale factor dynamically from the dataset
    non_zero = df[df["center_x"] > 0]
    if not non_zero.empty:
        scale = non_zero.iloc[0]["world_x"] / non_zero.iloc[0]["center_x"]
    else:
        scale = 0.0875  # default lane-based scale: 7.0 / 80.0

    # Constants (originally: X_c = 870 * 0.05 = 43.5, Y_c = 570 * 0.05 = 28.5, R_MIN = 120 * 0.05 = 6.0, R_MAX = 280 * 0.05 = 14.0)
    X_c = 870.0 * scale
    Y_c = 570.0 * scale
    FPS = 30.0
    R_MIN = 120.0 * scale
    R_MAX = 280.0 * scale
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

    # Prepare outputs
    if not violations:
        print("No Wrong-Way Driving Violations detected.")
        return

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

if __name__ == "__main__":
    # The default execution processes the dataset specified
    dataset_file = r"D:\btp\narain_data\test1.csv"
    detect_wrong_way_violations(dataset_file)
