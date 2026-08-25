import os
import pandas as pd
import numpy as np


def detect_wrong_way_violations(csv_file=None):
    if csv_file is None:
        base_dir = os.path.dirname(__file__)
        data_candidates = [
            os.path.join(base_dir, "..", "..", "data", "long1_tracks_narain_cleaned_edited.csv"),
            os.path.join(base_dir, "..", "..", "newsafety_rules", "data", "long1_tracks_narain_cleaned_edited.csv"),
            os.path.join(base_dir, "data", "long1_tracks_narain_cleaned_edited.csv"),
            r"data\long1_tracks_narain_cleaned_edited.csv",
        ]
        for candidate in data_candidates:
            if os.path.exists(candidate):
                csv_file = candidate
                break
        if csv_file is None:
            csv_file = r"data\long1_tracks_narain_cleaned_edited.csv"

    # Load dataset
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' (resolved) was not found.")
        return

    try:
        from .calibration import CENTER_X as X_c, CENTER_Y as Y_c, R_INNER as R_MIN, R_OUTER as R_MAX
    except ImportError:
        try:
            from calibration import CENTER_X as X_c, CENTER_Y as Y_c, R_INNER as R_MIN, R_OUTER as R_MAX
        except ImportError:
            X_c, Y_c, R_MIN, R_MAX = 28.5, 43.4, 6.0, 14.0

    FPS = 30.0
    OMEGA_THRESHOLD = -0.1
    CONSECUTIVE_FRAMES_THRESHOLD = 45
    MIN_SPEED = 1.5
    ALLOWED_CLASSES = ["car", "motorcycle", "bus", "truck", "van"]

    # Filter out pedestrians ('person') who walk on sidewalks
    if "class_name" in df.columns:
        df = df[df["class_name"].isin(ALLOWED_CLASSES)].copy()

    # Metric coordinates setup
    if "world_x" in df.columns and "world_y" in df.columns:
        df["x_m"] = df["world_x"]
        df["y_m"] = df["world_y"]
    else:
        df["x_m"] = df["x"]
        df["y_m"] = df["y"]

    df = df.sort_values(by=["track_id", "frame"]).reset_index(drop=True)

    # 1. Vectorized Polar Coordinates
    dx = df["x_m"].values - X_c
    dy = df["y_m"].values - Y_c
    df["r"] = np.hypot(dx, dy)
    df["theta"] = np.arctan2(dy, dx)

    # 2. Vectorized Velocity
    if "velocity_ms" not in df.columns:
        dt = 1.0 / FPS
        vx = (df.groupby("track_id")["x_m"].diff() / dt).bfill().ffill().fillna(0.0)
        vy = (df.groupby("track_id")["y_m"].diff() / dt).bfill().ffill().fillna(0.0)
        df["velocity_ms"] = np.hypot(vx, vy)

    # 3. Shortest angular change & angular velocity (Vectorized)
    theta_shift = df.groupby("track_id")["theta"].shift(1)
    delta_theta = np.arctan2(np.sin(df["theta"] - theta_shift), np.cos(df["theta"] - theta_shift))
    df["omega"] = delta_theta * FPS

    # 4. Ring, Speed & Direction Filter (Vectorized)
    df["is_in_ring"] = (df["r"] >= R_MIN) & (df["r"] <= R_MAX)
    df["is_moving"] = df["velocity_ms"] >= MIN_SPEED
    df["is_wrong_way"] = (df["omega"] < OMEGA_THRESHOLD) & df["is_in_ring"] & df["is_moving"]

    # 5. Consecutive block tracking (Vectorized)
    df["consecutive_group"] = (df["is_wrong_way"] != df.groupby("track_id")["is_wrong_way"].shift(1)).cumsum()
    
    wrong_df = df[df["is_wrong_way"]]
    if wrong_df.empty:
        print("No Wrong-Way Driving Violations detected.")
        return

    counts = wrong_df.groupby(["track_id", "consecutive_group"])["frame"].transform("count")
    valid_wrong = wrong_df[counts >= CONSECUTIVE_FRAMES_THRESHOLD]

    if valid_wrong.empty:
        print("No Wrong-Way Driving Violations detected.")
        return

    violations = {}
    for tid, grp in valid_wrong.groupby("track_id"):
        first_row = grp.iloc[0]
        c_name = first_row["class_name"] if "class_name" in first_row.index else "vehicle"
        violations[tid] = {
            "class_name": c_name,
            "start_frame": int(first_row["frame"])
        }

    print("--- Calibrated Wrong-Way Driving Violations Summary ---")
    class_counts = {}
    for vid, info in violations.items():
        c_name = info["class_name"]
        class_counts[c_name] = class_counts.get(c_name, 0) + 1
        
    print("\nTotal number of unique track IDs flagged by class:")
    for c_name, count in class_counts.items():
        print(f"- {c_name}: {count}")

    print("\nSpecific track IDs caught violating the rule:")
    for track_id, info in violations.items():
        print(f"Track ID {track_id} (Class: {info['class_name']}) - Violation started at frame: {info['start_frame']}")


if __name__ == "__main__":
    detect_wrong_way_violations()


# import os
# import pandas as pd
# import numpy as np


# def _resolve(path_str: str) -> str:
#     if not path_str or os.path.exists(path_str):
#         return path_str
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
#     cands = [
#         os.path.join(project_root, path_str),
#         os.path.join(project_root, "newsafety_rules", path_str),
#         os.path.join(project_root, "newsafety_rules", "data", os.path.basename(path_str)),
#         os.path.join(project_root, "data", os.path.basename(path_str)),
#         os.path.join(script_dir, "..", path_str),
#     ]
#     for c in cands:
#         if os.path.exists(c):
#             return os.path.abspath(c)
#     return path_str


# def detect_wrong_way_violations(csv_file=None):
#     if csv_file is None:
#         csv_file = r"data\long1_tracks_narain_cleaned_edited.csv"

#     resolved_csv = _resolve(csv_file)

#     # Load dataset
#     try:
#         df = pd.read_csv(resolved_csv)
#     except FileNotFoundError:
#         print(f"Error: The file '{csv_file}' (resolved: '{resolved_csv}') was not found.")
#         return

#     # Constants
#     X_c = 43.5
#     Y_c = 28.5
#     FPS = 30.0
#     R_MIN = 6.0
#     R_MAX = 14.0
#     OMEGA_THRESHOLD = -0.1
#     CONSECUTIVE_FRAMES_THRESHOLD = 15

#     # 1. Calculate Polar Radius 'r' and Angle 'theta'
#     # Assuming the coordinates to use are world_x and world_y
#     df['dx'] = df['world_x'] - X_c
#     df['dy'] = df['world_y'] - Y_c
#     df['r'] = np.sqrt(df['dx']**2 + df['dy']**2)
#     df['theta'] = np.arctan2(df['dy'], df['dx'])

#     # 2. Group by 'track_id' and sort by 'frame'
#     df = df.sort_values(by=['track_id', 'frame'])

#     # Flag to keep track of violations
#     violations = {}

#     for track_id, group in df.groupby('track_id'):
#         group = group.copy()
        
#         # 3. Compute shortest angular change
#         theta_shift = group['theta'].shift(1)
#         group['delta_theta'] = np.arctan2(np.sin(group['theta'] - theta_shift), np.cos(group['theta'] - theta_shift))
        
#         # 4. Calculate angular velocity
#         group['omega'] = group['delta_theta'] * FPS
        
#         # 5. Check if vehicle is inside the circulating ring and traveling wrong way
#         group['is_in_ring'] = (group['r'] >= R_MIN) & (group['r'] <= R_MAX)
#         group['is_wrong_way'] = (group['omega'] < OMEGA_THRESHOLD) & group['is_in_ring']
        
#         # 6. Track continuous frames
#         # We can find consecutive True values in 'is_wrong_way' by grouping by consecutive blocks
#         group['consecutive_group'] = (group['is_wrong_way'] != group['is_wrong_way'].shift()).cumsum()
        
#         # Filter only the wrong way frames
#         wrong_way_frames = group[group['is_wrong_way']]
        
#         # Count consecutive frames
#         if not wrong_way_frames.empty:
#             counts = wrong_way_frames.groupby('consecutive_group').size()
#             valid_groups = counts[counts >= CONSECUTIVE_FRAMES_THRESHOLD].index
            
#             if not valid_groups.empty:
#                 # Find the first frame of the first valid sequence
#                 first_valid_group = valid_groups[0]
#                 violation_start_frame = wrong_way_frames[wrong_way_frames['consecutive_group'] == first_valid_group]['frame'].iloc[0]
#                 class_name = group['class_name'].iloc[0]
                
#                 violations[track_id] = {
#                     'class_name': class_name,
#                     'start_frame': int(violation_start_frame)
#                 }

#     # Prepare outputs
#     if not violations:
#         print("No Wrong-Way Driving Violations detected.")
#         return

#     print("--- Wrong-Way Driving Violations Summary ---")
    
#     # Broken down by class_name
#     class_counts = {}
#     for vid, info in violations.items():
#         c_name = info['class_name']
#         class_counts[c_name] = class_counts.get(c_name, 0) + 1
        
#     print("\nTotal number of unique track IDs flagged by class:")
#     for c_name, count in class_counts.items():
#         print(f"- {c_name}: {count}")

#     print("\nSpecific track IDs caught violating the rule:")
#     for track_id, info in violations.items():
#         print(f"Track ID {track_id} (Class: {info['class_name']}) - Violation started at frame: {info['start_frame']}")

#     # Save output wrong_way.csv for visualization consumption
#     v_rows = [{"track_id": tid, "class_name": info["class_name"], "start_frame": info["start_frame"]} for tid, info in violations.items()]
#     out_df = pd.DataFrame(v_rows)
#     out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrong_way.csv")
#     out_df.to_csv(out_csv, index=False)


# if __name__ == "__main__":
#     dataset_file = r"data\long1_tracks_narain_cleaned_edited.csv"
#     detect_wrong_way_violations(dataset_file)


