import pandas as pd
import numpy as np

# Load the dataset
csv_path = r'D:\btp\narain_data\test1.csv'
df = pd.read_csv(csv_path)

# Determine the scale factor dynamically from the data
non_zero = df[df['center_x'] > 0]
if not non_zero.empty:
    scale = non_zero.iloc[0]['world_x'] / non_zero.iloc[0]['center_x']
else:
    scale = 0.0875  # default lane-based scale: 7.0 / 80.0

# Parameters (originally: X_c = 870 * 0.05 = 43.5, Y_c = 570 * 0.05 = 28.5)
X_c = 870.0 * scale
Y_c = 570.0 * scale
fps = 30
dt = 1/30

# Step 1: Coordinate Conversion & Lane Assignment
df['r'] = np.sqrt((df['world_x'] - X_c)**2 + (df['world_y'] - Y_c)**2)
df['theta'] = np.arctan2(df['world_y'] - Y_c, df['world_x'] - X_c)

# Recalculate radial boundaries based on scale
# Originally: inner=120px * scale, boundary=200px * scale, outer=280px * scale
r_inner = 120.0 * scale
r_boundary = 200.0 * scale
r_outer = 280.0 * scale

def assign_lane(r):
    if r_inner <= r < r_boundary:
        return 'Inner'
    elif r_boundary <= r <= r_outer:
        return 'Outer'
    else:
        return 'None'

df['lane'] = df['r'].apply(assign_lane)

# Filter out vehicles not in the circulating ring
ring_df = df[df['lane'] != 'None'].copy()

# 1. Total unique track IDs that entered the circulating ring
unique_ring_tracks = ring_df['track_id'].nunique()

# Step 2: Detect Part A - Lane Straddling Violation
ring_df['is_straddling'] = (np.abs(ring_df['r'] - r_boundary) <= 0.5)

straddling_violations = []
for track_id, group in ring_df.groupby('track_id'):
    group = group.sort_values('frame')
    # find consecutive true values in 'is_straddling'
    is_strad = group['is_straddling'].values
    # trick to find length of consecutive True
    cumsum = np.cumsum(~is_strad)
    # group by cumsum and count max size where is_strad is True
    max_consecutive = 0
    if is_strad.any():
        counts = pd.Series(cumsum[is_strad]).value_counts()
        if not counts.empty:
            max_consecutive = counts.max()
    
    if max_consecutive >= 30:
        class_name = group['class_name'].iloc[0]
        straddling_violations.append({'track_id': track_id, 'class_name': class_name})

straddling_df = pd.DataFrame(straddling_violations)

# Step 3: Detect Part B - Tailgating / Proximity Violation
tailgating_records = []

# Group frame-by-frame, then by lane
for (frame, lane), group in ring_df.groupby(['frame', 'lane']):
    if len(group) < 2:
        continue
    # Sort by theta
    sorted_group = group.sort_values('theta').to_dict('records')
    n = len(sorted_group)
    
    for i in range(n):
        follower = sorted_group[i]
        leader = sorted_group[(i + 1) % n]
        
        delta_theta = (leader['theta'] - follower['theta']) % (2 * np.pi)
        d = ((leader['r'] + follower['r']) / 2) * delta_theta
        
        if d < 4.0 and follower['velocity_ms'] > 1.0:
            tailgating_records.append({
                'frame': frame,
                'follower_track_id': follower['track_id'],
                'leader_track_id': leader['track_id'],
                'lane': lane,
                'd': d,
                'class_name': follower['class_name']
            })

tailgating_df = pd.DataFrame(tailgating_records)

# Outputs
print(f"1. Total unique track IDs that entered the circulating ring: {unique_ring_tracks}")

print("\n2. Lane Straddling Violations by class_name:")
if not straddling_df.empty:
    print(straddling_df['class_name'].value_counts().to_string())
else:
    print("None")

print("\n3. Tailgating Violations by class_name:")
if not tailgating_df.empty:
    # We want unique track IDs flagged for tailgating
    unique_tailgating = tailgating_df.drop_duplicates('follower_track_id')
    print(unique_tailgating['class_name'].value_counts().to_string())
else:
    print("None")

print("\n4. Sample summary dataframe showing 10 random frames where a Tailgating Violation occurred:")
if not tailgating_df.empty:
    sample_size = min(10, len(tailgating_df))
    sample_df = tailgating_df[['frame', 'follower_track_id', 'leader_track_id', 'lane', 'd']].sample(sample_size, random_state=42)
    print(sample_df.to_string(index=False))
else:
    print("No tailgating violations found.")

# Export to rule.csv
if not straddling_df.empty:
    straddling_export = straddling_df.copy()
    straddling_export['violation_type'] = 'Lane Straddling'
    straddling_export['frame'] = np.nan
    straddling_export['leader_track_id'] = np.nan
    straddling_export['lane'] = np.nan
    straddling_export['d'] = np.nan
else:
    straddling_export = pd.DataFrame()

if not tailgating_df.empty:
    tailgating_export = tailgating_df.copy()
    tailgating_export['violation_type'] = 'Tailgating'
    tailgating_export = tailgating_export.rename(columns={'follower_track_id': 'track_id'})
else:
    tailgating_export = pd.DataFrame()

combined_df = pd.concat([straddling_export, tailgating_export], ignore_index=True)
# Reorder columns for better readability
columns_order = ['violation_type', 'frame', 'track_id', 'leader_track_id', 'class_name', 'lane', 'd']
# Only keep columns that exist
columns_order = [c for c in columns_order if c in combined_df.columns]
combined_df = combined_df[columns_order]

combined_df.to_csv('rule.csv', index=False)
print("\n5. All violations have been successfully saved to 'rule.csv'.")
