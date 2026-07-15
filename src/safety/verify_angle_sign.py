import pandas as pd
import numpy as np

df = pd.read_csv('shortcut_analysis.csv')
original_df = pd.read_csv(r'D:\btp\narain_data\test1.csv')

try:
    from .calibration import CENTER_X as X_C, CENTER_Y as Y_C
except ImportError:
    from calibration import CENTER_X as X_C, CENTER_Y as Y_C

results = []
for tid in df['track_id'].values:
    track = original_df[original_df['track_id'] == tid].sort_values('frame')
    theta = np.unwrap(np.arctan2(track['world_y'] - Y_C, track['world_x'] - X_C))
    delta_theta = (theta.iloc[-1] - theta.iloc[0]) * 180 / np.pi
    movement = df[df['track_id'] == tid]['movement'].values[0]
    results.append({'track_id': tid, 'movement': movement, 'delta_theta': delta_theta})

res_df = pd.DataFrame(results)
print("Negative delta_theta (Counter-Clockwise / Shortcuts):")
print(res_df[res_df['delta_theta'] < -10].head(20))

print("\nPositive delta_theta (Clockwise / Proper):")
print(res_df[res_df['delta_theta'] > 10].head(20))
