import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

X_C = 43.5
Y_C = 28.5
R_OUTER = 14.0

def plot_unsafe_overtaking(tracks_csv_path: str, violations_csv_path: str, output_img_path: str):
    print(f"[reporter] Loading overtaking violations from: {violations_csv_path}")
    try:
        violations_df = pd.read_csv(violations_csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{violations_csv_path}' was not found.")
        return
        
    if violations_df.empty:
        print("[reporter] No unsafe overtaking violations to plot.")
        return

    print(f"[reporter] Loading tracks from: {tracks_csv_path}")
    tracks_df = pd.read_csv(tracks_csv_path)

    # Determine scale factor dynamically from the dataset
    non_zero = tracks_df[tracks_df["center_x"] > 0]
    if not non_zero.empty:
        scale = non_zero.iloc[0]["world_x"] / non_zero.iloc[0]["center_x"]
    else:
        scale = 0.0875  # default lane-based scale: 7.0 / 80.0

    x_c = 870.0 * scale
    y_c = 570.0 * scale
    r_outer = 280.0 * scale
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot roundabout center and boundary
    ax.plot(x_c, y_c, 'k+', markersize=15, label='Roundabout Center')
    roundabout = patches.Circle((x_c, y_c), r_outer, fill=False, color='black', linestyle='--', linewidth=2, label='Roundabout Boundary')
    ax.add_patch(roundabout)
    
    colors = plt.cm.tab10.colors
    
    for idx, row in violations_df.iterrows():
        overtaking_id = int(row['track_id'])
        overtaken_id = int(row['overtaken_vehicle_id'])
        start_frame = int(row['start_frame'])
        
        c = colors[idx % len(colors)]
        
        # Extract trajectories
        tr_overtaking = tracks_df[tracks_df['track_id'] == overtaking_id].sort_values('frame')
        tr_overtaken = tracks_df[tracks_df['track_id'] == overtaken_id].sort_values('frame')
        
        if tr_overtaking.empty or tr_overtaken.empty:
            continue
            
        # Plot overtaking trajectory
        ax.plot(tr_overtaking['world_x'], tr_overtaking['world_y'], color=c, linestyle='-', linewidth=2, label=f'Track {overtaking_id} (Overtaking)')
        
        # Plot overtaken trajectory
        ax.plot(tr_overtaken['world_x'], tr_overtaken['world_y'], color=c, linestyle=':', linewidth=2, label=f'Track {overtaken_id} (Overtaken)')
        
        # Overtaking start (green marker)
        start_row = tr_overtaking.iloc[0]
        ax.plot(start_row['world_x'], start_row['world_y'], marker='o', color='green', markersize=8, label='Overtaking Start' if idx==0 else "")
        
        # Overtaken start (different marker)
        start_row_oth = tr_overtaken.iloc[0]
        ax.plot(start_row_oth['world_x'], start_row_oth['world_y'], marker='s', color='blue', markersize=8, label='Overtaken Start' if idx==0 else "")
        
        # Violation start point (red marker)
        viol_row = tr_overtaking[tr_overtaking['frame'] == start_frame]
        if not viol_row.empty:
            v_x = viol_row['world_x'].values[0]
            v_y = viol_row['world_y'].values[0]
            ax.plot(v_x, v_y, marker='X', color='red', markersize=10, label='Violation Point' if idx==0 else "")
            
    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = {k: v for k, v in by_label.items() if k != ""}
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.35, 1.0))
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('World X (m)')
    ax.set_ylabel('World Y (m)')
    ax.set_title('Unsafe Overtaking Violations')
    ax.grid(True)
    
    Path(output_img_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_img_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[reporter] Unsafe overtaking visualization saved -> {output_img_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        plot_unsafe_overtaking(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python visualize_unsafe_overtaking_plot.py <tracks_csv> <violations_csv> <output_png>")
