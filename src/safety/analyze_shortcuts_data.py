import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

X_C = 43.5
Y_C = 28.5

TURNING_MOVEMENTS = {
    ("EAST", "NORTH"),
    ("EAST", "SOUTH"),
    ("WEST", "NORTH"),
    ("WEST", "SOUTH"),
    ("NORTH", "EAST"),
    ("NORTH", "WEST"),
    ("SOUTH", "EAST"),
    ("SOUTH", "WEST"),
}

def determine_direction(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    return "NORTH" if dy > 0 else "SOUTH"

def analyze_trajectories(csv_file: str):
    df = pd.read_csv(csv_file)
    df["dx"] = df["world_x"] - X_C
    df["dy"] = df["world_y"] - Y_C
    df["r"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)

    results = []
    
    for track_id, track in df.groupby("track_id"):
        track = track.sort_values("frame")
        if len(track) < 6:
            continue
            
        first = track.iloc[0]
        last = track.iloc[-1]
        entry = determine_direction(first["dx"], first["dy"])
        exit = determine_direction(last["dx"], last["dy"])
        
        if (entry, exit) not in TURNING_MOVEMENTS:
            continue
            
        r_min = track["r"].min()
        r_mean = track["r"].mean()
        
        theta = np.arctan2(track["dy"], track["dx"])
        theta_unwrapped = np.unwrap(theta)
        angular_change = np.abs(theta_unwrapped[-1] - theta_unwrapped[0]) * 180.0 / np.pi
        
        results.append({
            "track_id": track_id,
            "entry": entry,
            "exit": exit,
            "movement": f"{entry}->{exit}",
            "frames": len(track),
            "r_min": r_min,
            "r_mean": r_mean,
            "angular_change": angular_change
        })
        
    res_df = pd.DataFrame(results)
    print(f"Analyzed {len(res_df)} turning vehicles.")
    
    # Save CSV
    res_df.to_csv("shortcut_analysis.csv", index=False)
    
    # Plot Scatter
    plt.figure(figsize=(10, 8))
    
    for movement in res_df["movement"].unique():
        subset = res_df[res_df["movement"] == movement]
        plt.scatter(subset["angular_change"], subset["r_min"], label=movement, alpha=0.6)
        
    plt.axvline(x=35, color='r', linestyle='--', label='Original 35 deg threshold')
    plt.axhline(y=14, color='b', linestyle='--', label='R_OUTER (14m)')
    plt.axhline(y=6, color='g', linestyle='--', label='R_INNER (6m)')
    
    plt.xlabel("Total Angular Change (degrees)")
    plt.ylabel("Minimum Radius (r_min) in meters")
    plt.title("Turning Vehicles: Angular Change vs Minimum Radius")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    out_img = "shortcut_scatter.png"
    plt.savefig(out_img, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved analysis to shortcut_analysis.csv and {out_img}")

if __name__ == "__main__":
    analyze_trajectories(r"D:\btp\narain_data\test1.csv")
