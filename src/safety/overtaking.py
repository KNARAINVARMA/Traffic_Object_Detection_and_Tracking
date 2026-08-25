"""
Unsafe Overtaking Detection Rule

This module implements the spatial-temporal overtaking detection algorithm in roundabout zones.
It uses relative polar angle sign-flips and spatial proximity filtering to identify overtaking events.
"""

import os
from pathlib import Path
from typing import Union, List, Tuple, Set, Optional
import numpy as np
import pandas as pd

# Calibration Constants
CENTER_X = 43.5
CENTER_Y = 28.5


def _resolve_csv_path(path_str: Optional[str] = None) -> str:
    """
    Resolves the dataset CSV path by searching standard locations if not found directly.
    """
    if path_str and os.path.exists(path_str):
        return os.path.abspath(path_str)
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    candidates = [
        path_str if path_str else "",
        os.path.join(project_root, "ML Model", "data", "long1_tracks_narain_cleaned_edited.csv"),
        os.path.join(project_root, "data", "long1_tracks_narain_cleaned_edited.csv"),
        os.path.join(project_root, "newsafety_rules", "data", "long1_tracks_narain_cleaned_edited.csv"),
        os.path.join(script_dir, "..", "data", "long1_tracks_narain_cleaned_edited.csv"),
        r"ML Model\data\long1_tracks_narain_cleaned_edited.csv",
        r"data\long1_tracks_narain_cleaned_edited.csv",
    ]
    
    for cand in candidates:
        if cand and os.path.exists(cand):
            return os.path.abspath(cand)
            
    raise FileNotFoundError("Could not locate tracks dataset CSV. Searched candidate paths.")


def prepare_dataframe(df: pd.DataFrame, x_c: float = CENTER_X, y_c: float = CENTER_Y) -> pd.DataFrame:
    """
    Ensures required spatial columns (r, theta, velocity_ms) exist in the dataframe.
    """
    df = df.copy()
    
    # Coordinate setup
    x_col = "world_x" if "world_x" in df.columns else ("x_m" if "x_m" in df.columns else "x")
    y_col = "world_y" if "world_y" in df.columns else ("y_m" if "y_m" in df.columns else "y")
    
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Dataframe missing spatial coordinate columns ({x_col}, {y_col}).")
        
    df["world_x"] = df[x_col]
    df["world_y"] = df[y_col]

    # Compute polar coordinates if not already present
    if "r" not in df.columns:
        df["r"] = np.hypot(df["world_x"] - x_c, df["world_y"] - y_c)
    if "theta" not in df.columns:
        df["theta"] = np.arctan2(df["world_y"] - y_c, df["world_x"] - x_c)
        
    # Velocity setup
    if "velocity_ms" not in df.columns:
        if "vx" in df.columns and "vy" in df.columns:
            df["velocity_ms"] = np.hypot(df["vx"], df["vy"])
        else:
            df["velocity_ms"] = 0.0
            
    return df


def evaluate_old_overtaking_rule(
    df,
    min_dist=0.0,
    max_dist=4.5,
    max_radial_diff=2.2,
    min_speed=0.8,
    frame_window=20,
    r_min=6.0,
    r_max=14.0
):
    """
    UPDATED OLD OVERTAKING RULE LOGIC (min_dist = 0.0m):
    -----------------------------------------------------------------------------------
    Mechanism:
    - Monitors relative polar angle difference: diff = atan2(sin(theta_b - theta_a), cos(theta_b - theta_a))
    - Detects an overtaking event ONLY IF a sign flip occurs: prev_diff * diff < 0
    - Proximity Filter: min_dist = 0.0m (floor removed), max_dist <= 4.5m, radial_diff <= 2.2m
    - Speed Floor: Overtaker velocity >= 0.8 m/s
    - Temporal Roundabout Filter: Both vehicles must be inside the roundabout (r_min <= r <= r_max)
      at exactly (frame - 20) and (frame + 20) as well as the current frame.
    
    Performance: Detects overtaking events with spatial-temporal boundary verification.
    -----------------------------------------------------------------------------------
    """
    pair_states = {}
    overtaking_events = []
    detected_pairs = set()
    
    # Pre-build lookup table for fast boundary checking: (track_id, frame) -> r
    track_frame_r = dict(zip(zip(df['track_id'], df['frame']), df['r']))

    def is_in_roundabout(track_id: int, f: int) -> bool:
        r_val = track_frame_r.get((track_id, f))
        if r_val is None:
            return False
        return r_min <= r_val <= r_max

    # Sort track records chronologically frame-by-frame
    ring_df = df.sort_values('frame')
    
    for frame, group in ring_df.groupby('frame'):
        recs = group.to_dict('records')
        n = len(recs)
        if n < 2:
            continue
            
        for i in range(n):
            for j in range(i + 1, n):
                v1, v2 = recs[i], recs[j]
                track_a = min(v1['track_id'], v2['track_id'])
                track_b = max(v1['track_id'], v2['track_id'])
                
                rec_a = v1 if v1['track_id'] == track_a else v2
                rec_b = v2 if v1['track_id'] == track_a else v1
                
                # Spatial distance and radial lane difference
                dist = np.hypot(rec_a['world_x'] - rec_b['world_x'], rec_a['world_y'] - rec_b['world_y'])
                radial_diff = np.abs(rec_a['r'] - rec_b['r'])
                
                # Proximity filter with min_dist = 0.0m floor removed
                valid_distance = (min_dist <= dist <= max_dist) and (radial_diff <= max_radial_diff)
                
                # Relative angular position between pair
                theta_a = rec_a['theta']
                theta_b = rec_b['theta']
                diff = np.arctan2(np.sin(theta_b - theta_a), np.cos(theta_b - theta_a))
                pair_key = (track_a, track_b)
                
                if pair_key in pair_states:
                    prev_frame, prev_diff = pair_states[pair_key]
                    
                    if frame - prev_frame <= 10:
                        # SIGN-FLIP CONDITION: Checks if relative angle crossed 0
                        if prev_diff * diff < 0:
                            if prev_diff > 0:
                                overtaker = rec_a
                                overtaken = rec_b
                            else:
                                overtaker = rec_b
                                overtaken = rec_a
                                
                            if overtaker['velocity_ms'] >= min_speed and valid_distance:
                                ot_id = overtaker['track_id']
                                ok_id = overtaken['track_id']

                                # Temporal Roundabout Filter: Check boundary conditions at f - 20, f, and f + 20
                                f_prev = frame - frame_window
                                f_next = frame + frame_window

                                in_rb_prev = is_in_roundabout(ot_id, f_prev) and is_in_roundabout(ok_id, f_prev)
                                in_rb_curr = is_in_roundabout(ot_id, frame) and is_in_roundabout(ok_id, frame)
                                in_rb_next = is_in_roundabout(ot_id, f_next) and is_in_roundabout(ok_id, f_next)

                                if in_rb_prev and in_rb_curr and in_rb_next:
                                    overtaking_events.append((frame, ot_id, ok_id))
                                    detected_pairs.add((ot_id, ok_id))
                                
                if valid_distance:
                    pair_states[pair_key] = (frame, diff)
                    
    return overtaking_events, detected_pairs


def run_overtaking_detection(
    csv_file: Optional[str] = None,
    output_csv: str = "unsafe_overtaking_violations.csv",
    min_dist: float = 0.0,
    max_dist: float = 4.5,
    max_radial_diff: float = 2.2,
    min_speed: float = 0.8,
    frame_window: int = 20,
    r_min: float = 6.0,
    r_max: float = 14.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the overtaking rule evaluation on the dataset and exports violation records.
    """
    resolved_path = _resolve_csv_path(csv_file)
    print(f"Loading track dataset from: {resolved_path}")
    df_raw = pd.read_csv(resolved_path)
    
    df = prepare_dataframe(df_raw)
    
    print("Evaluating overtaking rule...")
    events, detected_pairs = evaluate_old_overtaking_rule(
        df,
        min_dist=min_dist,
        max_dist=max_dist,
        max_radial_diff=max_radial_diff,
        min_speed=min_speed,
        frame_window=frame_window,
        r_min=r_min,
        r_max=r_max
    )
    
    # Format events into DataFrame
    events_df = pd.DataFrame(events, columns=["frame", "track_id", "overtaken_vehicle_id"])
    events_df["violation_type"] = "Unsafe Overtaking"
    
    # Merge class_name of overtaker if present
    if "class_name" in df.columns:
        class_map = df.groupby("track_id")["class_name"].first().to_dict()
        events_df["class_name"] = events_df["track_id"].map(class_map)
        
    pairs_df = pd.DataFrame(sorted(list(detected_pairs)), columns=["overtaker_id", "overtaken_id"])
    
    print("\n--- Overtaking Detection Summary ---")
    print(f"Total Overtaking Violation Events: {len(events_df)}")
    print(f"Unique (Overtaker, Overtaken) Pairs Detected: {len(pairs_df)}")
    
    if not events_df.empty:
        print("\nSample Overtaking Violations (First 10):")
        print(events_df.head(10).to_string(index=False))
        
        # Save output CSV
        events_df.to_csv(output_csv, index=False)
        print(f"\nSaved violation records to '{output_csv}'")
        
        # Also save rule.csv if output directory is current working directory
        rule_csv_path = "rule.csv"
        events_df.to_csv(rule_csv_path, index=False)
        print(f"Saved violation records to '{rule_csv_path}'")
    else:
        print("No overtaking violations detected.")
        
    return events_df, pairs_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Unsafe Overtaking Detection Rule")
    parser.add_argument("--csv", type=str, default=None, help="Path to input tracks CSV")
    parser.add_argument("--output", type=str, default="unsafe_overtaking_violations.csv", help="Path to output violations CSV")
    parser.add_argument("--min-dist", type=float, default=0.0, help="Minimum spatial distance (m)")
    parser.add_argument("--max-dist", type=float, default=4.5, help="Maximum spatial distance (m)")
    parser.add_argument("--max-radial-diff", type=float, default=2.2, help="Maximum radial lane difference (m)")
    parser.add_argument("--min-speed", type=float, default=0.8, help="Minimum overtaker speed (m/s)")
    parser.add_argument("--frame-window", type=int, default=20, help="Frames before (-20) and after (+20) to verify roundabout location")
    parser.add_argument("--r-min", type=float, default=6.0, help="Inner roundabout radius (m)")
    parser.add_argument("--r-max", type=float, default=14.0, help="Outer roundabout radius (m)")
    
    args = parser.parse_args()
    
    run_overtaking_detection(
        csv_file=args.csv,
        output_csv=args.output,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
        max_radial_diff=args.max_radial_diff,
        min_speed=args.min_speed,
        frame_window=args.frame_window,
        r_min=args.r_min,
        r_max=args.r_max
    )

