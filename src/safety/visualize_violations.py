import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import math

def visualize_violations(
    video_path,
    tracks_csv_path,
    rule_csv_path,
    output_video_path
):
    print(f"Loading tracks from: {tracks_csv_path}")
    tracks_df = pd.read_csv(tracks_csv_path)
    
    print(f"Loading rules from: {rule_csv_path}")
    rules_df = pd.read_csv(rule_csv_path)

    # 1. Parse straddling tracks
    straddling_df = rules_df[rules_df['violation_type'] == 'Lane Straddling']
    straddling_tracks = set(straddling_df['track_id'].dropna().astype(int))
    
    # 2. Parse tailgating frames
    tailgating_df = rules_df[rules_df['violation_type'] == 'Tailgating']
    tailgating_dict = {}  # frame -> list of (follower_id, leader_id)
    
    for _, row in tailgating_df.iterrows():
        if pd.isna(row['frame']):
            continue
        frame = int(row['frame'])
        follower_id = int(row['track_id'])
        leader_id = int(row['leader_track_id']) if not pd.isna(row['leader_track_id']) else -1
        
        if frame not in tailgating_dict:
            tailgating_dict[frame] = []
        tailgating_dict[frame].append((follower_id, leader_id))
    
    # Pre-group tracks by frame for faster lookup
    print("Grouping track data by frame...")
    frames_group = tracks_df.groupby('frame')
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare VideoWriter
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Generating output video: {output_video_path}")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw bounding boxes if there are any for this frame
        if frame_idx in frames_group.groups:
            frame_data = frames_group.get_group(frame_idx)
            
            tg_data = tailgating_dict.get(frame_idx, [])
            tg_followers = {item[0] for item in tg_data}
            tg_leaders = {item[1] for item in tg_data}
            
            for _, track in frame_data.iterrows():
                tid = int(track['track_id'])
                
                is_straddling = tid in straddling_tracks
                is_tailgating_follower = tid in tg_followers
                is_tailgating_leader = tid in tg_leaders
                
                # Only draw if there's a violation
                if not (is_straddling or is_tailgating_follower or is_tailgating_leader):
                    continue
                    
                x1, y1, x2, y2 = int(track['x1']), int(track['y1']), int(track['x2']), int(track['y2'])
                
                # Determine colors and labels
                # If multiple violations, combine them
                color = (0, 0, 255) # Red default
                labels = []
                
                if is_straddling:
                    labels.append("Straddling")
                    color = (0, 0, 255) # Red
                
                if is_tailgating_follower:
                    labels.append("Tailgating")
                    color = (0, 165, 255) # Orange (BGR format: 255, 165, 0)
                elif is_tailgating_leader:
                    labels.append("Leader (TG)")
                    if not is_straddling:
                        color = (0, 255, 255) # Yellow
                
                label_str = f"ID:{tid} {' | '.join(labels)}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}")
            
        frame_idx += 1
        
    cap.release()
    out.release()
    print("Done generating visualization!")

if __name__ == "__main__":
    video_file = r"D:\btp\narain_data\test1.mp4"
    tracks_file = r"D:\btp\narain_data\test1.csv"
    rule_file = r"D:\btp\Traffic_Object_Detection_and_Tracking\src\safety\rule.csv"
    out_file = r"D:\btp\Traffic_Object_Detection_and_Tracking\outputs\video\test1_violations_only.mp4"
    
    visualize_violations(video_file, tracks_file, rule_file, out_file)
