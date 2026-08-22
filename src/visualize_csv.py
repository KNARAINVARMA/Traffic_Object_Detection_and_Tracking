import pandas as pd
import cv2
import sys
import argparse
from collections import deque

CLASS_COLORS = {
    "person": (0, 0, 255),
    "car": (255, 0, 0),
    "motorcycle": (0, 255, 0),
    "bus": (0, 255, 255),
    "truck": (255, 0, 255),
}

def draw_box_label(img, bbox, label, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - baseline - 2), font, font_scale, (255, 255, 255), thickness)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Loading CSV {args.csv}...")
    df = pd.read_csv(args.csv)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    traj_buffers = {}
    print(f"Rendering Video to {args.output}...")

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_gt = df[df["frame"] == frame_idx]
        for _, row in frame_gt.iterrows():
            box = [int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])]
            cls_name = row["class_name"]
            tid = int(row["track_id"])
            
            cx, cy = int(row["center_x"]), int(row["center_y"])
            if tid not in traj_buffers:
                traj_buffers[tid] = deque(maxlen=60)
            traj_buffers[tid].append((cx, cy))
            
            color = CLASS_COLORS.get(cls_name, (200, 200, 200))
            
            pts = list(traj_buffers[tid])
            for k in range(1, len(pts)):
                alpha = k / len(pts)
                faded = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pts[k-1], pts[k], faded, 2, cv2.LINE_AA)
                
            label = f"ID:{tid}"
            draw_box_label(frame, box, label, color)
            
        writer.write(frame)
        if frame_idx % 30 == 0:
            print(f"Rendered frame {frame_idx}/{total_frames}")
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Successfully saved clean-draw video to {args.output}")

if __name__ == "__main__":
    main()
