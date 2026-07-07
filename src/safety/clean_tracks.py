import argparse
import pandas as pd
import numpy as np
import os

def compute_iou(b1, b2):
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Clean duplicate detections from tracking CSV")
    parser.add_argument("--input", required=True, help="Path to input tracks CSV")
    parser.add_argument("--output", required=True, help="Path to output cleaned tracks CSV")
    parser.add_argument("--iou-threshold", type=float, default=0.40, help="IoU threshold for duplicate detection")
    args = parser.parse_args()

    print(f"[cleaner] Reading {args.input} ...")
    df = pd.read_csv(args.input)
    orig_len = len(df)

    print("[cleaner] Filtering overlapping duplicate detections (IoU > {:.2f}) ...".format(args.iou_threshold))
    cleaned_rows = []
    
    # Process frame-by-frame
    for frame, group in df.groupby('frame'):
        recs = group.to_dict('records')
        n = len(recs)
        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if not keep[j]:
                    continue
                box1 = [recs[i]['x1'], recs[i]['y1'], recs[i]['x2'], recs[i]['y2']]
                box2 = [recs[j]['x1'], recs[j]['y1'], recs[j]['x2'], recs[j]['y2']]
                
                # Fast coordinate bounding box overlap check
                if (box1[2] < box2[0] or box2[2] < box1[0] or box1[3] < box2[1] or box2[3] < box1[1]):
                    continue
                    
                iou = compute_iou(box1, box2)
                if iou > args.iou_threshold:
                    # Drop the duplicate with lower confidence
                    if recs[i]['confidence'] >= recs[j]['confidence']:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break # break outer comparison for recs[i] since it's dropped

        for k in range(n):
            if keep[k]:
                cleaned_rows.append(recs[k])

    df_clean = pd.DataFrame(cleaned_rows)
    clean_len = len(df_clean)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_clean.to_csv(args.output, index=False)
    print(f"[cleaner] Saved cleaned tracks to {args.output}")
    print(f"[cleaner] Removed {orig_len - clean_len} duplicates ({orig_len} -> {clean_len} rows)")

if __name__ == "__main__":
    main()
