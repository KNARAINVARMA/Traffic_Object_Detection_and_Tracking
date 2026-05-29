"""
finetune_motorcycles.py — Motorcycle-Focused Fine-Tuning Pipeline

This script implements a dedicated fine-tuning pipeline optimized for detecting
motorcycles in drone footage. It performs:
  1. Dataset scanning & identification of motorcycle-heavy / difficult frames.
  2. Oversampling of difficult frames (occluded, clustered, small scale).
  3. Hard-negative mining (including non-motorcycle frames to avoid false positives).
  4. Augmentation pipeline (random scaling, blur, brightness, JPEG compression, motion blur).
  5. Automatic creation of a structured YOLO dataset (images/labels splits) and data.yaml.
  6. Configuration and launching of the YOLOv8/RT-DETR training loop with focal loss (fl_gamma=2.0).
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from utils import ensure_dir, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image Augmentation Functions
# ---------------------------------------------------------------------------

def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    kernel_size = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_motion_blur(image: np.ndarray) -> np.ndarray:
    size = random.choice([3, 5, 7])
    kernel = np.zeros((size, size))
    # Randomly make horizontal or vertical motion kernel
    if random.choice([True, False]):
        kernel[int((size - 1) / 2), :] = np.ones(size)
    else:
        kernel[:, int((size - 1) / 2)] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)


def apply_brightness_change(image: np.ndarray) -> np.ndarray:
    factor = random.uniform(0.6, 1.4)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_jpeg_compression(image: np.ndarray) -> np.ndarray:
    quality = random.randint(30, 80)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    return cv2.imdecode(encimg, 1)


def apply_random_scaling_and_crop(
    image: np.ndarray,
    labels: List[List[float]],
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Scale the image randomly between 0.8x and 1.2x.
    Adjust bounding boxes accordingly, clipping to image boundaries.
    """
    H, W = image.shape[:2]
    scale = random.uniform(0.8, 1.2)
    new_H, new_W = int(round(H * scale)), int(round(W * scale))
    resized = cv2.resize(image, (new_W, new_H))
    
    new_labels = []
    for label in labels:
        cls_id, x_c, y_c, w, h = label
        
        # Scaling does not change relative centers if no crop is applied,
        # but if we crop/pad back to original size (H, W), we must adjust:
        if scale > 1.0:
            # Crop resized image to original H, W centered
            start_y = (new_H - H) // 2
            start_x = (new_W - W) // 2
            cropped = resized[start_y:start_y + H, start_x:start_x + W]
            
            # Map normalized boxes back to absolute, offset, and re-normalize
            abs_x_c = x_c * new_W - start_x
            abs_y_c = y_c * new_H - start_y
            abs_w = w * new_W
            abs_h = h * new_H
            
            x_c_new = abs_x_c / W
            y_c_new = abs_y_c / H
            w_new = abs_w / W
            h_new = abs_h / H
        else:
            # Pad resized image back to original H, W
            cropped = np.zeros((H, W, 3), dtype=np.uint8)
            pad_y = (H - new_H) // 2
            pad_x = (W - new_W) // 2
            cropped[pad_y:pad_y + new_H, pad_x:pad_x + new_W] = resized
            
            # Map normalized boxes back to absolute, offset, and re-normalize
            abs_x_c = x_c * new_W + pad_x
            abs_y_c = y_c * new_H + pad_y
            abs_w = w * new_W
            abs_h = h * new_H
            
            x_c_new = abs_x_c / W
            y_c_new = abs_y_c / H
            w_new = abs_w / W
            h_new = abs_h / H
            
        # Clip coordinates
        x1 = max(0.0, min(1.0, x_c_new - w_new / 2.0))
        y1 = max(0.0, min(1.0, y_c_new - h_new / 2.0))
        x2 = max(0.0, min(1.0, x_c_new + w_new / 2.0))
        y2 = max(0.0, min(1.0, y_c_new + h_new / 2.0))
        
        w_clipped = x2 - x1
        h_clipped = y2 - y1
        x_c_clipped = (x1 + x2) / 2.0
        y_c_clipped = (y1 + y2) / 2.0
        
        if w_clipped > 0.005 and h_clipped > 0.005:
            new_labels.append([cls_id, x_c_clipped, y_c_clipped, w_clipped, h_clipped])
            
    return cropped, new_labels


def apply_augmentations(image: np.ndarray, labels: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    """Apply a sequence of random augmentations."""
    aug_img = image.copy()
    aug_labels = [list(lbl) for lbl in labels]
    
    # Scale first
    if random.choice([True, False]):
        aug_img, aug_labels = apply_random_scaling_and_crop(aug_img, aug_labels)
        
    # Apply single image degradation
    deg_func = random.choice([
        apply_gaussian_blur,
        apply_motion_blur,
        apply_brightness_change,
        apply_jpeg_compression,
        None
    ])
    if deg_func is not None:
        aug_img = deg_func(aug_img)
        
    return aug_img, aug_labels


# ---------------------------------------------------------------------------
# Dataset Scoring & Oversampling Logic
# ---------------------------------------------------------------------------

def parse_yolo_labels(label_file: Path) -> List[List[float]]:
    labels = []
    if not label_file.exists():
        return labels
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                labels.append([
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                ])
    return labels


def score_frame_difficulty(labels: List[List[float]], img_w: int, img_h: int) -> float:
    """
    Score a frame's priority based on motorcycle density and difficulty factors:
      - Small motorcycle (area < 1200 pixels)
      - Clustered / occluded motorcycle (center distance to other vehicles < 120 pixels)
    """
    motos = [lbl for lbl in labels if lbl[0] == 3]
    if not motos:
        return 0.0  # Basic background / hard-negative
        
    score = 1.0  # Base score for containing motorcycles
    score += len(motos) * 0.5  # Heavy density weight
    
    other_vehs = [lbl for lbl in labels if lbl[0] in [2, 5, 7]]  # Cars, buses, trucks
    
    for m in motos:
        m_x_c, m_y_c, m_w, m_h = m[1], m[2], m[3], m[4]
        m_abs_w = m_w * img_w
        m_abs_h = m_h * img_h
        m_area = m_abs_w * m_abs_h
        
        # Small scale difficulty
        if m_area < 1200.0:
            score += 1.5
            
        # Occlusion / Proximity check
        m_cx, m_cy = m_x_c * img_w, m_y_c * img_h
        for ov in other_vehs:
            ov_cx, ov_cy = ov[1] * img_w, ov[2] * img_h
            dist = np.hypot(m_cx - ov_cx, m_cy - ov_cy)
            if dist < 120.0:
                score += 2.0  # High penalty for motorcycle clustered with car/truck
                
    return score


# ---------------------------------------------------------------------------
# Fine-Tuning Orchestrator
# ---------------------------------------------------------------------------

def prepare_dataset(
    frames_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
) -> Path:
    """
    Scan data/, perform difficulty-based oversampling, apply cv2 augmentations,
    include hard negatives, and structured YOLO splits.
    """
    logger.info("Initializing fine-tuning dataset preparation...")
    
    # Define directories
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"
    val_img_dir = output_dir / "images" / "val"
    val_lbl_dir = output_dir / "labels" / "val"
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        if d.exists():
            shutil.rmtree(d)
        ensure_dir(str(d))
        
    # Scan frames and labels
    frame_files = sorted(list(frames_dir.glob("*.jpg")))
    if not frame_files:
        logger.error("No training frame images found in: %s", frames_dir)
        sys.exit(1)
        
    # Score all frames
    scored_frames = []
    motorcycle_frames = []
    hard_negative_frames = []
    
    # Pick a dummy size to score
    dummy_img = cv2.imread(str(frame_files[0]))
    img_h, img_w = dummy_img.shape[:2]
    
    for f in frame_files:
        lbl_file = annotations_dir / f"{f.stem}.txt"
        labels = parse_yolo_labels(lbl_file)
        score = score_frame_difficulty(labels, img_w, img_h)
        
        has_motorcycle = any(lbl[0] == 3 for lbl in labels)
        if has_motorcycle:
            motorcycle_frames.append((f, lbl_file, labels, score))
            scored_frames.append((f, lbl_file, labels, score))
        else:
            hard_negative_frames.append((f, lbl_file, labels))
            
    logger.info(
        "Scanned dataset: %d motorcycle frames, %d potential hard-negatives.",
        len(motorcycle_frames), len(hard_negative_frames)
    )
    
    # Split motorcycle frames into Train (80%) and Val (20%)
    random.seed(42)
    random.shuffle(motorcycle_frames)
    split_idx = int(len(motorcycle_frames) * 0.8)
    train_motos = motorcycle_frames[:split_idx]
    val_motos = motorcycle_frames[split_idx:]
    
    # Copy baseline validation frames directly (no augmentations to validate raw accuracy)
    for f, lbl_f, labels, _ in val_motos:
        shutil.copy(str(f), val_img_dir / f.name)
        if lbl_f.exists():
            shutil.copy(str(lbl_f), val_lbl_dir / lbl_f.name)
        else:
            # Write empty label file if none exists
            with open(val_lbl_dir / f"{f.stem}.txt", "w") as empty_f:
                pass
                
    # Prepare training split with oversampling and augmentations
    copied_count = 0
    augmented_count = 0
    
    for f, lbl_f, labels, score in train_motos:
        # Load image
        img = cv2.imread(str(f))
        
        # Determine replication factor based on difficulty score
        # Base: 1, Highly difficult: up to 5 copies
        replications = max(1, min(5, int(score // 3.0) + 1))
        
        for rep in range(replications):
            if rep == 0:
                # Save original raw training sample
                shutil.copy(str(f), train_img_dir / f.name)
                if lbl_f.exists():
                    shutil.copy(str(lbl_f), train_lbl_dir / lbl_f.name)
                else:
                    with open(train_lbl_dir / f"{f.stem}.txt", "w") as empty_f:
                        pass
                copied_count += 1
            else:
                # Save augmented training sample
                aug_img, aug_labels = apply_augmentations(img, labels)
                
                # Write augmented image
                aug_filename = f"{f.stem}_aug_{rep}.jpg"
                cv2.imwrite(str(train_img_dir / aug_filename), aug_img)
                
                # Write augmented labels
                aug_lbl_filename = f"{f.stem}_aug_{rep}.txt"
                with open(train_lbl_dir / aug_lbl_filename, "w") as aug_lbl_f:
                    for l in aug_labels:
                        aug_lbl_f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")
                
                augmented_count += 1
                
    # Add Hard Negatives to Training Set (to teach class separation and reduce false-positive rate)
    # Include up to 20% of the training count as hard-negatives
    max_hard_negs = max(2, int(copied_count * 0.20))
    selected_hard_negs = random.sample(hard_negative_frames, min(len(hard_negative_frames), max_hard_negs))
    
    for f, lbl_f, labels in selected_hard_negs:
        shutil.copy(str(f), train_img_dir / f.name)
        if lbl_f.exists():
            shutil.copy(str(lbl_f), train_lbl_dir / lbl_f.name)
        else:
            with open(train_lbl_dir / f"{f.stem}.txt", "w") as empty_f:
                pass
        copied_count += 1
        
    logger.info(
        "Dataset prepared: Train=%d (baseline: %d, augmented: %d, hard-negs: %d), Val=%d.",
        copied_count + augmented_count, copied_count - len(selected_hard_negs),
        augmented_count, len(selected_hard_negs), len(val_motos)
    )
    
    # Create data.yaml
    yaml_file = output_dir / "data.yaml"
    with open(yaml_file, "w") as yf:
        yf.write(f"path: {output_dir.resolve().as_posix()}\n")
        yf.write("train: images/train\n")
        yf.write("val: images/val\n\n")
        yf.write("names:\n")
        yf.write("  0: person\n")
        yf.write("  2: car\n")
        yf.write("  3: motorcycle\n")
        yf.write("  5: bus\n")
        yf.write("  7: truck\n")
        
    logger.info("YOLO configuration file written: %s", yaml_file)
    return yaml_file


def run_training(
    yaml_file: Path,
    base_model_path: str,
    epochs: int,
    batch_size: int,
    imgsz: int,
    device: str,
) -> None:
    logger.info("Initializing fine-tuning training model '%s'...", base_model_path)
    model = YOLO(base_model_path)
    
    # Train using focal loss parameter (fl_gamma=2.0)
    logger.info("Launching YOLO training. Epochs=%d, Batch=%d, imgsz=%d", epochs, batch_size, imgsz)
    model.train(
        data=str(yaml_file),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        fl_gamma=2.0,  # focal loss tuning specifically for motorcycle hard-examples
        plots=True,
        workers=2,
    )
    logger.info("Training complete. Fine-tuned weights saved in runs/detect/train/weights/best.pt.")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="finetune_motorcycles",
        description="Run the motorcycle-focused detection fine-tuning pipeline.",
    )
    p.add_argument("--frames-dir", default="data/frames", help="Dir containing training frame images.")
    p.add_argument("--annotations-dir", default="data/annotations", help="Dir containing YOLO labels.")
    p.add_argument("--output-dataset-dir", default="data/finetuning", help="Dir to prepare the YOLO splits.")
    p.add_argument("--model", default="rtdetr-l.pt", help="Base model to finetune: 'yolov8m.pt' or 'rtdetr-l.pt'.")
    p.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    p.add_argument("--batch", type=int, default=4, help="Training batch size.")
    p.add_argument("--imgsz", type=int, default=1280, help="YOLO model input image size.")
    p.add_argument("--device", default="0", help="CUDA device index, or 'cpu' / 'mps'.")
    args = p.parse_args()
    
    setup_logging(level=logging.INFO)
    
    frames_path = Path(args.frames_dir)
    annotations_path = Path(args.annotations_dir)
    output_dataset_path = Path(args.output_dataset_dir)
    
    # 1. Prepare and oversample the dataset
    yaml_file = prepare_dataset(frames_path, annotations_path, output_dataset_path)
    
    # 2. Run the Ultralytics training loop
    run_training(
        yaml_file=yaml_file,
        base_model_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
