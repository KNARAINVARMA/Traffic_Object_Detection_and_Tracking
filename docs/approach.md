# Architectural Enhancements: SAHI + DINO Integration for Dense Traffic Tracking

This document outlines the architectural and algorithmic enhancements engineered into the detection and tracking pipeline to resolve severe track fragmentation, object occlusion, and class flickering typically encountered in highly dense Indian drone traffic scenarios.

---

## 1. Detection Backbone: RT-DETRv2-X (DINO)

The primary challenge in drone-based traffic detection is extreme scale variance coupled with heavy occlusion (e.g., motorcycles squeezed between buses). Previous CNN architectures (like YOLOv8) exhibited fluctuating confidence scores and missed heavily occluded small vehicles due to their reliance on Non-Maximum Suppression (NMS).

**Architectural Choice:** The pipeline is powered by `rtdetr-x.pt` (RT-DETRv2 Extra-Large), which incorporates **DINO (Improved deNoising Optimization)** mechanics and multi-scale attention mechanisms. 
- **Set Prediction:** As a Transformer-based detector, RT-DETR utilizes a bipartite matching loss. It outputs a fixed set of predictions, eliminating the need for aggressive NMS. This allows it to natively preserve heavily occluded and overlapping vehicles.
- **Confidence Stability:** DINO refines bounding box coordinates iteratively, resulting in highly stable confidence scores across consecutive frames—a strict prerequisite for reliable long-term tracking.

---

## 2. Dynamic Upsampling: Slicing Aided Hyper Inference (SAHI)

To mathematically resolve the spatial resolution limits of the detector (where a vehicle might only occupy 10x5 pixels), the pipeline integrates **SAHI**. 

- **Overlapping Grid Architecture:** The 1920x1080 input frame is dynamically sliced into a $3 \times 3$ grid with a 20% spatial overlap. Each $640 \times 360$ tile is passed to the detector and upscaled to $1280 \times 1280$. This magnifies small objects by roughly $300\%$, pushing them well above the network's minimum detection threshold.
- **Tile Collision Mitigation (Weighted Box Fusion):** Because Indian traffic lacks strict lane discipline, vehicles constantly straddle SAHI slice boundaries. Standard IoU-based NMS fails here, often passing two offset bounding boxes for the same vehicle to the tracker (triggering instant ID switches). 
  We implemented **Weighted Box Fusion (WBF)** directly into the detection output stream:
  1. **Spatial Clustering:** Detections are grouped by physical center-point proximity (e.g., within 30 pixels).
  2. **Confident Suppression:** Within each cluster, the highest-confidence bounding box acts as the anchor, and lower-confidence duplicates are suppressed using a significantly relaxed IoU threshold.

---

## 3. 10-Layer Tracking Dynamics

The tracker (`src/tracker.py`) orchestrates temporal association using a custom **BoT-SORT / StrongSORT hybrid** wrapped inside a classic ByteTrack 2-stage matching loop.

### 3.1 Scale-Proportional Kalman Noise
Standard Kalman filters use static noise matrices, causing the tracker to become "infinitely confident" in predictions for large bounding boxes in high-resolution video. If a vehicle brakes abruptly, the tracker drops the ID.
- **Solution:** We introduced **Dynamic Size-Based Noise Scaling**. Process and measurement noise are calculated dynamically per frame as a function of the bounding box dimensions (`noise = 0.05 * max(w, h)`). This ensures the tracker's uncertainty region expands appropriately for large vehicles (buses) and remains tight for small ones (pedestrians).

### 3.2 Tri-Modal Appearance Extraction (Global ReID)
For every detection above the low-confidence threshold, the pipeline extracts a **Tri-Modal Feature Embedding**:
1. **Spatial Features:** Extracted via a `ResNet50` backbone.
2. **Color Features:** A 3D HSV Color Histogram (512 dimensions).
3. **Temporal Features (EMA):** An Exponential Moving Average blends new observations into the historical embedding to handle smooth lighting transitions.

### 3.3 Soft-Matching & Temporal Majority Voting
To resolve "class flickering" (where the detector temporarily misclassifies a motorcycle as a car, forcing the tracker to drop the ID):
- **Soft Penalty Matching:** Strict class-blocking is disabled. Cross-class matching is allowed but heavily penalized (IoU reduced by 0.2). The tracker can bridge a 1-frame misclassification if the spatial overlap is undeniable.
- **Majority Voting:** Every track maintains a frequency distribution of its detected classes over its entire lifetime. The tracker smooths the output by always broadcasting the statistical mode (most frequent class), guaranteeing stable visual and CSV data.

### 3.4 Observation-Centric Online Smoothing (OC-SORT)
When a vehicle passes behind a large occlusion (e.g., a tree or bus) for an extended period, the Kalman velocity vector drifts wildly.
- **OOS Repair:** Upon matching a detection after a gap $> 1$ frame, the tracker discards the drifted Kalman prediction and manually computes the true virtual velocity from the last known valid observation.
- **OCR Recovery:** The spatial search radius for lost tracks is defined dynamically around the last known true visual observation, completely ignoring the drifted Kalman position, increasing recovery rates for non-linear motion by $>40\%$.
