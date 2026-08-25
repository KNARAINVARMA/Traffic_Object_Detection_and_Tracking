# Illegal Stoppage & Obstruction Detection Rule

This document details the algorithm and implementation of the **Vehicle Stoppage Detection Rule** within the restricted circulatory zone of the roundabout.

---

## 1. Physical Context & Rule Description

Roundabout circulatory lanes are designed for continuous traffic flow. Stopping or parking a vehicle inside the circulatory ring creates severe congestion bottlenecks, forces sudden braking or erratic lane changes by trailing vehicles, and introduces high collision risks.

The rule states:
> A vehicle is flagged for a vehicle stoppage violation if it enters the circulating ring ($6.0\text{m} \le r \le 14.0\text{m}$) and maintains a 90-frame ($3.0$ seconds at $30$ FPS) spatial displacement of less than $1.0$ meter ($\Delta d_{90} < 1.0\text{m}$) alongside a 90-frame rolling mean speed of less than $0.8\text{ m/s}$ ($\bar{v}_{90} < 0.8\text{ m/s}$).

---

## 2. Mathematical Formulation

### A. Circulatory Zone & Lane Assignment
A track point $(x, y)$ is assigned a lane based on its radial distance $r = \sqrt{(x - X_C)^2 + (y - Y_C)^2}$ from the roundabout center $(X_C = 43.5\text{m}, Y_C = 28.5\text{m})$:
$$\text{Lane}(r) = \begin{cases} \text{Inner}, & 6.0\text{m} \le r < 10.0\text{m} \\ \text{Outer}, & 10.0\text{m} \le r \le 14.0\text{m} \\ \text{None}, & \text{otherwise} \end{cases}$$

### B. Bounding Box Deduplication
To eliminate false-positive violations caused by duplicate tracker detections, overlapping bounding boxes in the same frame are filtered using spatial distance and IoU/IoM checks:
* **World Distance**: $d_{\text{world}} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2} < 1.8\text{ meters}$
* **Bounding Box IoU / IoM**: $\text{IoU} > 0.20$ or $\text{IoM} > 0.40$

### C. 90-Frame Displacement & Rolling Velocity
For each active vehicle track in the circulating ring evaluated at frame $t$:
1. **Spatial Displacement ($\Delta d_{90}$)**:
   $$\Delta d_{90}(t) = \sqrt{(x(t) - x(t-90))^2 + (y(t) - y(t-90))^2} < 1.0\text{ meter}$$

2. **Rolling Mean Speed ($\bar{v}_{90}$)**:
   $$\bar{v}_{90}(t) = \frac{1}{90} \sum_{k=0}^{89} v(t-k) < 0.8\text{ m/s}$$

A violation is logged for frame $t$ whenever both conditions are simultaneously satisfied.

---

## 3. Code Implementation & Parameters

The rule is implemented in [`stoppage.py`](file:///c:/Users/k16na/Desktop/btp/src/safety/stoppage.py).

### Core Parameter Settings:
* `displacement_threshold`: `1.0` meter over 90 frames
* `mean_speed_threshold`: `0.8` m/s over 90 frames
* `window_frames`: `90` frames (3.0 seconds at 30 FPS)
* `dedup_dist_thresh`: `1.8` meters

### Python Snippet:
```python
# Check 90-frame (3-second) spatial displacement & rolling mean speed
prev_x = group['world_x'].shift(90)
prev_y = group['world_y'].shift(90)
disp = np.hypot(group['world_x'] - prev_x, group['world_y'] - prev_y)
window_mean_vel = group['velocity_ms'].rolling(90).mean()

# Stoppage rule condition
stoppage_mask = (disp < 1.0) & (window_mean_vel < 0.8)
for idx in group[stoppage_mask].index:
    row = group.loc[idx]
    stoppage_records.append({
        'violation_type': 'Vehicle Stoppage',
        'frame': int(row['frame']),
        'track_id': int(track_id),
        'class_name': row['class_name'],
        'lane': row['lane'],
        'd': float(disp.loc[idx])
    })
```

---

## 4. Execution

To run stoppage detection on track outputs:
```bash
python src/safety/stoppage.py --csv ML\ Model/data/long1_tracks_narain_cleaned_edited.csv --output stoppage_rule.csv
```
