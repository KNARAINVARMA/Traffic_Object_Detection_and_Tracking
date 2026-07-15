# Unsafe Roundabout Shortcut Rule

This document details the algorithm and implementation of the **Unsafe Roundabout Shortcut Rule**, tailored for corner-cutting behaviors at the intersection.

---

## 1. Physical Context & Rule Description
Roundabouts are designed to force traffic through a circular path around a central island, slowing down vehicles and reducing conflict points. A common traffic violation at such intersections occurs when right-turning vehicles "cut the corner" (traveling counter-clockwise through a short arc) instead of navigating the circular path clockwise.

However, road layouts often contain physical structures that restrict where shortcuts can occur. For instance:
* **East/West Arms**: Concrete medians extend close to the central island, physically blocking right-turning vehicles from cutting the corner. They are forced to enter the intersection properly.
* **North/South Arms**: No medians are present, allowing drivers to perform highly dangerous, wrong-way corner-cut right turns:
  * **North $\rightarrow$ West**
  * **South $\rightarrow$ East**

The rule states:
> A vehicle is flagged for an unsafe roundabout shortcut if it performs a North-to-West or South-to-East right turn, cuts across the corner with a small angular traversal ($< 150^\circ$), and does so during congested traffic (at least 2 other vehicles in the roundabout).

---

## 2. Mathematical Formulation

### A. Phase-Unwrapped Angular Traversal
We convert Cartesian track coordinates relative to the roundabout center $(X_C, Y_C) = (43.5, 28.5)$ into polar coordinates:
$$\theta_i = \text{atan2}(y_i - Y_C, x_i - X_C)$$

Because $\theta_i$ wraps between $-\pi$ and $+\pi$, a vehicle circling the roundabout will experience a discontinuity. We resolve this by applying **phase unwrapping**:
$$\theta_{\text{unwrapped}} = \text{unwrap}(\theta)$$

The total angular traversal $\Delta\theta$ in degrees is:
$$\Delta\theta = |\theta_{\text{unwrapped}}[-1] - \theta_{\text{unwrapped}}[0]| \times \frac{180.0}{\pi}$$

* **Clockwise Navigation (Proper)**: A proper right-turn around the island covers $\approx 270^\circ$.
* **Shortcut (Illegal Corner Cut)**: Cutting directly across the corner covers $\approx 90^\circ$.

We apply a threshold of $\Delta\theta < 150.0^\circ$ to classify the shortcuts.

### B. Congestion Condition
To ensure shortcuts are flagged specifically during conflicting traffic conditions, the algorithm checks the frame-by-frame outer boundary occupancy. A violation is recorded only if at least $2$ other active vehicles are inside the outer radius $R_{\text{OUTER}} = 14.0$ meters at the time of the shortcut.

---

## 3. Code Implementation & Explanation

The logic is implemented in [`unsafe_roundabout_shortcut_rule.py`](file:///d:/btp/Traffic_Object_Detection_and_Tracking/src/safety/unsafe_roundabout_shortcut_rule.py).

### Path and Traversal Direction Mapping
```python
# Compass directions based on coordinate changes.
# Y increases downwards in screen coordinates, so dy > 0 points SOUTH.
def determine_direction(dx: float, dy: float) -> str:
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    return "SOUTH" if dy > 0 else "NORTH"
```
* **Explanation**: Shifting origin to $(X_C, Y_C)$ allows us to classify the vehicle's entry and exit points into compass quadrants (`NORTH`, `SOUTH`, `EAST`, `WEST`).

```python
# Strictly limit detection to corner-cut vulnerable paths
shortcut_paths = {("NORTH", "WEST"), ("SOUTH", "EAST")}

first = track.iloc[0]
last = track.iloc[-1]
entry_direction = determine_direction(first["dx"], first["dy"])
exit_direction = determine_direction(last["dx"], last["dy"])
```
* **Explanation**: This immediately filters out straight-throughs, left turns, and median-protected turns, preventing false positives.

### Angular Traversal Analysis
```python
# Calculate polar angles
theta = np.arctan2(track["dy"], track["dx"])

# Phase unwrap the angles to remove wrapping jumps at +/- pi
theta_unwrapped = np.unwrap(theta)

# Total absolute change in degrees
total_angular_change = np.abs(theta_unwrapped[-1] - theta_unwrapped[0]) * 180.0 / np.pi

is_shortcut = False
if (entry_direction, exit_direction) in shortcut_paths:
    # If the vehicle cut the corner instead of going around, angle change is low
    if total_angular_change < 150.0:
        is_shortcut = True
```
* **Explanation**: Applying `np.unwrap` allows the accumulation of continuous angular changes beyond $360^\circ$. A proper clockwise path around the roundabout results in a large angle, whereas the direct counter-clockwise cut results in a low angle change ($< 150^\circ$).

### Congestion Mapping
```python
conflict_frames = []
for _, row in track.iterrows():
    same_frame = frame_groups[row["frame"]]
    other_vehicles = same_frame[same_frame["track_id"] != track_id]
    
    # Count other vehicles inside outer boundary
    other_in_outer = (other_vehicles["r"] < R_OUTER).sum()
    if other_in_outer >= CONGESTION_THRESHOLD:
        conflict_frames.append(int(row["frame"]))
```
* **Explanation**: This steps frame-by-frame through the vehicle's track and counts other active tracks inside the $14$-meter outer zone. If the threshold is met, the frame is marked as a conflict frame.
