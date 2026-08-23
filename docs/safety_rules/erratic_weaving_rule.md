# Erratic Lane Weaving Rule

This document details the algorithm and implementation of the **Erratic Lane Weaving Detection Rule** (`jittering_rule.py`) within the restricted circulatory roundabout zone.

---

## 1. Physical Context & Rule Description
Roundabouts are designed to have smooth, continuous traffic flow. Frequent lane switching or erratic swerving (weaving) within the circulatory ring disrupts this flow and drastically increases the likelihood of sideswipe collisions.

The rule states:
> A vehicle is flagged for an erratic lane weaving violation if it physically crosses the boundary dividing the inner and outer lanes 3 or more times within a sliding window of 90 frames (3 seconds), whilst remaining inside the circulatory intersection.

---

## 2. Mathematical Formulation

### A. Radial Position Tracking
Using the centralized calibration settings, the metric center of the roundabout is defined as $(X_C, Y_C) = (43.5, 28.5)$.
For every vehicle track $i$ at frame $t$, we compute its metric radial distance $r_{i,t}$ from the center:
$$r_{i,t} = \sqrt{(x_{i,t} - X_C)^2 + (y_{i,t} - Y_C)^2}$$

The roundabout circulatory zone is strictly defined between:
- **Inner Boundary:** $R_{\text{INNER}} = 6.0\text{ meters}$
- **Outer Boundary:** $R_{\text{OUTER}} = 14.0\text{ meters}$
- **Lane Divider:** $R_{\text{LANE\_DIVIDER}} = 10.0\text{ meters}$

### B. Discrete State Mapping
Instead of tracking raw lateral velocity, we discretize the vehicle's position into a binary state representing its current lane:
$$
S_{i,t} = 
\begin{cases} 
1 & \text{if } r_{i,t} \geq R_{\text{LANE\_DIVIDER}} \text{ (Outer Lane)} \\
0 & \text{if } r_{i,t} < R_{\text{LANE\_DIVIDER}} \text{ (Inner Lane)}
\end{cases}
$$

### C. Transition Detection & Aggregation
A lane-crossing transition event $T_{i,t}$ occurs when the vehicle's state changes from the previous frame:
$$
T_{i,t} = 
\begin{cases} 
1 & \text{if } S_{i,t} \neq S_{i,t-1} \text{ and } r_{i,t} \in [R_{\text{INNER}}, R_{\text{OUTER}}] \\
0 & \text{otherwise}
\end{cases}
$$

To detect erratic weaving, we sum the transitions over a sliding 90-frame rolling window:
$$\text{Total Transitions}_{i,t} = \sum_{k=0}^{89} T_{i, t-k}$$

A violation is triggered if $\text{Total Transitions}_{i,t} \geq 3$.

---

## 3. Implementation Snippet
```python
# Calculate Radial Distance r
dx = df_out["x_m"].values - center_x
dy = df_out["y_m"].values - center_y
r = np.hypot(dx, dy)

# Filter mask: inside roundabout bounds
in_ring = (r >= r_min) & (r <= r_max)

# Track positional state (1 = Outer, 0 = Inner)
state = (r >= r_lane_divider).astype(int)
df_out["_state"] = state

# Detect state transitions (0->1 or 1->0)
prev_state = df_out.groupby("track_id")["_state"].shift(1)
has_prev = prev_state.notna()
is_transition = (df_out["_state"] != prev_state) & has_prev & in_ring
df_out["_transition"] = is_transition.astype(int)

# Group by track_id and apply rolling window of 90 frames
rolling_transitions = (
    df_out.groupby("track_id")["_transition"]
    .transform(lambda s: s.rolling(window=window_frames, min_periods=1).sum())
)

# Trigger: Flag violation if transitions >= min_transitions
df_out["is_erratic_weaving"] = (rolling_transitions >= min_transitions) & in_ring
```
