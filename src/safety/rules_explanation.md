# Roundabout Safety Rules and Algorithms Explanation

This document provides a detailed mathematical and code-level explanation of the roundabout safety rules, vehicle dynamics rules, conflict detection models, and classification algorithms implemented in this project (excluding all machine learning layers).

---

# I. Roundabout Tracking & Vehicle Dynamics Rules

The roundabout safety checks analyze coordinates relative to the center of the roundabout circle.
* Roundabout Center Point (Xc, Yc): (43.5, 28.5)
* Circulating Ring Boundaries: Inner radius R_min = 6.0 meters, Outer radius R_max = 14.0 meters.

## 1. Wrong-Way Driving Violation
This rule flags vehicles driving in the wrong direction inside the roundabout (clockwise instead of counter-clockwise).

### Mathematical Formulation
1. Polar Coordinates:
   Convert cartesian coordinates (x(t), y(t)) to polar coordinates (r(t), theta(t)):
   r(t) = sqrt((x(t) - X_c)^2 + (y(t) - Y_c)^2)
   theta(t) = atan2(y(t) - Y_c, x(t) - X_c)

2. Shortest Angular Difference:
   Calculate the angular change between successive frames in the range [-pi, pi]:
   delta_theta(t) = atan2(sin(theta(t) - theta(t-1)), cos(theta(t) - theta(t-1)))

3. Angular Velocity (omega):
   omega(t) = delta_theta(t) * FPS

4. Violation Constraint:
   The vehicle must be within the circulating ring and its angular velocity must be below a negative threshold (denoting clockwise rotation) for a continuous sequence:
   (R_min <= r(t) <= R_max) AND (omega(t) < omega_threshold)
   
   Where:
   * R_min = 6.0 m
   * R_max = 14.0 m
   * omega_threshold = -0.1 rad/s
   * Consecutive Frames threshold >= 15 frames

### Code Implementation
```python
# Calculate Polar Radius 'r' and Angle 'theta'
df['dx'] = df['world_x'] - X_c
df['dy'] = df['world_y'] - Y_c
df['r'] = np.sqrt(df['dx']**2 + df['dy']**2)
df['theta'] = np.arctan2(df['dy'], df['dx'])

# Compute shortest angular change and angular velocity
theta_shift = group['theta'].shift(1)
group['delta_theta'] = np.arctan2(np.sin(group['theta'] - theta_shift), np.cos(group['theta'] - theta_shift))
group['omega'] = group['delta_theta'] * FPS

# Apply Circulating Ring and Speed Constraints
group['is_in_ring'] = (group['r'] >= R_MIN) & (group['r'] <= R_MAX)
group['is_wrong_way'] = (group['omega'] < OMEGA_THRESHOLD) & group['is_in_ring']

# Check for consecutive frames
group['consecutive_group'] = (group['is_wrong_way'] != group['is_wrong_way'].shift()).cumsum()
wrong_way_frames = group[group['is_wrong_way']]
if not wrong_way_frames.empty:
    counts = wrong_way_frames.groupby('consecutive_group').size()
    valid_groups = counts[counts >= CONSECUTIVE_FRAMES_THRESHOLD].index
```

---

## 2. Lane Straddling Violation
Vehicles must maintain their lane and not float/straddle the boundary line dividing the lanes.

### Mathematical Formulation
* Lane Boundaries: The inner lane is [6.0, 10.0) meters, and the outer lane is [10.0, 14.0] meters. The dividing lane boundary is at r = 10.0 meters.
* Straddling Condition:
  abs(r(t) - 10.0) <= 0.5 meters
* Temporal Threshold: The condition must persist continuously for at least 30 frames (1.0 second at 30 FPS).

### Code Implementation
```python
# Check boundary proximity
ring_df['is_straddling'] = (np.abs(ring_df['r'] - 10.0) <= 0.5)

# Calculate consecutive frames
is_strad = group['is_straddling'].values
cumsum = np.cumsum(~is_strad)
max_consecutive = 0
if is_strad.any():
    counts = pd.Series(cumsum[is_strad]).value_counts()
    if not counts.empty:
        max_consecutive = counts.max()

if max_consecutive >= 30:
    # Lane Straddling violation detected...
```

---

## 3. Tailgating / Proximity Violation
This rule monitors the spacing between vehicles in the same roundabout lane.

### Mathematical Formulation
Let there be two vehicles i (follower) and j (leader) in the same lane (Inner or Outer) in the same frame, sorted by polar angle such that theta_j >= theta_i.
1. Angular Headway (delta_theta):
   delta_theta = (theta_j - theta_i) modulo 2*pi

2. Arc-length Gap Distance (d):
   Using the average radius of the two vehicles:
   d = ((r_j + r_i) / 2) * delta_theta

3. Violation Condition:
   d < 4.0 meters AND velocity_follower > 1.0 m/s

### Code Implementation
```python
# Group frame-by-frame, then by lane
for (frame, lane), group in ring_df.groupby(['frame', 'lane']):
    if len(group) < 2:
        continue
    
    sorted_group = group.sort_values('theta').to_dict('records')
    n = len(sorted_group)
    
    for i in range(n):
        follower = sorted_group[i]
        leader = sorted_group[(i + 1) % n]
        
        # Angular change wrapped to [0, 2*pi]
        delta_theta = (leader['theta'] - follower['theta']) % (2 * np.pi)
        d = ((leader['r'] + follower['r']) / 2) * delta_theta
        
        if d < 4.0 and follower['velocity_ms'] > 1.0:
            # Tailgating violation detected...
```

---

## 4. Unsafe Overtaking Violation
Flags vehicles that pass another vehicle in the same lane with a very narrow lateral separation.

### Mathematical Formulation
For two vehicles a and b inside the same lane in frame t:
1. Relative Angular Headway (theta_diff):
   theta_diff(t) = atan2(sin(theta_b(t) - theta_a(t)), cos(theta_b(t) - theta_a(t)))

2. Crossover Condition: A sign change/flip in theta_diff(t) compared to the previous frame t-1:
   theta_diff(t-1) * theta_diff(t) < 0

3. Boundary Wrap-around Filtering:
   To ignore boundary wrap-around transitions occurring at +/- pi, the angular shift must be small:
   angular_change = abs(atan2(sin(theta_diff(t) - theta_diff(t-1)), cos(theta_diff(t) - theta_diff(t-1)))) < 1.0 radian

4. Safety Verification:
   The crossover is unsafe if the vehicles are close in Euclidean distance and the overtaker is moving:
   d_Euclidean = sqrt((x_a(t) - x_b(t))^2 + (y_a(t) - y_b(t))^2) < 10.0 meters AND velocity_overtaker > 1.0 m/s

### Code Implementation
```python
diff = np.arctan2(np.sin(theta_b - theta_a), np.cos(theta_b - theta_a))

if pair_key in pair_states:
    prev_frame, prev_lane, prev_diff = pair_states[pair_key]
    
    if prev_frame == frame - 1 and prev_lane == lane:
        # Check if they crossed each other (sign flip)
        if prev_diff * diff < 0:
            ang_change = np.abs(np.arctan2(np.sin(diff - prev_diff), np.cos(diff - prev_diff)))
            dist = np.hypot(rec_a['world_x'] - rec_b['world_x'], rec_a['world_y'] - rec_b['world_y'])
            
            if ang_change < 1.0 and dist < 10.0:
                overtaker = rec_a if prev_diff > 0 else rec_b
                if overtaker['velocity_ms'] > 1.0:
                    # Unsafe Overtaking violation detected...
```

---

## 5. Sudden Braking Violation
Identifies dangerously high decelerations.

### Mathematical Formulation
1. Smoothing Window:
   Smooth the velocity curve using a 7-frame rolling average to remove tracking jitter:
   v_smooth(t) = (1/7) * sum_{k=0..6}( v(t-k) )

2. Instantaneous Acceleration (a):
   a(t) = (v_smooth(t) - v_smooth(t-1)) / dt = (v_smooth(t) - v_smooth(t-1)) * FPS

3. Braking Conditions:
   a(t) < -6.0 m/s^2 AND v_smooth(t-1) > 3.0 m/s

### Code Implementation
```python
# 7-frame rolling velocity to smooth tracking noise
smooth_vel = group['velocity_ms'].rolling(window=7, min_periods=1).mean()
accel = smooth_vel.diff() * fps
prev_smooth_vel = smooth_vel.shift(1)

# Flag deceleration rates below -6.0 m/s^2 when starting speed is above 3.0 m/s
braking_mask = (accel < -6.0) & (prev_smooth_vel > 3.0)
```

---

## 6. Vehicle Stoppage Violation
Identifies vehicles that stop inside the circulating lane, forming hazards.

### Mathematical Formulation
For a window size W = 90 frames (3.0 seconds at 30 FPS):
1. Spatial Displacement (disp_90):
   disp_90(t) = sqrt((x(t) - x(t-90))^2 + (y(t) - y(t-90))^2)

2. Windowed Mean Velocity (mean_v_90):
   mean_v_90(t) = (1/90) * sum_{k=0..89}( v(t-k) )

3. Stoppage Criteria:
   disp_90(t) < 1.0 meter AND mean_v_90(t) < 0.8 m/s

### Code Implementation
```python
# 90-frame window (3 seconds at 30 FPS)
prev_x = group['world_x'].shift(90)
prev_y = group['world_y'].shift(90)
disp = np.hypot(group['world_x'] - prev_x, group['world_y'] - prev_y)
window_mean_vel = group['velocity_ms'].rolling(90).mean()

stoppage_mask = (disp < 1.0) & (window_mean_vel < 0.8)
```

---
---

# II. Conflict Detection & Aggressor Classification Rules

These rules detect intersecting trajectory pathways and classify which vehicle is the active initiator of a conflict event.

## 1. Conflict/Near-Collision Event Detection
Identifies conflict events using trajectory projections.

### Mathematical Formulation
1. Displacement Filter:
   To filter out stationary objects, a vehicle must show a minimum displacement over lookahead W = 125 frames:
   disp_i = sqrt((x_i(t+W) - x_i(t))^2 + (y_i(t+W) - y_i(t))^2) >= 3.0 meters

2. Effective Stable Velocity:
   v_eff_i = disp_i / elapsed_time

3. Heading Ray Intersection:
   Represent trajectories as rays R_1(t1) = P_1 + t1 * D_1 and R_2(t2) = P_2 + t2 * D_2.
   Let P_conflict be their intersection point.
   The distances to the conflict point are:
   d1 = distance(P_conflict, P1)
   d2 = distance(P_conflict, P2)

4. Time-to-Collision (TTC):
   TTC_1 = d1 / v_eff_1
   TTC_2 = d2 / v_eff_2

5. Conflict Conditions:
   A conflict is logged if both vehicles are heading toward the intersection and:
   TTC_1 <= 7.0 seconds AND TTC_2 <= 7.0 seconds AND abs(TTC_1 - TTC_2) <= 1.0 second

### Code Implementation
```python
# Determine intersection point of rays
cross = d1x * d2y - d1y * d2x
if abs(cross) < 1e-10:
    return None

t = ((pos2[0] - pos1[0]) * d2y - (pos2[1] - pos1[1]) * d2x) / cross
u = ((pos2[0] - pos1[0]) * d1y - (pos2[1] - pos1[1]) * d1x) / cross

if t < 0 or u < 0:
    return None  # Intersection is behind the vehicles

px = pos1[0] + t * d1x
py = pos1[1] + t * d1y

d1 = math.hypot(px - pos1[0], py - pos1[1])
d2 = math.hypot(px - pos2[0], py - pos2[1])

ttc1 = d1 / s1 if s1 > STOPPED_SPEED_MS else float("inf")
ttc2 = d2 / s2 if s2 > STOPPED_SPEED_MS else float("inf")

# Flag only if they arrive in the intersection zone at the same time
if ttc1 > max_ttc or ttc2 > max_ttc or abs(ttc1 - ttc2) > max_ttc_diff:
    return None
return px, py
```

---

## 2. Conflict Aggressor Classification

### A. Angular Deviation Method (delta) (Primary)
Analyzes angular deviations to determine which vehicle crossed first.

#### Mathematical Formulation
Let P_1(t) be the position of V_1, P_1(t+Delta) be its future position, and P_2(t) be the position of V_2.
1. Heading Angle of V_1 (theta_heading):
   theta_heading = atan2(y1(t+Delta) - y1(t), x1(t+Delta) - x1(t))

2. Bearing Angle from V_1 to V_2 (theta_bearing):
   theta_bearing = atan2(y2(t) - y1(t), x2(t) - x1(t))

3. Angular Deviation (delta):
   delta(t) = theta_heading - theta_bearing

4. Classification Logic:
   * V1 crosses first (aggressor): if cos(delta(t)) crosses from positive to negative:
     cos(delta(t-1)) > 0 AND cos(delta(t)) < 0
   * V2 crosses first (aggressor): if sin(delta(t)) undergoes any sign flip:
     sin(delta(t-1)) * sin(delta(t)) < 0

#### Code Implementation
```python
def _compute_delta(v1_pos, v1_future, v2_pos):
    heading = math.atan2(v1_future[1] - v1_pos[1], v1_future[0] - v1_pos[0])
    bearing = math.atan2(v2_pos[1] - v1_pos[1], v2_pos[0] - v1_pos[0])
    return heading - bearing

# For each frame t in a window around the conflict:
d = _compute_delta(v1_pos, v1_future_pos, v2_pos)
cos_vals.append(math.cos(d))
sin_vals.append(math.sin(d))

# Evaluate crossings
cos_pos_to_neg = any(cos_vals[k] > 0 and cos_vals[k + 1] < 0 for k in range(len(cos_vals) - 1))
sin_sign_flip = any(sin_vals[k] * sin_vals[k + 1] < 0 for k in range(len(sin_vals) - 1))

if cos_pos_to_neg and not sin_sign_flip:
    return "v1" # V1 crossed first (aggressor)
if sin_sign_flip and not cos_pos_to_neg:
    return "v2" # V2 crossed first (aggressor)
```

### B. Time-to-Collision Comparison Method (Secondary/Fallback)
When angular evaluations are uncertain, the vehicle closer in time to the conflict point is the aggressor.

#### Mathematical Formulation
TTC_i = distance(P_conflict, P_i(t)) / v_i(t)
Aggressor is V1 if TTC_1 < TTC_2, otherwise V2.

#### Code Implementation
```python
ttc1 = d1 / event.v1_speed if event.v1_speed > STOPPED_SPEED_MS else float("inf")
ttc2 = d2 / event.v2_speed if event.v2_speed > STOPPED_SPEED_MS else float("inf")

if ttc1 < ttc2:
    return "v1", ttc1, ttc2
elif ttc2 < ttc1:
    return "v2", ttc1, ttc2
```

---
---

# III. Alternative/Heuristic Event Detectors

These additional detectors identify stop-and-go behavior and sharp driving maneuvers.

## 1. Stop-and-Wait Event Detection
Flags vehicles that stop due to a leading vehicle obstructing them.

### Mathematical Formulation
1. Current Stopped Speed:
   v_current = (displacement over last 10 frames) / dt_10 <= 1.0 m/s

2. Prior Moving Speed:
   v_prior = (displacement 50 frames ago) / dt_prior >= 1.5 m/s

3. Leading Obstruction Search:
   Another vehicle j must exist in the heading path cone of stopped vehicle i:
   d = distance(P_j, P_i) <= 25.0 meters
   theta_cone = abs(theta_heading_i - theta_bearing_i_to_j) <= 60.0 degrees

### Code Implementation
```python
# Confirm current speed is slow and prior speed was fast
cur_speed = _disp_speed(tf, recent[0], recent[-1], fps)
if cur_speed <= SW_STOP_SPEED_MS:
    prior_speed = _disp_speed(tf, prior_w[0], prior_w[-1], fps)
    if prior_speed >= SW_MOVING_SPEED_MS:
        # Check if there is an obstructing vehicle ahead
        for cid in frame_to_ids[frame]:
            cpos = (crec["world_x"], crec["world_y"])
            dist = math.hypot(cpos[0] - pos[0], cpos[1] - pos[1])
            if dist <= SW_CAUSE_PROX_M:
                bearing = math.atan2(cpos[1] - pos[1], cpos[0] - pos[0])
                diff = abs(math.degrees(heading - bearing)) % 360
                if (diff if diff <= 180 else 360 - diff) <= SW_CAUSE_CONE_DEG:
                    # Obstruction found; flag Stop-and-Wait event
```

---

## 2. Sharp Turn Event Detection
Flags rapid heading changes.

### Mathematical Formulation
Over a 10-frame window (0.4 seconds):
1. Displacement:
   disp_10 >= 1.0 meter AND v_10 >= 1.0 m/s

2. Angular velocity (omega_turn):
   theta_start = theta_heading(t-10), theta_end = theta_heading(t)
   omega_turn = (abs(theta_start - theta_end) modulo 360) / dt_10 >= 40.0 deg/s

### Code Implementation
```python
# Compute displacement and verify movement
disp = math.hypot(dx, dy)
if disp >= ST_MIN_DISP_M and (disp / elapsed) >= ST_MIN_SPEED_MS:
    h_start = math.atan2(...)
    h_end = math.atan2(...)
    
    # Calculate turn speed
    curvature = _angle_diff_deg(h_start, h_end)
    ang_vel = curvature / elapsed
    
    if ang_vel >= ST_MIN_ANG_VEL_DEG_S:
        # Flag Sharp Turn event...
```

---

## 3. Heuristic Overtaking (Q3 -> Q4 Transition)
Alternative overtaking detection based on quadrant tracking in the angular deviation space.

### Mathematical Formulation
Let follower V_1 have a heading ray R_1 and leader V_2 have a heading ray R_2.
Compute the relative bearing angle delta:
delta = theta_heading_2 - theta_bearing_2_to_1
* Quadrant 3 (Q3) - Behind Phase: cos(delta) < 0 AND sin(delta) < 0 (V_1 is behind-right of V_2).
* Quadrant 4 (Q4) - Ahead Phase: cos(delta) > 0 AND sin(delta) < 0 (V_1 is ahead-right of V_2).

An overtake is confirmed when a vehicle transitions from Q3 to Q4 within the tracking window:
Older frames in window are in Q3 AND recent frames in window are in Q4.

### Code Implementation
```python
# Compute angle deviation delta
d = _delta_v2_to_v1(v2_pos, v2_future, v1_pos)
cos_d, sin_d = math.cos(d), math.sin(d)

# Maintain history window
history.append((frame, cos_d, sin_d))

# Evaluate quadrant states
older_q3 = sum(1 for _, c, s in older if c < 0 and s < 0)  # Q3
recent_q4 = sum(1 for _, c, s in recent if c > 0 and s < 0) # Q4

if older_q3 >= 4 and recent_q4 >= 4:
    # Overtake event detected...
```
