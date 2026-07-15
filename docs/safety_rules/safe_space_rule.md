# Safe Space & Lane Straddling Rules

This document details the algorithm and implementation of the **Lane Straddling** and **Tailgating (Proximity Violation)** safety rules.

---

## 1. Lane Straddling Detection Rule

### A. Context & Description
Roundabouts are split into lanes to organize traffic flow. Straddling the boundary line between lanes reduces the effective capacity of the roundabout, blocks other vehicles, and increases collision risks.

The rule states:
> A vehicle is flagged for a lane straddling violation if its center point remains within a $\pm 0.5$ meter band of the lane boundary for a continuous duration of $\ge 30$ frames (1.0 second).

### B. Mathematical Formulation
Let the lane boundary between the Inner and Outer lanes of the roundabout be located at a radial distance of:
$$R_{\text{boundary}} = 10.0\text{ meters}$$

For a vehicle radial distance $r_i$ at frame $i$, the straddling condition is:
$$|r_i - R_{\text{boundary}}| \le 0.5\text{ meters}$$
which evaluates to:
$$9.5 \le r_i \le 10.5\text{ meters}$$

### C. Code Snippets & Explanation
The logic is implemented in [`safe_space_rule.py`](file:///d:/btp/Traffic_Object_Detection_and_Tracking/src/safety/safe_space_rule.py).

```python
# Step 1: Assign Lane based on radial coordinate
def assign_lane(r):
    if 6.0 <= r < 10.0:
        return 'Inner'
    elif 10.0 <= r <= 14.0:
        return 'Outer'
    else:
        return 'None'

df['lane'] = df['r'].apply(assign_lane)

# Step 2: Flag Straddling Condition (close to r = 10.0)
ring_df['is_straddling'] = (np.abs(ring_df['r'] - 10.0) <= 0.5)
```
* **Explanation**: The roundabout is partitioned into `Inner` ($6.0\text{m} \le r < 10.0\text{m}$) and `Outer` ($10.0\text{m} \le r \le 14.0\text{m}$) lanes. The lane straddling condition marks any frame where the vehicle is within $0.5$ meters of the $10.0$ meter lane separator.

```python
# Step 3: Find consecutive frames of straddling
for track_id, group in ring_df.groupby('track_id'):
    group = group.sort_values('frame')
    is_strad = group['is_straddling'].values
    
    # Calculate sizes of consecutive True sequences
    cumsum = np.cumsum(~is_strad)
    max_consecutive = 0
    if is_strad.any():
        counts = pd.Series(cumsum[is_strad]).value_counts()
        if not counts.empty:
            max_consecutive = counts.max()
            
    if max_consecutive >= 30:
        straddling_violations.append({'track_id': track_id, 'class_name': class_name})
```
* **Explanation**: 
  1. `~is_strad` inverts the boolean array (True becomes False, False becomes True).
  2. `np.cumsum(~is_strad)` creates an index that increments only when the vehicle is *not* straddling. This means that during a continuous sequence of straddling frames, the cumulative sum remains constant.
  3. Grouping by this sum and counting the size of the groups identifies the lengths of all continuous straddling blocks.
  4. If the longest block is $\ge 30$ frames (1.0 second at 30 FPS), it is flagged.

---

## 2. Tailgating (Proximity Violation) Rule

### A. Context & Description
Tailgating occurs when a vehicle follows the vehicle ahead of it too closely, leaving an unsafe gap. If the lead vehicle brakes suddenly, a rear-end collision is highly likely.

The rule states:
> A vehicle is flagged for tailgating if its arc distance to the leading vehicle in the same lane is $< 4.0$ meters while traveling at a velocity $> 1.0\text{ m/s}$.

### B. Mathematical Formulation
To calculate the distance between two vehicles in a circular lane at a specific frame:
1. We group all vehicles in the same lane (`Inner` or `Outer`) at the same frame.
2. We sort them by their polar angle $\theta_i \in (-\pi, \pi]$.
3. For a follower vehicle $A$ and its adjacent leader vehicle $B$ (where $B$ is ahead of $A$ in the angular order):
   * **Angular separation ($\Delta\theta$)** (modulo $2\pi$ to account for wrapping):
     $$\Delta\theta = (\theta_B - \theta_A) \pmod{2\pi}$$
   * **Mean circular radius ($R_{\text{avg}}$)**:
     $$R_{\text{avg}} = \frac{r_A + r_B}{2}$$
   * **Circular Arc Distance ($d$)**:
     $$d = R_{\text{avg}} \times \Delta\theta$$
4. A violation is flagged if:
   $$d < 4.0\text{ meters} \quad \text{and} \quad v_A > 1.0\text{ m/s}$$

---

### C. Code Snippets & Explanation
The tailgating detection loop processes each frame and lane dynamically:

```python
# Group frame-by-frame, then by lane
for (frame, lane), group in ring_df.groupby(['frame', 'lane']):
    if len(group) < 2:
        continue
    
    # Sort vehicles by polar angle theta to find follower-leader order
    sorted_group = group.sort_values('theta').to_dict('records')
    n = len(sorted_group)
    
    for i in range(n):
        follower = sorted_group[i]
        leader = sorted_group[(i + 1) % n]  # Wraps around to form a closed ring
        
        # Calculate circular angle difference modulo 2*pi
        delta_theta = (leader['theta'] - follower['theta']) % (2 * np.pi)
        
        # Arc length formula: d = radius * delta_theta
        d = ((leader['r'] + follower['r']) / 2) * delta_theta
        
        # Apply threshold criteria
        if d < 4.0 and follower['velocity_ms'] > 1.0:
            tailgating_records.append({
                'frame': frame,
                'follower_track_id': follower['track_id'],
                'leader_track_id': leader['track_id'],
                'lane': lane,
                'd': d,
                'class_name': follower['class_name']
            })
```
* **Explanation**:
  1. We sort vehicles along the circular path by sorting their polar angle `theta`.
  2. The next index `(i + 1) % n` retrieves the leader. The modulo `% n` ensures that the vehicle at the end of the array (e.g. angle $+\pi$) is correctly paired with the vehicle at the beginning of the array (e.g. angle $-\pi$), closing the loop.
  3. `delta_theta = (leader['theta'] - follower['theta']) % (2 * np.pi)` handles the boundary wrap properly.
  4. The arc distance is the product of the average radius of the two vehicles and their angular distance.
  5. The velocity threshold (`follower['velocity_ms'] > 1.0`) prevents flagging parked or stationary vehicles queueing in congestion.
