# Traffic Safety Rules Documentation

This directory contains comprehensive, detailed documentation for all traffic safety violation detection rules implemented in the project. Each rule includes physical context, mathematical formulation, parameter settings, and detailed explanations of the key Python code snippets.

## Implemented Safety Rules

### 1. [Wrong-Way Driving Detection Rule](wrong_way_rule.md)
* **Script**: [`src/safety/wrong_way_rule.py`](file:///c:/Users/k16na/Desktop/btp/src/safety/wrong_way_rule.py)
* **Goal**: Detect vehicles traveling in the wrong direction (counter-clockwise) within the circulatory ring of the roundabout.
* **Key Math**: Polar coordinate transformation, shortest-angular path displacement, and temporal noise filtering.

### 2. [Safe Space Rule (Tailgating)](safe_space_rule.md)
* **Script**: [`src/safety/safe_space_rule.py`](file:///c:/Users/k16na/Desktop/btp/src/safety/safe_space_rule.py)
* **Goal**: Detect vehicles following too closely behind leading vehicles inside the same lane (Proximity Violation).
* **Key Math**: Radial classification, relative polar-angle ordering, and arc-length calculation ($d = R_{\text{avg}} \times \Delta\theta$).

### 3. [Unsafe Overtaking Detection Rule](unsafe_overtaking_rule.md)
* **Script**: [`src/safety/overtaking.py`](file:///c:/Users/k16na/Desktop/btp/src/safety/overtaking.py)
* **Goal**: Identify vehicles executing high-risk overtaking maneuvers within the restricted roundabout zone.
* **Key Math**: Relative polar angle difference ($\Delta \theta$), sign-flip condition ($\text{prev\_diff} \times \text{diff} < 0$), spatial proximity filtering ($d \le 4.5\text{m}$, $\Delta r \le 2.2\text{m}$), and overtaker speed floor ($v \ge 0.8\text{ m/s}$).

### 4. [Vehicle Stoppage & Obstruction Rule](stoppage_rule.md)
* **Script**: [`src/safety/stoppage.py`](file:///c:/Users/k16na/Desktop/btp/src/safety/stoppage.py)
* **Goal**: Detect vehicles coming to a stop or moving at near-zero speed inside the circulating ring for an extended duration.
* **Key Math**: Radial lane classification (`Inner`/`Outer`), overlapping bounding box deduplication ($\text{dist} < 1.8\text{m}$, $\text{IoU} > 0.20$), 90-frame (3-second) spatial displacement ($\Delta d_{90} < 1.0\text{m}$), and 90-frame rolling mean speed ($\bar{v}_{90} < 0.8\text{ m/s}$).

### 5. [Unsafe Roundabout Shortcut Rule](unsafe_roundabout_shortcut_rule.md)
* **Goal**: Detect vehicles cutting across corners at the intersection instead of traveling properly around the central island.
* **Key Math**: Compass-based entry/exit verification, phase-unwrapped angular displacement ($\Delta\theta_{\text{unwrapped}}$), and intersection-level congestion mapping.

### 6. [Erratic Lane Weaving Rule](erratic_weaving_rule.md)
* **Script**: [`src/safety/jittering_rule.py`](file:///c:/Users/k16na/Desktop/btp/src/safety/jittering_rule.py)
* **Goal**: Detect erratic lane weaving by tracking physical lane boundary crosses inside the roundabout.
* **Key Math**: Temporal tracking of radial distance $r$, mapping discrete lane states (Inner vs Outer), and accumulating state transitions over a sliding time window.

---

## Constants Reference
Below is the common spatial frame of reference used across all safety rules:

* **Roundabout Center ($X_C, Y_C$)**: $(43.5, 28.5)$
* **Inner Island Radius ($R_{\text{INNER}}$)**: $6.0$ meters
* **Outer Circulatory Boundary ($R_{\text{OUTER}}$)**: $14.0$ meters
* **Video FPS**: $30.0$ frames per second (time step $\Delta t = 1/30$ seconds)
