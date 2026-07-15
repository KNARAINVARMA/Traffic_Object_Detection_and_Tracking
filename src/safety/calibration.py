# Centralized Geometric Calibration Configuration

# 1. Calibration References
LANE_WIDTH_M = 7.0
LANE_WIDTH_PX = 85.0

# 2. Scale Factor (meters per pixel)
SCALE = LANE_WIDTH_M / LANE_WIDTH_PX  # ~0.082352941 m/px

# 3. Roundabout Geometry in Pixel Coordinates (Original measurements)
CENTER_X_PX = 870.0
CENTER_Y_PX = 570.0

INNER_RADIUS_PX = 120.0
OUTER_RADIUS_PX = 280.0
LANE_SEPARATOR_PX = 200.0

# 4. Computed Roundabout Geometry in Metric coordinates (meters)
CENTER_X = CENTER_X_PX * SCALE
CENTER_Y = CENTER_Y_PX * SCALE

R_INNER = INNER_RADIUS_PX * SCALE
R_OUTER = OUTER_RADIUS_PX * SCALE
R_SEPARATOR = LANE_SEPARATOR_PX * SCALE
