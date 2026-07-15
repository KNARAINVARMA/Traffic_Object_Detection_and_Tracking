from __future__ import annotations
from typing import List, Tuple

# Approximate intersection zone corners in pixel coordinates (1920x1080 frame).
# Adjust after visual verification by running main_safety.py --preview-zone.
INTERSECTION_ZONE_PX: List[Tuple[float, float]] = [
    (0,    0),     # top-left
    (1920, 0),     # top-right
    (1920, 1080),  # bottom-right
    (0,    1080),  # bottom-left
]

from .calibration import SCALE as SCALE_MPX


def point_in_polygon(px: float, py: float, polygon: List[Tuple[float, float]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def in_intersection_zone(center_x: float, center_y: float) -> bool:
    return point_in_polygon(center_x, center_y, INTERSECTION_ZONE_PX)


def world_to_px(world_x: float, world_y: float) -> Tuple[float, float]:
    return world_x / SCALE_MPX, world_y / SCALE_MPX
