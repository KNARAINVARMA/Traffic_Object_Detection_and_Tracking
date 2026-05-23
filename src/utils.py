"""
Utility Module — Logging, Visualisation, Shared Constants

Centralising these here prevents duplicate definitions and makes it easy to
change the colour palette or logging format in one place.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Class colour palette (BGR, for OpenCV)
# ---------------------------------------------------------------------------

CLASS_COLORS: dict = {
    "person":     (0,   255,  0),    # green
    "car":        (255,  0,   0),    # blue
    "motorcycle": (0,   165, 255),   # orange
    "bus":        (255,  0,  255),   # magenta
    "truck":      (0,   255, 255),   # yellow
}

# Fallback colour for unknown classes
_DEFAULT_COLOR: Tuple[int, int, int] = (200, 200, 200)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(
    level:    int          = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root logger with a console handler and an optional file handler.

    Args:
        level:    Minimum log level (logging.DEBUG / INFO / WARNING / ERROR).
        log_file: If provided, also write logs to this file path.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# OpenCV drawing helpers
# ---------------------------------------------------------------------------

def draw_box_label(
    image:  "import numpy; numpy.ndarray",
    bbox:   List[int],
    label:  str,
    color:  Tuple[int, int, int] = _DEFAULT_COLOR,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> None:
    """
    Draw a bounding box with a filled label badge onto *image* (in-place).

    Args:
        image:      BGR numpy array (modified in-place).
        bbox:       [x1, y1, x2, y2] in integer pixel coordinates.
        label:      Text to display above the box.
        color:      BGR colour for the box and badge background.
        thickness:  Box line thickness in pixels.
        font_scale: OpenCV font scale factor.
    """
    import cv2  # local import keeps the module usable without OpenCV in tests

    x1, y1, x2, y2 = bbox

    # Bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Label badge
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    badge_y1 = max(0, y1 - th - baseline - 4)
    badge_y2 = y1
    badge_x2 = min(image.shape[1], x1 + tw + 4)

    cv2.rectangle(image, (x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)

    # Choose text colour that contrasts with the badge background
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

    cv2.putText(
        image, label,
        (x1 + 2, max(th, y1 - baseline - 2)),
        font, font_scale, text_color, 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Progress / statistics helpers
# ---------------------------------------------------------------------------

def format_stats(stats: dict) -> str:
    """
    Pretty-print a statistics dictionary as a multi-line string.

    Args:
        stats: Arbitrary key-value dict of processing statistics.

    Returns:
        Human-readable string suitable for logging / printing.
    """
    lines = ["=" * 56, "  PROCESSING SUMMARY", "=" * 56]
    for k, v in stats.items():
        if isinstance(v, float):
            lines.append(f"  {k:<30} {v:.4f}")
        elif isinstance(v, dict):
            lines.append(f"  {k}:")
            for sub_k, sub_v in v.items():
                lines.append(f"      {sub_k:<26} {sub_v}")
        else:
            lines.append(f"  {k:<30} {v}")
    lines.append("=" * 56)
    return "\n".join(lines)


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
