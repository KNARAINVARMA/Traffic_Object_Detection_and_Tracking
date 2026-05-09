"""
Homography / Coordinate-Mapping Module — Pixel → Real-World Metres

WHY PIXEL DISTANCES ARE MISLEADING
====================================
Pixel coordinates by themselves say nothing about physical reality.  Two objects
whose pixel centres are 50 px apart could be 2 m or 20 m apart in the real world
depending on the camera height, focal length, and tilt angle.

Safety rules are defined in physical units:
  "Two vehicles are dangerously close if they are < 3 m apart"
  "A vehicle is speeding if v > 15 m/s"

Without a pixel-to-metre mapping, neither rule can be evaluated.

METHODS SUPPORTED
==================
1. Simple scale factor (default — recommended for stationary top-down cameras)
   Assumes the ground plane is approximately parallel to the image plane
   (pure top-down view, no perspective distortion).  One scale factor
   (metres / pixel) is computed from a single known-length reference object
   (e.g. a car whose real length is known) and applied uniformly.

   scale [m/px] = real_length_m / pixel_length_px

   Limitations:
   • Assumes uniform scale across the entire image (valid when camera is
     truly nadir / straight down and scene is flat).
   • Even a 15° tilt introduces ≈4% error at image centre, growing toward
     the near edge.

2. Full perspective homography (optional — for tilted cameras)
   If the camera has any tilt, the scale factor changes across the image.
   The user can supply 4+ ground-control points (GCPs) — pixel (u, v) pairs
   whose real-world (X, Y) coordinates in metres are known — and the module
   fits a 3×3 homography matrix using OpenCV's `findHomography`.

   The mapped coordinates are then in the same metric coordinate system as the
   GCPs (e.g. a local ENU frame centred at the intersection).

   Limitations:
   • Requires manual annotation of reference points.
   • Assumes the ground is flat (planar homography).

VELOCITY CALCULATION
=====================
Given world coordinates at frames t and t−1:
    vx [m/s] = (wx_t − wx_{t−1}) × fps
    vy [m/s] = (wy_t − wy_{t−1}) × fps
    speed    = √(vx² + vy²)

Smoothing (done in smoothing.py before this module) is essential to suppress
phantom velocity spikes from detection jitter.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CoordinateMapper
# ---------------------------------------------------------------------------

class CoordinateMapper:
    """
    Converts pixel coordinates to approximate real-world metric coordinates.

    Supports two modes:
      • *Scale* mode: single scale factor (m / px), uniform across the image.
      • *Homography* mode: full 3×3 perspective transform derived from GCPs.

    Use :py:meth:`from_scale_factor` or :py:meth:`from_reference_object` for
    scale mode, and :py:meth:`from_ground_control_points` for homography mode.
    """

    # ---- constructors -------------------------------------------------------

    def __init__(self, scale_factor: float) -> None:
        """
        Initialise with a pre-computed uniform scale factor.

        Args:
            scale_factor: Metres per pixel (m / px).
        """
        self._scale      = scale_factor
        self._homography: Optional[np.ndarray] = None
        logger.info("CoordinateMapper initialised — scale=%.6f m/px", scale_factor)

    @classmethod
    def from_scale_factor(cls, scale_factor: float) -> "CoordinateMapper":
        """Construct directly from a known scale (m / px)."""
        return cls(scale_factor)

    @classmethod
    def from_reference_object(
        cls,
        real_length_m:   float,
        pixel_length_px: float,
    ) -> "CoordinateMapper":
        """
        Compute scale from a single reference object of known size.

        Example: a typical Indian sedan is ≈4.0 m long.  Measure its pixel
        length along the vehicle's axis in a representative frame, and pass
        both values here.

        Args:
            real_length_m:   Actual length of the reference object in metres.
            pixel_length_px: Length of the same object measured in pixels.
        """
        if pixel_length_px <= 0:
            raise ValueError("pixel_length_px must be positive.")
        scale = real_length_m / pixel_length_px
        logger.info(
            "Scale computed from reference object: %.2f m / %.1f px = %.6f m/px",
            real_length_m, pixel_length_px, scale,
        )
        return cls(scale)

    @classmethod
    def from_ground_control_points(
        cls,
        pixel_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]],
        fallback_scale: float = 0.05,
    ) -> "CoordinateMapper":
        """
        Fit a full perspective homography from ≥4 ground-control points.

        This accounts for camera tilt and provides spatially-varying scale.
        Coordinates in world_points define the origin and units of the output
        (e.g., metres relative to the intersection centre).

        Args:
            pixel_points:   List of (u, v) pixel positions.
            world_points:   Corresponding (X, Y) real-world positions in metres.
            fallback_scale: Used only as the uniform scale fallback if GCP count
                            is insufficient.

        Returns:
            CoordinateMapper with homography mode active.

        Raises:
            ValueError: If fewer than 4 GCPs are provided.
        """
        if len(pixel_points) < 4 or len(world_points) < 4:
            raise ValueError(
                f"At least 4 GCPs required, got {len(pixel_points)}."
            )

        pts_px  = np.array(pixel_points, dtype=np.float32)
        pts_wld = np.array(world_points, dtype=np.float32)

        H, mask = cv2.findHomography(pts_px, pts_wld, cv2.RANSAC, 5.0)
        if H is None:
            logger.warning("findHomography failed; falling back to scale mode.")
            return cls(fallback_scale)

        inliers = int(mask.sum()) if mask is not None else len(pixel_points)
        logger.info(
            "Homography fitted from %d GCPs (%d inliers).",
            len(pixel_points), inliers,
        )

        mapper = cls(fallback_scale)
        mapper._homography = H
        return mapper

    # ---- conversion ---------------------------------------------------------

    def to_world(
        self,
        cx: float,
        cy: float,
    ) -> Tuple[float, float]:
        """
        Map a pixel centre coordinate to world (metric) coordinates.

        Args:
            cx, cy: Pixel column and row of the object centre.

        Returns:
            (world_x, world_y) in metres.
        """
        if self._homography is not None:
            return self._apply_homography(cx, cy)
        return cx * self._scale, cy * self._scale

    def _apply_homography(self, cx: float, cy: float) -> Tuple[float, float]:
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pt, self._homography)
        wx = float(dst[0, 0, 0])
        wy = float(dst[0, 0, 1])
        return wx, wy

    # ---- properties ---------------------------------------------------------

    @property
    def scale_factor(self) -> float:
        """Uniform scale factor in m/px (or approximate if homography is active)."""
        return self._scale

    @property
    def mode(self) -> str:
        return "homography" if self._homography is not None else "scale"

    def __repr__(self) -> str:
        return f"CoordinateMapper(mode={self.mode}, scale={self._scale:.6f} m/px)"
