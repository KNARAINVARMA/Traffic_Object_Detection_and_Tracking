"""
postprocess_vehicle_classes.py — Vehicle Class Postprocessing Module

This module implements postprocessing heuristics to reduce false-positive
truck detections and improve separation between cars and trucks on drone footage.
It applies higher confidence thresholds for trucks, size-based filters, class-frequency
priors (relabeling ambiguous trucks as cars), and saves cropped debug images.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Constants
TRUCK_CLASS_ID = 7
CAR_CLASS_ID = 2
BUS_CLASS_ID = 5

def postprocess_vehicle_detections(
    detections: List[Dict],
    frame: np.ndarray,
    frame_idx: int,
    truck_conf_thresh: float = 0.55,
    truck_min_area: float = 8000.0,
    truck_min_width: float = 120.0,
    truck_min_height: float = 50.0,
    bus_conf_thresh: float = 0.30,
    car_conf_thresh: float = 0.25,
    person_conf_thresh: float = 0.25,
    motorcycle_conf_thresh: float = 0.15,
    debug_trucks: bool = False,
    stats: Optional[Dict] = None,
) -> List[Dict]:
    """
    Postprocess a frame of detections to filter and relabel vehicle classes.

    Args:
        detections: List of detections from the detector.
        frame: BGR image frame (H x W x C).
        frame_idx: Current frame index in the video.
        truck_conf_thresh: Confidence threshold for keeping a truck detection.
        truck_min_area: Minimum bbox area for keeping a truck detection.
        truck_min_width: Minimum bbox width for keeping a truck detection.
        truck_min_height: Minimum bbox height for keeping a truck detection.
        bus_conf_thresh: Confidence threshold for keeping a bus detection.
        car_conf_thresh: Confidence threshold for keeping a car detection.
        person_conf_thresh: Confidence threshold for keeping a person detection.
        motorcycle_conf_thresh: Confidence threshold for keeping a motorcycle detection.
        debug_trucks: If True, save cropped truck detections to outputs/debug/truck_detections/
        stats: Mutable dictionary to update global run statistics.

    Returns:
        List of postprocessed detections.
    """
    postprocessed: List[Dict] = []

    for det in detections:
        class_id = det["class_id"]
        confidence = det["confidence"]
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        area = w * h

        # -------------------------------------------------------------------
        # Bus filtering
        # -------------------------------------------------------------------
        if class_id == BUS_CLASS_ID:
            if confidence < bus_conf_thresh:
                if stats is not None:
                    stats["bus_filtered_out"] = stats.get("bus_filtered_out", 0) + 1
                continue
            postprocessed.append(det)
            if stats is not None:
                stats["bus_detections"] = stats.get("bus_detections", 0) + 1
            continue

        # -------------------------------------------------------------------
        # Truck postprocessing (Heuristics + Relabeling / Filtering)
        # -------------------------------------------------------------------
        if class_id == TRUCK_CLASS_ID:
            is_valid_size = (
                area >= truck_min_area
                and w >= truck_min_width
                and h >= truck_min_height
            )
            is_valid_conf = confidence >= truck_conf_thresh

            action = "keep"
            final_class_id = TRUCK_CLASS_ID
            final_class_name = "truck"

            if not is_valid_size or not is_valid_conf:
                # Ambiguous detection: try to relabel as car (class-frequency prior)
                if confidence >= car_conf_thresh:
                    action = "relabel"
                    final_class_id = CAR_CLASS_ID
                    final_class_name = "car"
                else:
                    action = "discard"

            # Debug crop export for every candidate truck
            if debug_trucks:
                debug_dir = Path("outputs/debug/truck_detections")
                debug_dir.mkdir(parents=True, exist_ok=True)
                
                # Get integer crop coordinates bounded by frame dimensions
                H_f, W_f = frame.shape[:2]
                rx1 = max(0, int(round(x1)))
                ry1 = max(0, int(round(y1)))
                rx2 = min(W_f, int(round(x2)))
                ry2 = min(H_f, int(round(y2)))
                
                if rx2 > rx1 and ry2 > ry1:
                    crop = frame[ry1:ry2, rx1:rx2].copy()
                    filename = (
                        f"frame_{frame_idx:06d}_conf_{confidence:.2f}_"
                        f"w{int(w)}_h{int(h)}_orig_truck_final_{final_class_name}.jpg"
                    )
                    cv2.imwrite(str(debug_dir / filename), crop)

            # Process action
            if action == "keep":
                postprocessed.append(det)
                if stats is not None:
                    stats["truck_detections"] = stats.get("truck_detections", 0) + 1
                    stats["raw_truck_predictions"] = stats.get("raw_truck_predictions", 0) + 1
            elif action == "relabel":
                relabeled_det = {
                    **det,
                    "class_id": CAR_CLASS_ID,
                    "class_name": "car",
                }
                postprocessed.append(relabeled_det)
                if stats is not None:
                    stats["car_detections"] = stats.get("car_detections", 0) + 1
                    stats["truck_to_car_relabels"] = stats.get("truck_to_car_relabels", 0) + 1
                    stats["raw_truck_predictions"] = stats.get("raw_truck_predictions", 0) + 1
            elif action == "discard":
                if stats is not None:
                    stats["truck_filtered_out"] = stats.get("truck_filtered_out", 0) + 1
                    stats["raw_truck_predictions"] = stats.get("raw_truck_predictions", 0) + 1
            continue

        # -------------------------------------------------------------------
        # Other classes (person, car, motorcycle)
        # -------------------------------------------------------------------
        if class_id == 0:  # person
            if confidence < person_conf_thresh:
                continue
            postprocessed.append(det)
            if stats is not None:
                stats["person_detections"] = stats.get("person_detections", 0) + 1
        elif class_id == 3:  # motorcycle
            if confidence < motorcycle_conf_thresh:
                continue
            # DEBUG: surface tile-collision metadata from SahiDinoDetector (informational only)
            if det.get("is_tile_collision", False):
                logger.debug(
                    "Frame %d: motorcycle tile collision (cluster_size=%d, conf=%.3f) — "
                    "passed postprocessing threshold %.2f",
                    frame_idx, det.get("tile_origin_count", 1), confidence,
                    motorcycle_conf_thresh,
                )
            postprocessed.append(det)
            if stats is not None:
                stats["motorcycle_detections"] = stats.get("motorcycle_detections", 0) + 1
        elif class_id == CAR_CLASS_ID:
            if confidence < car_conf_thresh:
                continue
            postprocessed.append(det)
            if stats is not None:
                stats["car_detections"] = stats.get("car_detections", 0) + 1

    return postprocessed
