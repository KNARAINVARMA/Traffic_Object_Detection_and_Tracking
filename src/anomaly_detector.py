import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class AnomalyType:
    KINEMATIC_SHOCK = "Kinematic Shock"
    IDENTITY_OVERLAP = "Identity Overlap"
    CLASS_FLICKER = "Class Flickering"
    CONFIDENCE_DROP = "Confidence Drop"

class RCA_Source:
    DETECTION = "Detection Failure"
    MATCHING = "Matching Failure"
    CLASSIFICATION = "Classification Failure"

class AnomalyDetector:
    """
    Passively observes tracking data in real-time to detect physical impossibilities
    and tracking errors.
    """
    def __init__(self, max_speed_px_per_frame: float = 150.0):
        self.max_speed = max_speed_px_per_frame
        self.track_history: Dict[int, Dict] = {}
        self.anomaly_log: List[Dict] = []
        
        # Phase 4: Active Learning Setup
        self.needs_training_dir = Path("../outputs/needs_training")
        self.needs_training_dir.mkdir(parents=True, exist_ok=True)
        
    def observe(self, tracked_stracks: List["STrack"], frame_id: int, frame: Optional[np.ndarray] = None) -> None:
        """
        Scan all active tracks for anomalies.
        """
        current_ids = set()
        
        # 1. Identity Overlap (O(N^2) but N is small)
        for i in range(len(tracked_stracks)):
            t1 = tracked_stracks[i]
            current_ids.add(t1.track_id)
            
            # Kinematic Shock
            hist = self.track_history.get(t1.track_id)
            if hist is not None:
                dx = t1.center[0] - hist["center"][0]
                dy = t1.center[1] - hist["center"][1]
                speed = np.hypot(dx, dy)
                if speed > self.max_speed and t1.lost_frames == 0:
                    self._log_anomaly(t1.track_id, frame_id, AnomalyType.KINEMATIC_SHOCK, RCA_Source.MATCHING)
                    
                # Class Flickering
                if t1.class_id != hist["class_id"]:
                    self._log_anomaly(t1.track_id, frame_id, AnomalyType.CLASS_FLICKER, RCA_Source.CLASSIFICATION, frame, t1.bbox_xyxy)
                    
                # Confidence Drop
                if hist["score"] - t1.score > 0.4:
                    self._log_anomaly(t1.track_id, frame_id, AnomalyType.CONFIDENCE_DROP, RCA_Source.DETECTION, frame, t1.bbox_xyxy)

            for j in range(i + 1, len(tracked_stracks)):
                t2 = tracked_stracks[j]
                iou = self._calculate_iou(t1.bbox_xyxy, t2.bbox_xyxy)
                if iou > 0.8:
                    self._log_anomaly(f"{t1.track_id}-{t2.track_id}", frame_id, AnomalyType.IDENTITY_OVERLAP, RCA_Source.MATCHING)
                    
            # Update history
            self.track_history[t1.track_id] = {
                "center": t1.center,
                "score": t1.score,
                "class_id": t1.class_id,
                "bbox": t1.bbox_xyxy,
                "frame": frame_id
            }
            
        # Clean up history for removed tracks
        stale_ids = [tid for tid in self.track_history.keys() if tid not in current_ids and (frame_id - self.track_history[tid]["frame"] > 60)]
        for tid in stale_ids:
            del self.track_history[tid]

    def _log_anomaly(self, target: str, frame: int, anomaly: str, rca: str, raw_img: Optional[np.ndarray] = None, bbox: Optional[List[float]] = None) -> None:
        self.anomaly_log.append({
            "target": target,
            "frame": frame,
            "anomaly": anomaly,
            "rca": rca
        })
        logger.debug(f"[AnomalyDetector] Frame {frame}: {anomaly} detected on {target}. Root Cause: {rca}")
        
        # Phase 4: Save edge cases to needs_training/
        if rca in [RCA_Source.DETECTION, RCA_Source.CLASSIFICATION] and raw_img is not None and bbox is not None:
            try:
                x1, y1, x2, y2 = map(int, bbox)
                H, W = raw_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 > x1 and y2 > y1:
                    crop = raw_img[y1:y2, x1:x2]
                    filename = self.needs_training_dir / f"frame_{frame}_{anomaly.replace(' ', '_')}_{target}.jpg"
                    cv2.imwrite(str(filename), crop)
            except Exception as e:
                logger.warning(f"Failed to save active learning crop: {e}")

    def _calculate_iou(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
