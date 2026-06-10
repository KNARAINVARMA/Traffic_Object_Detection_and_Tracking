import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class SceneProfiler:
    """
    Layer 5: Meta-Learning (Scene Profiles).
    Saves and loads optimal tracking hyperparameters based on the scene context.
    """
    def __init__(self, profile_dir: str = "../profiles"):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
    def save_profile(self, scene_name: str, params: Dict[str, Any]) -> None:
        path = self.profile_dir / f"{scene_name}.json"
        with open(path, "w") as f:
            json.dump(params, f, indent=4)
        logger.info(f"Saved optimal tracking profile to {path}")
        
    def load_profile(self, scene_name: str) -> Dict[str, Any]:
        path = self.profile_dir / f"{scene_name}.json"
        if path.exists():
            with open(path, "r") as f:
                logger.info(f"Loaded existing scene profile from {path}")
                return json.load(f)
        return {}


class BayesianController:
    """
    Layer 3: Adaptive Learning.
    Dynamically scales hyperparameters based on real-time anomaly rates.
    """
    def __init__(self):
        self.base_track_buffer = 30
        self.base_match_thresh = 0.8
        
        # Current active states
        self.current_track_buffer = self.base_track_buffer
        self.current_match_thresh = self.base_match_thresh
        
    def update_priors(self, anomaly_log: List[Dict]) -> None:
        """
        Adjust tracking parameters based on recent anomalies.
        """
        if not anomaly_log:
            return
            
        recent = anomaly_log[-50:] # Look at last 50 anomalies
        
        # Count anomaly types
        kinematic_shocks = sum(1 for a in recent if a["anomaly"] == "Kinematic Shock")
        identity_overlaps = sum(1 for a in recent if a["anomaly"] == "Identity Overlap")
        
        # Bayesian update logic
        if kinematic_shocks > 5:
            # Too much fragmentation / bad matching, increase buffer to give more time
            self.current_track_buffer = min(120, int(self.current_track_buffer * 1.2))
            logger.debug(f"[BayesianController] High Kinematic Shocks. Increased track_buffer to {self.current_track_buffer}")
            
        if identity_overlaps > 5:
            # Too many boxes colliding, decrease match threshold so we require stricter matches
            self.current_match_thresh = max(0.5, self.current_match_thresh * 0.9)
            logger.debug(f"[BayesianController] High Identity Overlap. Tightened match_thresh to {self.current_match_thresh:.2f}")
            
        # Decay back to base if no severe anomalies
        if kinematic_shocks <= 2 and self.current_track_buffer > self.base_track_buffer:
            self.current_track_buffer = max(self.base_track_buffer, int(self.current_track_buffer * 0.95))
            
        if identity_overlaps <= 2 and self.current_match_thresh < self.base_match_thresh:
            self.current_match_thresh = min(self.base_match_thresh, self.current_match_thresh * 1.05)
            
    def get_active_params(self) -> Dict[str, Any]:
        return {
            "track_buffer": self.current_track_buffer,
            "match_thresh": self.current_match_thresh
        }
