"""
reid.py — Lightweight Appearance Feature Extractor
This module implements a hybrid appearance descriptor for multi-object tracking,
combining CNN spatial features (MobileNetV3 or custom CNN fallback) and
color features (3D HSV Color Histograms). This allows maintaining identity
across long intervals, occlusions, and dense traffic clusters.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """
    Fallback lightweight CNN for feature extraction when pre-trained weights
    cannot be loaded from the internet.
    """
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


class ReIDExtractor:
    """
    Extracts high-quality appearance embeddings.
    Combines unit-normalized MobileNetV3 CNN embeddings and 3D HSV Color Histograms.
    """
    def __init__(self, device: Optional[str] = None, embedding_dim: int = 128) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim
        self.model: nn.Module
        
        # Attempt to load a pre-trained MobileNetV3 small model
        try:
            logger.info("Attempting to load pre-trained MobileNetV3 for ReID...")
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            backbone = models.mobilenet_v3_small(weights=weights)
            # Replace the classifier block with a projection layer to embedding_dim
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Sequential(
                nn.Linear(in_features, embedding_dim),
                nn.BatchNorm1d(embedding_dim)
            )
            self.model = backbone
            logger.info("Successfully loaded pre-trained MobileNetV3 for ReID.")
        except Exception as e:
            logger.warning(
                "Could not load pre-trained MobileNetV3 weights (%s). "
                "Falling back to custom lightweight CNN with random initialization.",
                e
            )
            self.model = SimpleCNN(embedding_dim=embedding_dim)
            
        self.model.to(self.device)
        self.model.eval()

    def extract_cnn_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract unit-normalized deep CNN features from BGR image crops.
        """
        if not crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
            
        tensors = []
        for crop in crops:
            if crop.size == 0:
                crop = np.zeros((64, 64, 3), dtype=np.uint8)
            img = cv2.resize(crop, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            # Normalize with standard ImageNet statistics
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            
            tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
            tensors.append(tensor)
            
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch)
            # Unit normalization
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy()

    def extract_color_hist(self, crop: np.ndarray) -> np.ndarray:
        """
        Compute a normalized 3D HSV Color Histogram.
        Bins: 8 H bins, 8 S bins, 8 V bins = 512 dimensions.
        """
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def extract_combined(self, crops: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
        """
        Extract combined CNN and Color features from list of BGR image crops.
        """
        if not crops:
            return []
            
        cnn_embs = self.extract_cnn_features(crops)
        results = []
        for i, crop in enumerate(crops):
            color_emb = self.extract_color_hist(crop)
            results.append({
                "cnn": cnn_embs[i],
                "color": color_emb,
            })
        return results


def compute_appearance_distance(
    emb1: Dict[str, np.ndarray],
    emb2: Dict[str, np.ndarray],
    w_cnn: float = 0.5,
    w_color: float = 0.5,
) -> float:
    """
    Compute a combined appearance distance between two appearance descriptors.
    Uses cosine distance for both CNN features and HSV Color Histograms.
    
    Cosine Distance = 1.0 - Cosine Similarity
    """
    # CNN Cosine Similarity (dot product since they are unit-normalized)
    cnn_sim = float(np.dot(emb1["cnn"], emb2["cnn"]))
    d_cnn = 1.0 - max(-1.0, min(1.0, cnn_sim))
    
    # Color Cosine Similarity
    color_norm1 = np.linalg.norm(emb1["color"]) + 1e-7
    color_norm2 = np.linalg.norm(emb2["color"]) + 1e-7
    color_sim = float(np.dot(emb1["color"], emb2["color"])) / (color_norm1 * color_norm2)
    d_color = 1.0 - max(-1.0, min(1.0, color_sim))
    
    return w_cnn * d_cnn + w_color * d_color
