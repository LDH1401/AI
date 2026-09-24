"""
Mô hình Deep Learning Video Action Recognition Classifier (3D ResNet / Conv3D Model).
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models
from src.config import CVConfig


class VideoActionClassifier3D(nn.Module):
    """
    Mô hình Phân loại Hành động trong Video dựa trên kiến trúc 3D ResNet (ResNet3D-18 / MC3-18).
    """
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        print("--> Initializing 3D ResNet Video Action Classifier...")
        
        # Load backbone 3D ResNet từ torchvision
        weights = video_models.R3D_18_Weights.DEFAULT if pretrained else None
        self.backbone = video_models.r3d_18(weights=weights)
        
        # Thay thế Linear Classification Head phía cuối
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        video_tensor shape: [batch_size, 3, T, H, W] (3D Video Stream Tensor)
        Returns: Logits Tensor [batch_size, num_classes]
        """
        logits = self.backbone(video_tensor)
        return logits
