"""
Module Speaker Verification & Diarization (Trích xuất đặc trưng người nói ECAPA-TDNN).
"""

from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimplifiedECAPATDNN(nn.Module):
    """
    Minh họa kiến trúc ECAPA-TDNN trích xuất Speaker Embeddings (192-dimensional vector).
    """
    def __init__(self, in_channels: int = 80, channels: int = 512, lin_neurons: int = 192):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        
        # Pooling Layer (Mean + Std Attention Pooling)
        self.fc = nn.Linear(channels * 2, lin_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [batch_size, num_mels=80, num_frames]
        Returns: Speaker Embedding [batch_size, 192]
        """
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Mean & Std Pooling trên trục thời gian
        mean = torch.mean(out, dim=2)
        std = torch.std(out, dim=2)
        pooled = torch.cat([mean, std], dim=1)

        embedding = self.fc(pooled)
        return F.normalize(embedding, p=2, dim=1)
