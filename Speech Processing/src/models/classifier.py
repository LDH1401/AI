"""
Mô hình Deep Learning Audio Classifier (CNN 2D ResNet Block + Attention Pooling Head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig


class ConvBlock2D(nn.Module):
    """Khối Convolution 2D chuẩn gồm Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionPooling(nn.Module):
    """
    Cơ chế Self-Attention Pooling gom tụ trục thời gian (Time Dimension) linh hoạt theo trọng số chú ý.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, time_frames, feature_dim]
        weights = self.attention(x)  # [batch_size, time_frames, 1]
        weights = F.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1)  # [batch_size, feature_dim]
        return pooled


class AudioClassifier(nn.Module):
    """
    Mô hình Phân loại Âm thanh 2D CNN với Attention Pooling Head.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        layers = []
        in_ch = config.in_channels
        for out_ch in config.conv_channels:
            layers.append(ConvBlock2D(in_ch, out_ch))
            in_ch = out_ch
            
        self.encoder = nn.Sequential(*layers)
        last_conv_dim = config.conv_channels[-1]

        self.attention_pool = AttentionPooling(feature_dim=last_conv_dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(last_conv_dim, config.num_classes)

    def forward(self, log_mel_spec: torch.Tensor) -> torch.Tensor:
        """
        log_mel_spec shape: [batch_size, 1, n_mels, time_frames]
        Returns: Logits Tensor [batch_size, num_classes]
        """
        # Feature extraction qua các tầng CNN 2D
        features = self.encoder(log_mel_spec)  # [batch_size, C, Mel_reduced, Time_reduced]

        # Lấy trung bình trên trục Mel tần số
        features = torch.mean(features, dim=2)  # [batch_size, C, Time_reduced]
        features = features.permute(0, 2, 1)    # [batch_size, Time_reduced, C]

        # Gom tụ thời gian bằng Attention Pooling
        pooled = self.attention_pool(features)  # [batch_size, C]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)        # [batch_size, num_classes]
        return logits
