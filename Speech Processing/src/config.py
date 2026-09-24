"""
Module Cấu hình (Configuration) cho Hệ thống Xử lý Tín hiệu & Tiếng nói.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AudioConfig:
    """Cấu hình cho Trích xuất Đặc trưng & Xử lý Tín hiệu Âm thanh."""
    sample_rate: int = 16000
    mono: bool = True
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    n_mfcc: int = 13
    f_min: float = 0.0
    f_max: Optional[float] = 8000.0
    power: float = 2.0
    normalize_signal: bool = True
    
    # Cấu hình SpecAugment
    apply_spec_augment: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 35


@dataclass
class ModelConfig:
    """Cấu hình Kiến trúc Mô hình Deep Learning Audio Classifier."""
    model_name: str = "resnet_attention_classifier"
    in_channels: int = 1
    num_classes: int = 10
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dropout: float = 0.3
    use_attention_pooling: bool = True


@dataclass
class TrainConfig:
    """Cấu hình Huấn luyện Mô hình (Training Configurations)."""
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"  # Hoặc "cpu"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42
