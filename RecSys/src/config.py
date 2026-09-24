"""
Module Cấu hình (Configuration) cho Hệ thống Recommendation Systems (RecSys).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RecSysConfig:
    """Cấu hình số lượng User và Item."""
    num_users: int = 10000
    num_items: int = 50000
    embedding_dim: int = 64
    max_seq_length: int = 20  # Dành cho Sequential RecSys (SASRec)


@dataclass
class ModelConfig:
    """Cấu hình Kiến trúc Mô hình RecSys (Two-Tower / Wide & Deep / BPR)."""
    model_type: str = "two_tower"  # "bpr_mf", "two_tower", "wide_and_deep", "sasrec"
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    temperature: float = 0.07  # For In-batch Softmax Loss


@dataclass
class TrainConfig:
    """Cấu hình Huấn luyện Mô hình RecSys."""
    batch_size: int = 256
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda"  # Hoặc "cpu"
    checkpoint_dir: str = "./checkpoints_recsys"
    seed: int = 42
