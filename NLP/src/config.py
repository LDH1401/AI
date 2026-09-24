"""
Module Cấu hình (Configuration) cho Hệ thống NLP.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NLPConfig:
    """Cấu hình cho Tiền xử lý & Tokenizer."""
    model_name_or_path: str = "vinai/phobert-base"
    max_seq_length: int = 256
    do_lowercase: bool = False
    use_vietnamese_segmenter: bool = True
    padding: str = "max_length"
    truncation: bool = True


@dataclass
class ModelConfig:
    """Cấu hình Mô hình Deep Learning NLP."""
    model_type: str = "classifier"  # "classifier", "qa", "seq2seq"
    num_classes: int = 3
    hidden_dropout_prob: float = 0.2
    use_attention_head: bool = True


@dataclass
class TrainConfig:
    """Cấu hình Huấn luyện Mô hình NLP."""
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    device: str = "cuda"  # Hoặc "cpu"
    checkpoint_dir: str = "./checkpoints_nlp"
    seed: int = 42
