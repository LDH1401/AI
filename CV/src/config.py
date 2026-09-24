"""
Module Cấu hình (Configuration) cho Hệ thống Advanced CV.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class CVConfig:
    """Cấu hình cho Tiền xử lý & Kích thước Đầu vào Hình ảnh/Video."""
    image_size: Tuple[int, int] = (224, 224)
    num_frames: int = 16  # Số khung hình lấy mẫu cho Video
    sample_rate: int = 2
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class DetectionConfig:
    """Cấu hình cho Bài toán Advanced Object Detection (YOLO / DETR)."""
    model_name: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    num_classes: int = 80


@dataclass
class VLMConfig:
    """Cấu hình cho Vision-Language Models (BLIP / LLaVA / VQA)."""
    caption_model_name: str = "Salesforce/blip-image-captioning-base"
    vqa_model_name: str = "Salesforce/blip-vqa-base"
    max_new_tokens: int = 50


@dataclass
class TrainConfig:
    """Cấu hình Huấn luyện Mô hình CV."""
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda"  # Hoặc "cpu"
    checkpoint_dir: str = "./checkpoints_cv"
    seed: int = 42
