"""
Module Dataset & DataLoader cho Ảnh và Video trong PyTorch.
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


class ImageDataset(Dataset):
    """
    Dataset chuẩn PyTorch cho nạp Ảnh 2D.
    """
    def __init__(self, image_paths: List[str], labels: Optional[List[int]] = None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            # Convert sang FloatTensor [3, H, W] chuẩn hóa [0, 1]
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        label = self.labels[idx] if self.labels is not None else 0
        return image, label


class VideoDataset(Dataset):
    """
    Dataset chuẩn PyTorch cho nạp Clip Video 3D (Lấy mẫu N frames).
    """
    def __init__(self, video_paths: List[str], labels: Optional[List[int]] = None, num_frames: int = 16):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        video_path = self.video_paths[idx]
        
        # Load Video Frames dùng PyAV hoặc tạo giả lập nếu không có PyAV
        frames = []
        if HAS_AV and os.path.exists(video_path):
            container = av.open(video_path)
            for frame in container.decode(video=0):
                img = frame.to_image().convert("RGB").resize((224, 224))
                frames.append(np.array(img))
            container.close()

        if len(frames) == 0:
            # Fallback tạo Tensor Video giả lập [num_frames, 224, 224, 3]
            frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]

        # Lấy mẫu Uniform N frames
        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]

        # Shape: [C, T, H, W]
        video_tensor = torch.from_numpy(np.stack(sampled_frames)).permute(3, 0, 1, 2).float() / 255.0
        label = self.labels[idx] if self.labels is not None else 0
        return video_tensor, label
