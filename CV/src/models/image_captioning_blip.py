"""
Module Image Captioning (Mô tả ảnh bằng Ngôn ngữ Tự nhiên) sử dụng Hugging Face Pretrained BLIP / BLIP-2.
"""

from typing import List, Union
import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from src.config import VLMConfig


class BLIPImageCaptioner:
    """
    Wrapper bọc mô hình Hugging Face Pretrained BLIP (Salesforce/blip-image-captioning-base) cho Image Captioning.
    """
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--> Initializing Pretrained Hugging Face BLIP Model from '{model_name}' on device '{self.device}'...")
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_caption(self, image_input: Union[str, Image.Image], prompt: str = None) -> str:
        """
        Nhận đầu vào là PIL Image hoặc Đường dẫn file ảnh.
        Returns: Câu văn bản mô tả nội dung bức ảnh.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input

        # Trích xuất đặc trưng Ảnh & Tokenize Prompt nếu có
        if prompt:
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        # Autoregressive Generation qua Hugging Face Model
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=50)

        caption = self.processor.decode(output_ids[0], skip_special_tokens=True)
        print(f"--> Generated Image Caption: \"{caption}\"")
        return caption.strip()


class CustomVisionLanguageProjection(nn.Module):
    """
    Tự thiết lập (Custom PyTorch Setup): Lớp Ma trận Chiếu Tuyến tính (Linear Projection) 
    giữa Vision Encoder Vector (ViT) và Language Model Embedding.
    """
    def __init__(self, vision_dim: int = 768, text_dim: int = 768):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        """
        vision_embeds shape: [batch_size, num_patches, vision_dim]
        Returns: Projected Tokens shape [batch_size, num_patches, text_dim]
        """
        return self.projection(vision_embeds)
