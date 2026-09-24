"""
Module Visual Question Answering (VQA) sử dụng Pretrained Hugging Face BLIP-VQA / Multimodal Fusion.
"""

from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from src.config import VLMConfig


class PretrainedVisualQAPipeline:
    """
    Wrapper bọc mô hình Hugging Face Pretrained BLIP-VQA (Salesforce/blip-vqa-base).
    """
    def __init__(self, model_name: str = "Salesforce/blip-vqa-base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--> Initializing Pretrained Hugging Face VQA Model from '{model_name}' on device '{self.device}'...")
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def answer_question(self, image_input: Union[str, Image.Image], question: str) -> str:
        """
        Nhận đầu vào là Bức ảnh và Câu hỏi tự do.
        Returns: Câu trả lời dự đoán bằng ngôn ngữ tự nhiên.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input

        # Trích xuất đặc trưng Đa thức (Multimodal Inputs)
        inputs = self.processor(image, text=question, return_tensors="pt").to(self.device)

        # Predict Answer Tokens
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=30)

        answer = self.processor.decode(output_ids[0], skip_special_tokens=True)
        print(f"--> Question: \"{question}\"")
        print(f"--> VQA Answer: \"{answer}\"")
        return answer.strip()


class CustomCoAttentionMultimodalFusion(nn.Module):
    """
    Tự thiết lập (Custom PyTorch Setup): Lớp Co-Attention Fusion 
    dung hợp Vector Ảnh (Vision Embeddings) và Vector Câu hỏi (Text Embeddings).
    """
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.vision_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

    def forward(self, vision_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        vision_embeds: [batch, num_patches, embed_dim]
        text_embeds: [batch, seq_len, embed_dim]
        Returns: Fused Multimodal Feature Vector [batch, embed_dim]
        """
        v_proj = self.vision_proj(vision_embeds)
        t_proj = self.text_proj(text_embeds)

        # Cross-Attention: Query = Text, Key/Value = Vision
        attn_out, _ = self.attn(query=t_proj, key=v_proj, value=v_proj)
        
        # Mean Pooling
        fused_vector = torch.mean(attn_out, dim=1)
        return fused_vector
