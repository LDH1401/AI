"""
Module Tokenizer hỗ trợ Tách từ Tiếng Việt (PyVi) và Subword Tokenization (Hugging Face).
"""

from typing import Dict, List, Union
import unicodedata
import torch
from transformers import AutoTokenizer
from pyvi import ViTokenizer
from src.config import NLPConfig


class VietnameseNLPTokenizer:
    """
    Wrapper bọc Tokenizer xử lý văn bản tiếng Việt kết hợp Tách từ PyVi & Subword Tokenization.
    """
    def __init__(self, config: NLPConfig):
        self.config = config
        print(f"--> Initializing Tokenizer from '{config.model_name_or_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    def preprocess_text(self, text: str) -> str:
        """Chuẩn hóa Unicode NFC và tách từ ghép tiếng Việt bằng PyVi nếu được cấu hình."""
        if not text:
            return ""
        # 1. Unicode NFC Normalization
        text = unicodedata.normalize("NFC", text)
        
        # 2. Tách từ ghép tiếng Việt với PyVi
        if self.config.use_vietnamese_segmenter:
            text = ViTokenizer.tokenize(text)
            
        return text

    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Mã hóa 1 chuỗi văn bản thành Tensors input_ids và attention_mask."""
        clean_text = self.preprocess_text(text)
        inputs = self.tokenizer(
            clean_text,
            max_length=self.config.max_seq_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors="pt"
        )
        return inputs

    def batch_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Mã hóa một Batch danh sách các văn bản."""
        cleaned_texts = [self.preprocess_text(t) for t in texts]
        inputs = self.tokenizer(
            cleaned_texts,
            max_length=self.config.max_seq_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors="pt"
        )
        return inputs
