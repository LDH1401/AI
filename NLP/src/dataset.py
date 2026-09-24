"""
Module Dataset & Collate Function cho các bài toán NLP PyTorch.
"""

from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from src.tokenizer import VietnameseNLPTokenizer


class TextClassificationDataset(Dataset):
    """
    Dataset chuẩn PyTorch cho bài toán Phân loại Văn bản / Sentiment Analysis.
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer_wrapper: VietnameseNLPTokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer_wrapper = tokenizer_wrapper

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoded = self.tokenizer_wrapper.encode(text)
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def text_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function gộp Batch các tensor NLP.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
