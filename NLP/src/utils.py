"""
Module Tiện ích (Utilities) cho NLP: Chuẩn hóa Văn bản, Save/Load Checkpoints.
"""

import os
import re
import unicodedata
import torch


def clean_text(text: str) -> str:
    """
    Hàm làm sạch văn bản tổng quát: Chuẩn hóa Unicode NFC, xóa HTML, URL.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def save_nlp_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, filepath: str):
    """
    Lưu checkpoint trọng số mô hình NLP.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None
    }
    torch.save(state, filepath)
    print(f"--> Saved NLP checkpoint to '{filepath}'")


def load_nlp_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str, device: str = "cpu"):
    """
    Tải lại trọng số mô hình NLP từ file checkpoint.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint '{filepath}' not found.")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"--> Loaded NLP checkpoint from '{filepath}' (Epoch {checkpoint.get('epoch', 0)})")
    return checkpoint.get("epoch", 0)
