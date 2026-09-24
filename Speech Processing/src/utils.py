"""
Module Tiện ích (Utilities) xử lý tín hiệu âm thanh và tính toán độ đo Đánh giá (Metrics).
"""

import os
from typing import Dict, List, Union
import numpy as np
import torch
import jiwer


def calculate_wer_cer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    Tính toán Word Error Rate (WER) và Character Error Rate (CER) cho ASR/STT.
    """
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    return {
        "WER": float(wer),
        "CER": float(cer)
    }


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Tính toán Độ chính xác (Accuracy) cho bài toán Phân loại Âm thanh.
    """
    return float(np.mean(predictions == targets))


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, filepath: str):
    """
    Lưu trọng số mô hình PyTorch và trạng thái Optimizer vào đĩa.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(state, filepath)
    print(f"--> Saved checkpoint to '{filepath}'")


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filepath: str, device: str = "cpu"):
    """
    Tải lại trọng số mô hình từ checkpoint file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint '{filepath}' does not exist.")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"--> Loaded checkpoint from '{filepath}' (Epoch {checkpoint.get('epoch', 0)})")
    return checkpoint.get("epoch", 0)
