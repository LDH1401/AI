"""
Module Tiện ích cho RL: Thiết lập Random Seed, Lưu Checkpoint Trọng số Agent.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Cố định Random Seed cho PyTorch, NumPy và Random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rl_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, episode: int, filepath: str):
    """Lưu trọng số mô hình RL vào file checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "episode": episode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None
    }
    torch.save(state, filepath)
    print(f"--> Saved RL checkpoint to '{filepath}'")
