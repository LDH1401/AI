"""
Script Huấn luyện Mô hình RecSys PyTorch chuẩn Production (Training Pipeline).
"""

import os
import time
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.config import RecSysConfig, ModelConfig, TrainConfig
from src.dataset import BPRUserItemDataset
from src.models.matrix_factorization import BPRMatrixFactorization
from src.models.two_tower_dssm import TwoTowerDSSMModel
from src.utils import save_recsys_checkpoint


def train_recsys_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    model_type: str = "two_tower"
) -> float:
    """Huấn luyện 1 Epoch cho mô hình RecSys."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        if model_type == "bpr_mf":
            users, pos_items, neg_items = [b.to(device) for b in batch]
            loss = model(users, pos_items, neg_items)
        elif model_type == "two_tower":
            users, pos_items, _ = [b.to(device) for b in batch]
            loss, _ = model(users, pos_items)
        else:
            users, pos_items, _ = [b.to(device) for b in batch]
            loss = model(users, pos_items).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch[0].size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def run_recsys_training(
    user_item_pairs: List[Tuple[int, int]],
    recsys_cfg: RecSysConfig = RecSysConfig(),
    model_cfg: ModelConfig = ModelConfig(),
    train_cfg: TrainConfig = TrainConfig()
):
    """Pipeline tổng thể chạy huấn luyện mô hình RecSys."""
    torch.manual_seed(train_cfg.seed)
    device = train_cfg.device if torch.cuda.is_available() else "cpu"
    print(f"--> Starting RecSys Training Pipeline ({model_cfg.model_type}) on Device: '{device}'")

    dataset = BPRUserItemDataset(user_item_pairs, num_items=recsys_cfg.num_items)
    dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    if model_cfg.model_type == "bpr_mf":
        model = BPRMatrixFactorization(recsys_cfg.num_users, recsys_cfg.num_items, recsys_cfg.embedding_dim).to(device)
    else:
        model = TwoTowerDSSMModel(recsys_cfg.num_users, recsys_cfg.num_items, recsys_cfg.embedding_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    best_loss = float("inf")

    for epoch in range(1, train_cfg.num_epochs + 1):
        start_time = time.time()
        loss = train_recsys_epoch(model, dataloader, optimizer, device, model_cfg.model_type)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{train_cfg.num_epochs:02d}] ({elapsed:.1f}s) | RecSys Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            save_path = os.path.join(train_cfg.checkpoint_dir, f"best_{model_cfg.model_type}_model.pt")
            save_recsys_checkpoint(model, optimizer, epoch, save_path)

    print(f"--> RecSys Training Completed! Best Loss: {best_loss:.4f}")
