"""
Script Huấn luyện Mô hình CV PyTorch chuẩn Production (Training Pipeline).
"""

import os
import time
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.config import CVConfig, TrainConfig
from src.dataset import VideoDataset
from src.models.video_classifier_3d import VideoActionClassifier3D
from src.utils import save_cv_checkpoint


def train_cv_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """Huấn luyện 1 Epoch cho bài toán Video Action Classification."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for video_tensors, labels in dataloader:
        video_tensors = video_tensors.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(video_tensors)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * video_tensors.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    return avg_loss, acc


@torch.no_grad()
def evaluate_cv(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Đánh giá mô hình CV trên tập Validation."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for video_tensors, labels in dataloader:
        video_tensors = video_tensors.to(device)
        labels = labels.to(device)

        logits = model(video_tensors)
        loss = criterion(logits, labels)

        total_loss += loss.item() * video_tensors.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    return avg_loss, acc


def run_cv_training(
    train_videos: List[str],
    train_labels: List[int],
    val_videos: List[str],
    val_labels: List[int],
    num_classes: int = 10,
    train_cfg: TrainConfig = TrainConfig()
):
    """Pipeline tổng thể chạy huấn luyện mô hình 3D Video Classifier."""
    torch.manual_seed(train_cfg.seed)
    device = train_cfg.device if torch.cuda.is_available() else "cpu"
    print(f"--> Starting CV Training Pipeline on Device: '{device}'")

    train_dataset = VideoDataset(train_videos, train_labels)
    val_dataset = VideoDataset(val_videos, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False)

    model = VideoActionClassifier3D(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    best_val_acc = 0.0

    for epoch in range(1, train_cfg.num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_cv_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_cv(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{train_cfg.num_epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(train_cfg.checkpoint_dir, "best_video_classifier_3d.pt")
            save_cv_checkpoint(model, optimizer, epoch, save_path)

    print(f"--> CV Training Completed! Best Validation Accuracy: {best_val_acc*100:.2f}%")
