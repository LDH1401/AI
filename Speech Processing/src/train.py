"""
Script Huấn luyện Mô hình Phân loại Âm thanh PyTorch (Training Pipeline).
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.config import AudioConfig, ModelConfig, TrainConfig
from src.dataset import AudioClassificationDataset, audio_collate_fn
from src.features import AudioFeatureExtractor
from src.models.classifier import AudioClassifier
from src.utils import compute_accuracy, save_checkpoint


def train_epoch(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """Huấn luyện 1 Epoch."""
    model.train()
    feature_extractor.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # Trích xuất Mel Spectrogram on-the-fly với SpecAugment
        log_mel_specs = feature_extractor(waveforms, augment=True)

        optimizer.zero_grad()
        logits = model(log_mel_specs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * waveforms.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = compute_accuracy(np.array(all_preds), np.array(all_targets))
    return avg_loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Đánh giá mô hình trên tập Validation/Test."""
    model.eval()
    feature_extractor.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for waveforms, labels in dataloader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        log_mel_specs = feature_extractor(waveforms, augment=False)
        logits = model(log_mel_specs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * waveforms.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = compute_accuracy(np.array(all_preds), np.array(all_targets))
    return avg_loss, acc


def run_training_pipeline(
    train_paths: List[str],
    train_labels: List[int],
    val_paths: List[str],
    val_labels: List[int],
    audio_cfg: AudioConfig = AudioConfig(),
    model_cfg: ModelConfig = ModelConfig(),
    train_cfg: TrainConfig = TrainConfig()
):
    """Pipeline tổng thể chạy huấn luyện mô hình."""
    torch.manual_seed(train_cfg.seed)
    device = train_cfg.device if torch.cuda.is_available() else "cpu"
    print(f"--> Starting Audio Training Pipeline on Device: '{device}'")

    # Datasets & DataLoaders
    train_dataset = AudioClassificationDataset(train_paths, train_labels, audio_cfg)
    val_dataset = AudioClassificationDataset(val_paths, val_labels, audio_cfg)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=audio_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=audio_collate_fn)

    # Models & Extractor
    feature_extractor = AudioFeatureExtractor(audio_cfg).to(device)
    model = AudioClassifier(model_cfg).to(device)

    # Criterion (Cross-Entropy Loss / NLL) & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.num_epochs)

    best_val_acc = 0.0

    for epoch in range(1, train_cfg.num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, feature_extractor, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, feature_extractor, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{train_cfg.num_epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(train_cfg.checkpoint_dir, "best_audio_classifier.pt")
            save_checkpoint(model, optimizer, epoch, save_path)

    print(f"--> Training Completed! Best Validation Accuracy: {best_val_acc*100:.2f}%")
