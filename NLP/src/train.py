"""
Script Huấn luyện Mô hình NLP PyTorch chuẩn Production (Training Pipeline).
"""

import os
import time
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.config import NLPConfig, ModelConfig, TrainConfig
from src.tokenizer import VietnameseNLPTokenizer
from src.dataset import TextClassificationDataset, text_collate_fn
from src.models.text_classifier import TransformerTextClassifier
from src.metrics import NLPMetricsCalculator
from src.utils import save_nlp_checkpoint


def train_nlp_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str
) -> Tuple[float, float, float]:
    """Huấn luyện 1 Epoch cho bài toán Phân loại Văn bản."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = NLPMetricsCalculator.compute_accuracy_f1(np.array(all_preds), np.array(all_targets))
    return avg_loss, metrics["Accuracy"], metrics["Macro_F1"]


@torch.no_grad()
def evaluate_nlp(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float, float]:
    """Đánh giá mô hình trên tập Validation/Test."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = NLPMetricsCalculator.compute_accuracy_f1(np.array(all_preds), np.array(all_targets))
    return avg_loss, metrics["Accuracy"], metrics["Macro_F1"]


def run_nlp_training(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    nlp_cfg: NLPConfig = NLPConfig(),
    model_cfg: ModelConfig = ModelConfig(),
    train_cfg: TrainConfig = TrainConfig()
):
    """Pipeline tổng thể chạy huấn luyện mô hình NLP."""
    torch.manual_seed(train_cfg.seed)
    device = train_cfg.device if torch.cuda.is_available() else "cpu"
    print(f"--> Starting NLP Training Pipeline on Device: '{device}'")

    tokenizer_wrapper = VietnameseNLPTokenizer(nlp_cfg)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer_wrapper)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer_wrapper)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=text_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=text_collate_fn)

    model = TransformerTextClassifier(nlp_cfg.model_name_or_path, model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    best_val_f1 = 0.0

    for epoch in range(1, train_cfg.num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc, train_f1 = train_nlp_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_nlp(model, val_loader, criterion, device)

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{train_cfg.num_epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% - F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% - F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(train_cfg.checkpoint_dir, "best_text_classifier.pt")
            save_nlp_checkpoint(model, optimizer, epoch, save_path)

    print(f"--> NLP Training Completed! Best Validation Macro-F1: {best_val_f1:.4f}")
