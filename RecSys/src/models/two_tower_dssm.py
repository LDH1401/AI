"""
Mô hình Two-Tower DSSM Candidate Retrieval trong PyTorch.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig


class UserTower(nn.Module):
    """
    Tháp Người dùng (User Tower): Chiếu User ID và User Features về Vector u (d-dimensional).
    """
    def __init__(self, num_users: int, embedding_dim: int = 64, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        layers = []
        in_dim = embedding_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.user_embedding(user_ids)
        user_vector = self.mlp(embeds)
        # L2 Normalization
        return F.normalize(user_vector, p=2, dim=1)


class ItemTower(nn.Module):
    """
    Tháp Sản phẩm (Item Tower): Chiếu Item ID và Item Features về Vector v (d-dimensional).
    """
    def __init__(self, num_items: int, embedding_dim: int = 64, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers = []
        in_dim = embedding_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.item_embedding(item_ids)
        item_vector = self.mlp(embeds)
        # L2 Normalization
        return F.normalize(item_vector, p=2, dim=1)


class TwoTowerDSSMModel(nn.Module):
    """
    Kiến trúc Mô hình Hai Tháp Two-Tower DSSM hoàn chỉnh.
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_items, embedding_dim)
        self.temperature = temperature

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates In-batch Negative Softmax Loss.
        """
        u_vector = self.user_tower(user_ids)  # [batch, d]
        v_vector = self.item_tower(item_ids)  # [batch, d]

        # Similarity Matrix: [batch, batch]
        logits = torch.matmul(u_vector, v_vector.T) / self.temperature
        
        # In-batch Targets: Đường chéo chính (i == j)
        targets = torch.arange(user_ids.size(0), device=user_ids.device)
        loss = F.cross_entropy(logits, targets)

        return loss, logits
