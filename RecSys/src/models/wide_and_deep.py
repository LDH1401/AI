"""
Mô hình Wide & Deep Ranking (Google 2016) trong PyTorch.
"""

from typing import List
import torch
import torch.nn as nn


class WideAndDeepModel(nn.Module):
    """
    Triển khai mô hình Wide & Deep Ranking trong PyTorch.
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, deep_hidden_dims: List[int] = [128, 64]):
        super().__init__()
        # 1. Embeddings cho Deep Component
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # 2. Linear Layers cho Wide Component
        self.wide_user_linear = nn.Embedding(num_users, 1)
        self.wide_item_linear = nn.Embedding(num_items, 1)

        # 3. MLP cho Deep Component
        layers = []
        in_dim = embedding_dim * 2
        for h_dim in deep_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.deep_mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        user_ids shape: [batch_size]
        item_ids shape: [batch_size]
        Returns: CTR Logits [batch_size, 1]
        """
        # Wide Component Score
        wide_score = self.wide_user_linear(user_ids) + self.wide_item_linear(item_ids)  # [batch, 1]

        # Deep Component Score
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)
        deep_in = torch.cat([u_emb, i_emb], dim=-1)  # [batch, 2 * embed_dim]
        deep_score = self.deep_mlp(deep_in)          # [batch, 1]

        # Combined Logits
        logits = wide_score + deep_score
        return logits.squeeze(-1)
