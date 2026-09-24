"""
Mô hình Matrix Factorization & Bayesian Personalized Ranking (BPR Loss) trong PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRMatrixFactorization(nn.Module):
    """
    Triển khai mô hình BPR-MF (Bayesian Personalized Ranking Matrix Factorization).
    """
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Khởi tạo trọng số ngẫu nhiên nhỏ
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor = None) -> torch.Tensor:
        """
        Nếu truyền neg_items: Tính BPR Loss cho Pairwise Ranking.
        Nếu neg_items = None: Trả về điểm dự đoán score(u, pos_item).
        """
        u_emb = self.user_embeddings(users)            # [batch, embed_dim]
        pos_i_emb = self.item_embeddings(pos_items)    # [batch, embed_dim]
        u_b = self.user_bias(users).squeeze(-1)         # [batch]
        pos_i_b = self.item_bias(pos_items).squeeze(-1) # [batch]

        pos_scores = torch.sum(u_emb * pos_i_emb, dim=1) + u_b + pos_i_b

        if neg_items is None:
            return pos_scores

        neg_i_emb = self.item_embeddings(neg_items)    # [batch, embed_dim]
        neg_i_b = self.item_bias(neg_items).squeeze(-1) # [batch]

        neg_scores = torch.sum(u_emb * neg_i_emb, dim=1) + u_b + neg_i_b

        # BPR Loss: -ln sigmoid(pos_scores - neg_scores)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Regularization Loss
        reg_loss = 1e-4 * (u_emb.norm(2).pow(2) + pos_i_emb.norm(2).pow(2) + neg_i_emb.norm(2).pow(2))
        return bpr_loss + reg_loss
