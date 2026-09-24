"""
Mô hình Sequential RecSys dựa trên Transformer Causal Attention: SASRec trong PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRecModel(nn.Module):
    """
    Triển khai mô hình SASRec (Self-Attention Sequential Recommendation - Kang & McAuley, ICDM 2018).
    """
    def __init__(self, num_items: int, max_seq_length: int = 20, hidden_dim: int = 64, num_heads: int = 2, num_blocks: int = 2):
        super().__init__()
        self.num_items = num_items
        self.max_seq_length = max_seq_length

        self.item_embeddings = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_embeddings = nn.Embedding(max_seq_length, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.2,
                batch_first=True
            ) for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(p=0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_seqs: torch.Tensor) -> torch.Tensor:
        """
        input_seqs shape: [batch_size, seq_len]
        Returns: Sequential User State Representations [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = input_seqs.size()

        # Positional Indices
        positions = torch.arange(seq_len, device=input_seqs.device).unsqueeze(0).expand(batch_size, -1)

        # Item Embeddings + Positional Embeddings
        seq_emb = self.item_embeddings(input_seqs) + self.pos_embeddings(positions)
        seq_emb = self.dropout(seq_emb)

        # Causal Attention Mask (Tam giác dưới)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_seqs.device), diagonal=1).bool()

        # Transformer Encoder Blocks
        for block in self.transformer_blocks:
            seq_emb = block(seq_emb, src_mask=causal_mask)

        seq_emb = self.layer_norm(seq_emb)
        return seq_emb  # [batch_size, seq_len, hidden_dim]
