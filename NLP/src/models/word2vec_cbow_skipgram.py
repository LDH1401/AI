"""
Mô hình Word2Vec (CBOW & Skip-gram với Negative Sampling) triển khai bằng PyTorch & Gensim.
"""

from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from gensim.models import Word2Vec as GensimWord2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False


class CBOWModelPyTorch(nn.Module):
    """
    Mô hình Continuous Bag-of-Words (CBOW) trong PyTorch.
    Đầu vào: Context Words -> Dự đoán: Target Center Word.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_inputs: torch.Tensor) -> torch.Tensor:
        # context_inputs shape: [batch_size, 2 * context_window]
        embeds = self.embeddings(context_inputs)          # [batch_size, 2*k, embed_dim]
        context_mean = torch.mean(embeds, dim=1)           # [batch_size, embed_dim]
        logits = self.linear(context_mean)                 # [batch_size, vocab_size]
        return logits


class SkipGramNegativeSamplingPyTorch(nn.Module):
    """
    Mô hình Skip-gram với Negative Sampling (SGNS) trong PyTorch.
    Đầu vào: Target Word, Context Word, Negative Samples.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 100):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Khởi tạo trọng số ngẫu nhiên nhỏ
        initrange = 0.5 / embedding_dim
        self.target_embeddings.weight.data.uniform_(-initrange, initrange)
        self.context_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, target_words: torch.Tensor, context_words: torch.Tensor, neg_words: torch.Tensor) -> torch.Tensor:
        """
        target_words shape: [batch_size]
        context_words shape: [batch_size]
        neg_words shape: [batch_size, num_negatives]
        """
        target_embed = self.target_embeddings(target_words)        # [batch, embed_dim]
        context_embed = self.context_embeddings(context_words)      # [batch, embed_dim]
        neg_embed = self.context_embeddings(neg_words)              # [batch, num_neg, embed_dim]

        # 1. Positives Loss: -log sigmoid(v_target . v_context)
        pos_score = torch.sum(target_embed * context_embed, dim=1)  # [batch]
        pos_loss = -F.logsigmoid(pos_score)                         # [batch]

        # 2. Negatives Loss: -sum(log sigmoid(-v_target . v_neg))
        # target_embed shape: [batch, 1, embed_dim]
        neg_score = torch.bmm(neg_embed, target_embed.unsqueeze(2)).squeeze(2) # [batch, num_neg]
        neg_loss = -torch.sum(F.logsigmoid(-neg_score), dim=1)                 # [batch]

        return torch.mean(pos_loss + neg_loss)


def train_gensim_word2vec(sentences: List[List[str]], vector_size: int = 100, sg: int = 1) -> Any:
    """
    Huấn luyện mô hình Word2Vec cực nhanh với Gensim.
    sg=0 (CBOW), sg=1 (Skip-gram).
    """
    if not HAS_GENSIM:
        print("[Warning] Thư viện 'gensim' chưa được cài đặt.")
        return None

    print(f"--> Training Gensim Word2Vec (Vector Size={vector_size}, Model={'Skip-gram' if sg==1 else 'CBOW'})...")
    model = GensimWord2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        sg=sg
    )
    print(f"--> Trained Vocab Size: {len(model.wv)}")
    return model
