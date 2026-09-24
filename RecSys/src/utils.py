"""
Module Tiện ích cho RecSys: FAISS Indexing cho Candidate Retrieval, Negative Sampling, Checkpointing.
"""

import os
from typing import List, Tuple
import numpy as np
import torch

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FAISSRetrievalIndex:
    """
    Bộ chỉ số Vector Indexing FAISS cho Candidate Retrieval siêu tốc.
    """
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = None
        if HAS_FAISS:
            # Dùng Inner Product (Cosine similarity cho normalized vectors)
            self.index = faiss.IndexFlatIP(embedding_dim)

    def build_index(self, item_embeddings: np.ndarray):
        """Build FAISS Index từ ma trận Item Embeddings [num_items, embed_dim]."""
        if self.index is not None:
            # L2 Normalize
            faiss.normalize_L2(item_embeddings)
            self.index.add(item_embeddings.astype('float32'))
            print(f"--> Built FAISS Index with {self.index.ntotal} items.")

    def search(self, user_vector: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Tìm Top-K Candidate Item Indices cho 1 User Vector."""
        if self.index is None or self.index.ntotal == 0:
            return np.array([[0.0]]), np.array([[0]])

        if user_vector.ndim == 1:
            user_vector = user_vector.reshape(1, -1)

        user_norm = user_vector.copy().astype('float32')
        faiss.normalize_L2(user_norm)
        scores, indices = self.index.search(user_norm, top_k)
        return scores, indices


def save_recsys_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, filepath: str):
    """Lưu trọng số mô hình RecSys vào file checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None
    }
    torch.save(state, filepath)
    print(f"--> Saved RecSys checkpoint to '{filepath}'")
