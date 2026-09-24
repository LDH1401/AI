"""
Module Hỏi Đáp trên Tài liệu (Document QA & Document RAG):
1. Extractive Span QA với PhoBERT / BERT.
2. Tìm kiếm Kép Hybrid Search (BM25 + Dense Embeddings) trên văn bản tài liệu.
3. Layout-Aware Document Understanding với 2D Bounding Box Embeddings (LayoutLM Style).
"""

from typing import List, Dict, Tuple, Any
import torch
import torch.nn as nn
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


class HybridDocumentRetriever:
    """
    Bộ tìm kiếm Kép Hybrid Search kết hợp Sparse BM25 và Dense Cosine Similarity cho Document RAG.
    """
    def __init__(self, document_chunks: List[str]):
        self.chunks = document_chunks
        self.bm25 = None
        if HAS_BM25:
            tokenized_corpus = [doc.lower().split() for doc in document_chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve_bm25(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Tìm kiếm bằng thuật toán Sparse BM25."""
        if self.bm25 is None:
            return [(i, 1.0) for i in range(min(top_k, len(self.chunks)))]
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def reciprocal_rank_fusion(self, dense_ranks: List[int], bm25_ranks: List[int], k: int = 60) -> List[Tuple[int, float]]:
        """Kết hợp điểm thứ hạng RRF từ Dense và BM25 Search."""
        rrf_scores = {}
        for rank, idx in enumerate(dense_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(bm25_ranks):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
            
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_chunks


class LayoutLMSimplifiedEmbedder(nn.Module):
    """
    Minh họa 2D Spatial Positional Embedding của LayoutLM cho Visual Document QA.
    Nhận đầu vào: 2D Bounding Boxes (x0, y0, x1, y1) chuẩn hóa từ 0 đến 1000.
    """
    def __init__(self, hidden_dim: int = 768, max_position: int = 1001):
        super().__init__()
        self.x_position_embeddings = nn.Embedding(max_position, hidden_dim)
        self.y_position_embeddings = nn.Embedding(max_position, hidden_dim)
        self.h_position_embeddings = nn.Embedding(max_position, hidden_dim)
        self.w_position_embeddings = nn.Embedding(max_position, hidden_dim)

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        bbox shape: [batch_size, seq_len, 4] -> (x0, y0, x1, y1)
        Returns: 2D Spatial Layout Embedding Tensor [batch_size, seq_len, hidden_dim]
        """
        x0 = bbox[:, :, 0]
        y0 = bbox[:, :, 1]
        x1 = bbox[:, :, 2]
        y1 = bbox[:, :, 3]

        w = x1 - x0
        h = y1 - y0

        left_emb = self.x_position_embeddings(x0)
        right_emb = self.x_position_embeddings(x1)
        top_emb = self.y_position_embeddings(y0)
        bottom_emb = self.y_position_embeddings(y1)
        height_emb = self.h_position_embeddings(h)
        width_emb = self.w_position_embeddings(w)

        layout_embedding = left_emb + right_emb + top_emb + bottom_emb + height_emb + width_emb
        return layout_embedding
