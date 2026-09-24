"""
Module Tính toán Độ đo Đánh giá Xếp hạng Top-K cho RecSys (NDCG@K, MRR@K, Recall@K, HitRate@K).
"""

from typing import List, Dict, Set
import numpy as np


class RecSysMetricsCalculator:
    """Class tập hợp các hàm tính toán chỉ số Top-K Ranking Metrics chuẩn công nghiệp."""

    @staticmethod
    def compute_recall_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
        """Tính Recall@K."""
        rec_k = recommended[:k]
        hits = len(set(rec_k) & ground_truth)
        return float(hits / len(ground_truth)) if ground_truth else 0.0

    @staticmethod
    def compute_precision_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
        """Tính Precision@K."""
        rec_k = recommended[:k]
        hits = len(set(rec_k) & ground_truth)
        return float(hits / k)

    @staticmethod
    def compute_hit_rate_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
        """Tính HitRate@K (1 nếu có ít nhất 1 hit, ngược lại 0)."""
        rec_k = recommended[:k]
        hits = len(set(rec_k) & ground_truth)
        return 1.0 if hits > 0 else 0.0

    @staticmethod
    def compute_mrr_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
        """Tính Mean Reciprocal Rank (MRR@K)."""
        rec_k = recommended[:k]
        for rank, item in enumerate(rec_k, start=1):
            if item in ground_truth:
                return float(1.0 / rank)
        return 0.0

    @staticmethod
    def compute_ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
        """Tính Normalized Discounted Cumulative Gain (NDCG@K)."""
        rec_k = recommended[:k]
        dcg = 0.0
        for rank, item in enumerate(rec_k, start=1):
            if item in ground_truth:
                dcg += 1.0 / np.log2(rank + 1)

        # Calculate Ideal DCG (IDCG)
        idcg = sum([1.0 / np.log2(r + 1) for r in range(1, min(len(ground_truth), k) + 1)])
        return float(dcg / idcg) if idcg > 0 else 0.0
