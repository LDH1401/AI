"""
Script Dự đoán Gợi ý Sản phẩm (RecSys Inference Script) cho Candidate Retrieval (Two-Tower + FAISS) & Ranking.
"""

import argparse
import os
import torch
import numpy as np

from src.config import RecSysConfig, ModelConfig
from src.models.two_tower_dssm import TwoTowerDSSMModel
from src.utils import FAISSRetrievalIndex


def recommend_top_k_items(
    user_id: int,
    num_users: int = 1000,
    num_items: int = 5000,
    embedding_dim: int = 64,
    top_k: int = 10,
    device: str = "cpu"
) -> List[Tuple[int, float]]:
    """
    Gợi ý Top-K sản phẩm cho User ID dùng Two-Tower + FAISS ANN Search.
    """
    model = TwoTowerDSSMModel(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim).to(device)
    model.eval()

    with torch.no_grad():
        # 1. Trích xuất offline toàn bộ Item Embeddings v_i
        all_item_ids = torch.arange(num_items, device=device)
        item_vectors = model.item_tower(all_item_ids).cpu().numpy()

        # 2. Build FAISS ANN Index
        faiss_index = FAISSRetrievalIndex(embedding_dim=embedding_dim)
        faiss_index.build_index(item_vectors)

        # 3. Trích xuất online User Vector u_user
        user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
        user_vector = model.user_tower(user_tensor).cpu().numpy()

        # 4. Search ANN Top-K
        scores, indices = faiss_index.search(user_vector, top_k=top_k)

    results = []
    print(f"--> Top {top_k} Recommended Items for User {user_id}:")
    for idx, score in zip(indices[0], scores[0]):
        results.append((int(idx), float(score)))
        print(f"  Item ID: {idx:5d} | Similarity Score: {score:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RecSys Inference Pipeline")
    parser.add_argument("--user_id", type=int, default=42, help="User ID cần gợi ý sản phẩm")
    parser.add_argument("--top_k", type=int, default=10, help="Số lượng sản phẩm Top-K gợi ý")

    args = parser.parse_args()
    recommend_top_k_items(user_id=args.user_id, top_k=args.top_k)
