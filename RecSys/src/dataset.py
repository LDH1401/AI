"""
Module Dataset & DataLoader cho RecSys trong PyTorch (BPR Triplet & Sequential Sequence Dataset).
"""

from typing import List, Tuple, Dict, Any
import random
import torch
from torch.utils.data import Dataset
import numpy as np


class BPRUserItemDataset(Dataset):
    """
    Dataset chuẩn PyTorch cho BPR (Bayesian Personalized Ranking).
    Trả về bộ ba (user_id, pos_item_id, neg_item_id).
    """
    def __init__(self, user_item_pairs: List[Tuple[int, int]], num_items: int):
        self.user_item_pairs = user_item_pairs
        self.num_items = num_items
        
        # Xây dựng tập hợp các items đã tương tác cho từng user
        self.user_pos_items = {}
        for u, i in user_item_pairs:
            if u not in self.user_pos_items:
                self.user_pos_items[u] = set()
            self.user_pos_items[u].add(i)

    def __len__(self) -> int:
        return len(self.user_item_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user, pos_item = self.user_item_pairs[idx]
        
        # Random Negative Sampling một item mà user chưa từng tương tác
        neg_item = random.randint(0, self.num_items - 1)
        while neg_item in self.user_pos_items[user]:
            neg_item = random.randint(0, self.num_items - 1)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long)
        )


class SequentialRecDataset(Dataset):
    """
    Dataset cho Sequential Recommendation (SASRec).
    Nhận chuỗi tương tác lịch sử [i_1, i_2, ..., i_t] và trả về Padding Sequence.
    """
    def __init__(self, user_sequences: List[List[int]], max_len: int = 20):
        self.user_sequences = user_sequences
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.user_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.user_sequences[idx]
        
        if len(seq) > self.max_len:
            input_seq = seq[-self.max_len-1:-1]
            target_seq = seq[-self.max_len:]
        else:
            input_seq = seq[:-1]
            target_seq = seq[1:]

        # Padding zeros cho input_seq và target_seq
        pad_len = self.max_len - len(input_seq)
        padded_input = [0] * pad_len + input_seq
        padded_target = [0] * pad_len + target_seq

        return (
            torch.tensor(padded_input, dtype=torch.long),
            torch.tensor(padded_target, dtype=torch.long)
        )
