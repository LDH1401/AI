"""
Module Triển khai Thuật toán GRPO (Group Relative Policy Optimization - DeepSeek-R1) trong PyTorch.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRPOLossCalculator(nn.Module):
    """
    Tính toán hàm mất mát GRPO Loss cho bài toán huấn luyện LLM Reasoning mà KHÔNG CẦN mạng Critic.
    """
    def __init__(self, clip_eps: float = 0.2, beta_kl: float = 0.04):
        super().__init__()
        self.clip_eps = clip_eps
        self.beta_kl = beta_kl

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        rewards shape: [group_size] (Danh sách phần thưởng r_1 .. r_G của G câu trả lời).
        Returns: Normalized Advantages A_i shape [group_size]
        """
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r
        return advantages

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        log_probs: Log probabilities từ policy mới [group_size, seq_len]
        old_log_probs: Log probabilities từ policy cũ [group_size, seq_len]
        ref_log_probs: Log probabilities từ reference SFT model [group_size, seq_len]
        rewards: Phần thưởng đo điểm đúng/sai [group_size]
        """
        group_size, seq_len = log_probs.shape
        advantages = self.compute_group_advantages(rewards).unsqueeze(-1)  # [group_size, 1]

        # Ratio r_i(theta)
        ratios = torch.exp(log_probs - old_log_probs)

        # Clipped Loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL Divergence Penalty
        kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1.0
        kl_penalty = self.beta_kl * kl_div.mean()

        total_grpo_loss = policy_loss + kl_penalty
        return total_grpo_loss
