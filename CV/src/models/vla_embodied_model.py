"""
Module Tokenizer & Controller cho Mô hình Vision-Language-Action (VLA Embodied AI).
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn


class RobotActionTokenizer:
    """
    Bộ chuyển đổi (Tokenizer/Detokenizer) giữa Vector Hành động Robot số thực [dx, dy, dz, droll, dpitch, dyaw, gripper]
    và các Action Tokens đặc biệt (0 - 256) cho VLA LLM.
    """
    def __init__(self, num_bins: int = 256, min_action: float = -1.0, max_action: float = 1.0):
        self.num_bins = num_bins
        self.min_action = min_action
        self.max_action = max_action
        self.bins = np.linspace(min_action, max_action, num_bins)

    def actions_to_tokens(self, action_vector: np.ndarray) -> List[int]:
        """Chuyển đổi Vector hành động số thực [7] sang danh sách Token IDs [7]."""
        clipped_actions = np.clip(action_vector, self.min_action, self.max_action)
        tokens = np.digitize(clipped_actions, self.bins) - 1
        return [int(t) for t in np.clip(tokens, 0, self.num_bins - 1)]

    def tokens_to_actions(self, token_ids: List[int]) -> np.ndarray:
        """Giải mã danh sách Token IDs sang Vector hành động số thực [7]."""
        token_array = np.clip(np.array(token_ids), 0, self.num_bins - 1)
        action_vector = self.bins[token_array]
        return action_vector


class VLAActionHead(nn.Module):
    """
    Action Head tự thiết lập (Custom PyTorch Setup): Chiếu Hidden Vector từ VLM sang 7 giá trị hành động Robot.
    """
    def __init__(self, hidden_dim: int = 768, action_dim: int = 7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Chuẩn hóa về [-1.0, 1.0]
        )

    def forward(self, vlm_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        vlm_hidden_states shape: [batch_size, hidden_dim]
        Returns: Robot Action Vector [batch_size, 7]
        """
        return self.fc(vlm_hidden_states)
