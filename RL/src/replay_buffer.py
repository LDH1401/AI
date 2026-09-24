"""
Module Replay Buffer (Bộ đệm Trải nghiệm) cho Off-Policy RL Agents (DQN / SAC).
"""

from typing import Tuple, List
import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    """
    Replay Buffer chuẩn PyTorch lưu trữ các bộ chuyển đổi (State, Action, Reward, Next_State, Done).
    """
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: Any, reward: float, next_state: np.ndarray, done: bool):
        """Thêm 1 Transition vào bộ đệm."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lấy mẫu ngẫu nhiên không liên tục 1 Mini-Batch.
        Returns: Tensors (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long if isinstance(actions[0], (int, np.integer)) else torch.float32)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)
