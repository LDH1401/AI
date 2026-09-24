"""
Mô hình & Thuật toán Deep Q-Network (DQN, Double DQN & Dueling DQN) trong PyTorch.
"""

from typing import Tuple, List, Any
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """Mạng Nơ-ron Xấp xỉ Q-Function Q(s, a)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, is_dueling: bool = False):
        super().__init__()
        self.is_dueling = is_dueling
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if is_dueling:
            self.value_stream = nn.Linear(hidden_dim, 1)
            self.advantage_stream = nn.Linear(hidden_dim, action_dim)
        else:
            self.q_stream = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feat = self.feature_layer(state)
        
        if self.is_dueling:
            v = self.value_stream(feat)
            adv = self.advantage_stream(feat)
            q_val = v + (adv - adv.mean(dim=-1, keepdim=True))
            return q_val
        else:
            return self.q_stream(feat)


class DQNAgent:
    """
    DQN Agent hỗ trợ Standard DQN, Double DQN và Dueling DQN.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        is_double: bool = True,
        is_dueling: bool = True,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.is_double = is_double
        self.device = device

        # Online Network & Target Network
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim, is_dueling).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim, is_dueling).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, epsilon: float = 0.05) -> int:
        """Epsilon-Greedy Strategy chọn hành động."""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            action = torch.argmax(q_values, dim=1).item()
        return action

    def update(self, batch: Tuple[torch.Tensor, ...]) -> float:
        """Cập nhật trọng số Policy Network bằng TD Loss."""
        states, actions, rewards, next_states, dones = [b.to(self.device) for b in batch]

        # Current Q-values
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.long())

        with torch.no_grad():
            if self.is_double:
                # Double DQN: Chọn action bằng Online net, lấy Q bằng Target net
                next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(-1)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # Standard DQN: Max Q trên Target net
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(-1)

            expected_state_action_values = rewards + (self.gamma * next_q_values * (1.0 - dones))

        loss = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Hard Copy trọng số từ Policy Net sang Target Net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
