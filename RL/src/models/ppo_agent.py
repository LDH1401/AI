"""
Mô hình & Thuật toán Proximal Policy Optimization (PPO) trong PyTorch.
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCriticNetwork(nn.Module):
    """Mạng Nơ-ron Actor-Critic chung cho PPO."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Shared Feature Backbone
        self.actor_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits = self.actor_head(state)
        state_value = self.critic_head(state)
        return action_logits, state_value


class PPOAgent:
    """
    PPO Agent với Clipped Surrogate Objective Loss và GAE.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 10,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.device = device

        self.ac_net = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.ac_net.parameters(), lr=lr)

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Chọn hành động từ Policy distribution."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.ac_net(state_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tính toán Generalized Advantage Estimation (GAE)."""
        advantages = []
        gae = 0.0
        values_extended = values + [next_value]

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values_extended[step + 1] * (1.0 - float(dones[step])) - values_extended[step]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[step])) * gae
            advantages.insert(0, gae)

        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = advantages_t + torch.tensor(values, dtype=torch.float32, device=self.device)
        # Normalize Advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        return advantages_t, returns_t

    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        old_log_probs: List[float],
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> float:
        """Cập nhật PPO Clipped Surrogate Loss qua nhiều PPO Epochs."""
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)

        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            logits, values = self.ac_net(states_t)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            # Probability Ratio r_t(theta)
            ratios = torch.exp(new_log_probs - old_log_probs_t)

            # Clipped Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), returns)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.ppo_epochs
