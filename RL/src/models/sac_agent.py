"""
Mô hình & Thuật toán Soft Actor-Critic (SAC SOTA) trong PyTorch cho Continuous Control.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class SoftQNetwork(nn.Module):
    """Mạng Soft Q-Network Q(s, a)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class GaussianPolicyNetwork(nn.Module):
    """Mạng Actor Gaussian Policy với Reparameterization Trick."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization Trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # rsample cho phép backprop
        action = torch.tanh(x_t)

        # Enforcing Action Bound (Tanh Correction cho Log-prob)
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action * self.max_action, log_prob


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent cho Continuous Action Control.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device

        # Soft Q-Networks (Dual Critics)
        self.q1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q1_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q2_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Actor Network
        self.policy = GaussianPolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.policy(state_t)
                action = torch.tanh(mean)
            else:
                action, _ = self.policy.sample(state_t)

        return action.cpu().numpy()[0]

    def update(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[float, float]:
        states, actions, rewards, next_states, dones = [b.to(self.device) for b in batch]

        with torch.no_grad():
            next_actions, next_log_pi = self.policy.sample(next_states)
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            q_target = rewards + (self.gamma * (1.0 - dones) * min_target_q)

        # Update Q-Networks
        q1_loss = F.mse_loss(self.q1(states, actions), q_target)
        q2_loss = F.mse_loss(self.q2(states, actions), q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update Policy Network
        pi_actions, log_pi = self.policy.sample(states)
        q1_pi = self.q1(states, pi_actions)
        q2_pi = self.q2(states, pi_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft Update Target Networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return q1_loss.item(), policy_loss.item()
