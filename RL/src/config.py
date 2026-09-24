"""
Module Cấu hình (Configuration) cho Hệ thống Reinforcement Learning (RL).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RLConfig:
    """Cấu hình Không gian Trạng thái, Hành động & Môi trường Gymnasium."""
    env_name: str = "CartPole-v1"  # Hoặc "Pendulum-v1" cho Continuous Control
    state_dim: int = 4
    action_dim: int = 2
    is_continuous: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95  # Dành cho PPO


@dataclass
class AgentConfig:
    """Cấu hình Thuật toán RL Agent (DQN / PPO / SAC)."""
    agent_type: str = "dqn"  # "dqn", "double_dqn", "ppo", "sac"
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    buffer_capacity: int = 50000
    batch_size: int = 64
    tau: float = 0.005       # Soft update parameter
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    ppo_clip_eps: float = 0.2
    ppo_epochs: int = 10


@dataclass
class TrainConfig:
    """Cấu hình Huấn luyện RL Environment."""
    max_episodes: int = 500
    max_steps_per_episode: int = 500
    target_update_freq: int = 100
    checkpoint_dir: str = "./checkpoints_rl"
    seed: int = 42
    device: str = "cuda"  # Hoặc "cpu"
