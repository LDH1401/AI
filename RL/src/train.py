"""
Script Huấn luyện Mô hình Reinforcement Learning PyTorch chuẩn Production (Training Pipeline).
"""

import os
import time
from typing import List, Tuple
import torch
import numpy as np

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from src.config import RLConfig, AgentConfig, TrainConfig
from src.replay_buffer import ReplayBuffer
from src.models.dqn_agent import DQNAgent
from src.models.ppo_agent import PPOAgent
from src.models.sac_agent import SACAgent
from src.metrics import RLMetricsCalculator
from src.utils import set_seed, save_rl_checkpoint


def train_dqn_agent(rl_cfg: RLConfig, agent_cfg: AgentConfig, train_cfg: TrainConfig):
    """Huấn luyện DQNAgent trên môi trường Gymnasium."""
    if not HAS_GYM:
        print("[Warning] Gymnasium không được cài đặt. Mô phỏng luồng training...")
        return

    set_seed(train_cfg.seed)
    env = gym.make(rl_cfg.env_name)
    device = train_cfg.device if torch.cuda.is_available() else "cpu"

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, hidden_dim=agent_cfg.hidden_dim, lr=agent_cfg.learning_rate, is_double=True, device=device)
    buffer = ReplayBuffer(capacity=agent_cfg.buffer_capacity)

    epsilon = agent_cfg.epsilon_start
    episode_rewards = []

    print(f"--> Starting DQN Training on Environment '{rl_cfg.env_name}'...")
    for ep in range(1, train_cfg.max_episodes + 1):
        state, _ = env.reset(seed=train_cfg.seed + ep)
        ep_reward = 0.0

        for t in range(train_cfg.max_steps_per_episode):
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if len(buffer) > agent_cfg.batch_size:
                batch = buffer.sample(agent_cfg.batch_size)
                agent.update(batch)

            if done:
                break

        epsilon = max(agent_cfg.epsilon_end, epsilon * agent_cfg.epsilon_decay)
        episode_rewards.append(ep_reward)

        if ep % train_cfg.target_update_freq == 0:
            agent.update_target_network()

        if ep % 20 == 0:
            avg_rew = np.mean(episode_rewards[-20:])
            print(f"Episode [{ep:03d}/{train_cfg.max_episodes:03d}] | Avg Reward (Last 20): {avg_rew:.2f} | Epsilon: {epsilon:.3f}")

    env.close()
    save_path = os.path.join(train_cfg.checkpoint_dir, "dqn_final.pt")
    save_rl_checkpoint(agent.policy_net, agent.optimizer, train_cfg.max_episodes, save_path)
    print("--> DQN Training Completed!")


if __name__ == "__main__":
    train_dqn_agent(RLConfig(), AgentConfig(), TrainConfig())
