"""
Script Dự đoán & Đánh giá RL Agent (Inference & Rollout Script).
"""

import argparse
import os
import torch
import numpy as np

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from src.models.dqn_agent import DQNAgent


def evaluate_agent(env_name: str = "CartPole-v1", num_episodes: int = 5, model_path: str = None):
    """
    Chạy thử nghiệm Rollout của RL Agent trong môi trường Gymnasium.
    """
    if not HAS_GYM:
        print("[Warning] Gymnasium không được cài đặt. Không thể chạy Rollout.")
        return

    env = gym.make(env_name, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    if model_path and os.path.exists(model_path):
        print(f"--> Loading trained weights from '{model_path}'...")
        checkpoint = torch.load(model_path, map_location="cpu")
        agent.policy_net.load_state_dict(checkpoint["model_state_dict"])

    total_rewards = []
    print(f"--> Evaluating RL Agent on '{env_name}' for {num_episodes} episodes...")
    
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset(seed=ep * 10)
        done = False
        ep_reward = 0.0

        while not done:
            # Exploitation (epsilon = 0.0)
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward

        total_rewards.append(ep_reward)
        print(f"  Episode {ep:02d}: Total Reward = {ep_reward:.1f}")

    env.close()
    print(f"--> Average Reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Agent Rollout Inference Pipeline")
    parser.add_argument("--env_name", type=str, default="CartPole-v1", help="Tên môi trường Gymnasium")
    parser.add_argument("--num_episodes", type=int, default=5, help="Số lượng episodes thử nghiệm")
    parser.add_argument("--model_path", type=str, default=None, help="Đường dẫn checkpoint file .pt")

    args = parser.parse_args()
    evaluate_agent(args.env_name, args.num_episodes, args.model_path)
