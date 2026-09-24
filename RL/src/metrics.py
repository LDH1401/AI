"""
Module Tính toán Đánh giá Metrics cho Học Tăng cường (Episode Rewards, Moving Averages, Entropy).
"""

from typing import List, Dict
import numpy as np


class RLMetricsCalculator:
    """Class tập hợp các hàm tính toán chỉ số đánh giá cho RL Agents."""

    @staticmethod
    def compute_moving_average(rewards: List[float], window_size: int = 20) -> List[float]:
        """Tính trung bình trượt Moving Average Rewards."""
        if len(rewards) == 0:
            return []
        moving_avg = []
        for i in range(len(rewards)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(float(np.mean(rewards[start_idx:i+1])))
        return moving_avg

    @staticmethod
    def compute_summary_stats(episode_rewards: List[float]) -> Dict[str, float]:
        """Thống kê tổng quan phần thưởng tích lũy."""
        if not episode_rewards:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}

        return {
            "Mean_Reward": float(np.mean(episode_rewards)),
            "Std_Reward": float(np.std(episode_rewards)),
            "Max_Reward": float(np.max(episode_rewards)),
            "Min_Reward": float(np.min(episode_rewards))
        }
