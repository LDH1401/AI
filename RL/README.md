# Comprehensive Reinforcement Learning (RL) Repository & Documentation

Kho tài liệu lý thuyết toàn diện và Codebase chuẩn Production về **Học Tăng cường (Reinforcement Learning - RL)**, phủ rộng từ Nền tảng Markov Decision Process (MDP), Phương trình Bellman, Model-Free Value-Based RL (Q-Learning, **Deep Q-Networks DQN, Double DQN, Dueling DQN, Experience Replay, Target Networks**), Policy Gradient Methods (**REINFORCE**), Actor-Critic Framework (**A2C, A3C**), **Proximal Policy Optimization (PPO)**, Continuous Action Control (**DDPG, TD3, Soft Actor-Critic SAC SOTA**), Model-Based RL & World Models (**DreamerV3, MCTS**), Multi-Agent RL (**MADDPG, CTDE Framework**) đến mối liên kết toán học sâu sắc với **Lý thuyết Thông tin, Maximum Entropy RL, Boltzmann MLE & RLHF (Reward Modeling & Bradley-Terry Loss trong LLMs)**.

---

## 📚 Chuỗi Tài liệu Lý thuyết Toàn diện (10 Chương)

Tài liệu được thiết kế theo trình tự đầu cuối rõ ràng, chi tiết và dễ hiểu trong thư mục `docs/`:

1. [Chương 1: Nền tảng Học Tăng cường & Markov Decision Process](docs/01_rl_fundamentals_and_mdp.md)
   - Khái niệm RL, Agent-Environment Interaction Loop, State $s_t$, Action $a_t$, Reward $r_t$, Discount Factor $\gamma$.
   - **Markov Decision Process (MDP):** Tính chất Markov $P(s_{t+1} \mid s_t, a_t)$.
   - Policy $\pi(a \mid s)$, Value Function $V^\pi(s)$, Action-Value Function $Q^\pi(s, a)$.
   - **Phương trình Bellman (Bellman Expectation & Optimality Equations).**

2. [Chương 2: Phương pháp Dựa trên Giá trị & Deep Q-Networks](docs/02_value_based_methods_q_learning_dqn.md)
   - Model-Free Value-Based RL, TD Learning & TD Error.
   - **Q-Learning (Watkins 1989):** Off-policy TD control $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max Q(s', a') - Q(s, a)]$.
   - **Deep Q-Networks (DQN - DeepMind 2015):** Experience Replay Buffer & Target Network $Q_{\theta^-}$.
   - **Double DQN:** Triệt tiêu Overestimation Bias.
   - **Dueling DQN:** Tách biệt State-Value $V(s)$ và Advantage $A(s, a)$.

3. [Chương 3: Phương pháp Gradient Chiến lược & Thuật toán REINFORCE](docs/03_policy_gradient_methods_reinforce.md)
   - Policy-Based RL & Maximization Objective $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$.
   - **Định lý Policy Gradient (Policy Gradient Theorem):**
     $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) Q^{\pi_\theta}(s_t, a_t) \right]$$
   - **REINFORCE Algorithm (Monte Carlo Policy Gradient)** & Kỹ thuật giảm variance bằng Baseline.

4. [Chương 4: Kiến trúc Actor-Critic (Actor-Critic Framework, A2C & A3C)](docs/04_actor_critic_architectures_a2c_a3c.md)
   - Dung hợp Policy-Based (Actor) và Value-Based (Critic).
   - **Advantage Function:** $A(s, a) = Q(s, a) - V(s) \approx r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$.
   - **A2C (Synchronous)** & **A3C (Asynchronous Advantage Actor-Critic - DeepMind 2016)**.

5. [Chương 5: Thuật toán Tối ưu hóa Chiến lược Tiệm cận PPO](docs/05_proximal_policy_optimization_ppo.md)
   - **PPO (Proximal Policy Optimization - OpenAI 2017):** Thuật toán RL chuẩn mực công nghiệp.
   - Probability Ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$.
   - **Clipped Surrogate Objective Function:**
     $$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
   - Generalized Advantage Estimation (GAE $\gamma, \lambda$).

6. [Chương 6: RL trên Không gian Hành động Liên tục & Thuật toán DDPG / TD3](docs/06_continuous_action_spaces_ddpg_td3.md)
   - RL trên Continuous Action Spaces $a \in \mathbb{R}^d$.
   - **DDPG (Deep Deterministic Policy Gradient - DeepMind):** Deterministic Policy $a = \mu_\theta(s)$ & Polyak Soft Update.
   - **TD3 (Twin Delayed DDPG - Fujimoto et al.):** Clipped Double Q-Learning, Delayed Policy Updates, Target Policy Smoothing.

7. [Chương 7: Thuật toán Soft Actor-Critic & Maximum Entropy RL](docs/07_soft_actor_critic_sac_maximum_entropy_rl.md)
   - **Maximum Entropy RL Framework:** Tối đại hóa phần thưởng và Entropy $\mathcal{H}(\pi(\cdot \mid s_t))$.
   - **Soft Actor-Critic (SAC - Haarnoja et al., UC Berkeley 2018):** Continuous Control Off-policy SOTA.
   - Reparameterization Trick $a_t = \tanh(\mu + \sigma \odot \epsilon)$ & Automatic Temperature Tuning $\alpha$.

8. [Chương 8: Học Tăng cường Dựa trên Mô hình Môi trường (Model-Based RL & World Models)](docs/08_model_based_reinforcement_learning.md)
   - Model-Free vs Model-Based RL (MBRL).
   - World Models $\hat{s}_{t+1}, \hat{r}_t = f_\theta(s_t, a_t)$.
   - Monte Carlo Tree Search (**MCTS AlphaGo/AlphaZero**) & **DreamerV3** (Huấn luyện Agent trong Latent Imagination).

9. [Chương 9: Học Tăng cường Đa Agent (Multi-Agent RL - MARL)](docs/09_multi_agent_reinforcement_learning_marl.md)
   - Thử thách Non-Stationarity trong MARL.
   - **CTDE Framework:** Centralized Training with Decentralized Execution.
   - **MADDPG (Multi-Agent DDPG - OpenAI 2017)** & **QMIX**.

10. [Chương 10: Nền tảng Toán học: Maximum Entropy MLE & RLHF](docs/10_information_theory_mle_and_rlhf_in_rl.md)
    - Kết nối Maximum Entropy RL với Phân phối Boltzmann / Gibbs MLE $\pi^*(a \mid s) \propto \exp(Q^*(s, a) / \alpha)$.
    - **RLHF (Reinforcement Learning from Human Feedback trong LLMs):**
      - Bradley-Terry Preference Loss cho Reward Model $R_\psi$:
        $$\mathcal{L}_{\text{Reward}}(\psi) = -\mathbb{E} \left[ \log \sigma (R_\psi(y_w) - R_\psi(y_l)) \right]$$
      - PPO Optimization với KL-Divergence Penalty chống Reward Hacking.

---

## 🛠️ Cấu trúc Kho lưu trữ (Project Repository Structure)

```
g:/Speech Processing/RL/
├── .gemini/                       # Lưu trữ context dự án & kế hoạch thực hiện
│   ├── implementation_plan.md
│   └── todo.md
├── docs/                          # Chuỗi 10 chương tài liệu lý thuyết RL
│   ├── 01_rl_fundamentals_and_mdp.md
│   ├── 02_value_based_methods_q_learning_dqn.md
│   ├── 03_policy_gradient_methods_reinforce.md
│   ├── 04_actor_critic_architectures_a2c_a3c.md
│   ├── 05_proximal_policy_optimization_ppo.md
│   ├── 06_continuous_action_spaces_ddpg_td3.md
│   ├── 07_soft_actor_critic_sac_maximum_entropy_rl.md
│   ├── 08_model_based_reinforcement_learning.md
│   ├── 09_multi_agent_reinforcement_learning_marl.md
│   └── 10_information_theory_mle_and_rlhf_in_rl.md
├── src/                           # Codebase Python chuẩn Production
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dqn_agent.py            # PyTorch DQN / Double DQN / Dueling DQN Agent
│   │   ├── ppo_agent.py            # PyTorch PPO Agent với GAE
│   │   └── sac_agent.py            # PyTorch SAC Agent cho Continuous Control
│   ├── __init__.py
│   ├── config.py                 # Dataclasses cấu hình RL Agents & Environments
│   ├── replay_buffer.py          # Replay Buffer chuẩn PyTorch
│   ├── metrics.py                # Module tính toán Episode Rewards & Moving Averages
│   ├── utils.py                  # Tiện ích Seed setting & Checkpointing
│   ├── train.py                  # Script huấn luyện PyTorch RL Agents trên Gymnasium
│   └── infer.py                  # Script Inference cho Agent Rollout Evaluation
├── requirements.txt               # Danh sách thư viện Python cần thiết
└── README.md                      # Hướng dẫn chính của dự án RL
```

---

## 💻 Hướng dẫn Cài đặt & Sử dụng (Dành cho Người dùng thực thi trên Terminal/CMD)

> [!IMPORTANT]
> Theo Quy tắc 2, bạn hãy tự mở Terminal (CMD/PowerShell) tại thư mục `g:\Speech Processing\RL` để thực hiện các lệnh dưới đây.

### Bước 1: Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy script huấn luyện DQN Agent trên môi trường Gymnasium CartPole-v1

```bash
python -m src.train
```

### Bước 3: Chạy script thử nghiệm Rollout của RL Agent

```bash
python -m src.infer --env_name CartPole-v1 --num_episodes 5
```
