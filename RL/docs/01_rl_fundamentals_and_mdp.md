# Chương 1: Nền tảng Toán học Học Tăng cường: MDP, Toán học Operator & Chứng minh Sự hội tụ Bellman

---

## 1. Phát biểu Toán học Đầy đủ của Markov Decision Process (MDP)

Học Tăng cường (Reinforcement Learning - RL) được phát biểu chặt chẽ dưới dạng một **Quá trình Quyết định Markov (Markov Decision Process - MDP)** định nghĩa bởi bộ 5 thành phần $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

1. $\mathcal{S}$: Không gian trạng thái (State space), có thể là hữu hạn hoặc tập hợp liên tục $\mathbb{R}^d$.
2. $\mathcal{A}$: Không gian hành động (Action space).
3. $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$: Hàm xác suất chuyển trạng thái (State Transition Probability Kernel):
   $$\mathcal{P}(s' \mid s, a) = \mathbb{P}(S_{t+1} = s' \mid S_t = s, A_t = a)$$
4. $\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$: Hàm phần thưởng (Reward Function), trong đó phần thưởng kỳ vọng là $R(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$.
5. $\gamma \in [0, 1)$: Hệ số chiết khấu (Discount factor) đảm bảo sự hội tụ của chuỗi tổng phần thưởng vô hạn.

---

### 1.1. Giả thuyết Markov (Markov Property)

Một quá trình ngẫu nhiên thỏa mãn **Tính chất Markov** khi và chỉ khi phân phối trạng thái tương lai $S_{t+1}$ chỉ phụ thuộc duy nhất vào trạng thái $S_t$ và hành động $A_t$ hiện tại, hoàn toàn độc lập với toàn bộ lịch sử quá khứ $H_t = (S_0, A_0, \dots, S_{t-1}, A_{t-1})$:

$$\mathbb{P}(S_{t+1} = s_{t+1} \mid S_t = s_t, A_t = a_t, S_{t-1} = s_{t-1}, \dots, S_0 = s_0, A_0 = a_0) = \mathbb{P}(S_{t+1} = s_{t+1} \mid S_t = s_t, A_t = a_t)$$

---

## 2. Chuỗi Tổng Phần thưởng & Hàm Giá trị

Chiến lược (Policy) $\pi(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$ quy định phân phối xác suất chọn hành động.

Tổng phần thưởng tích lũy có chiết khấu từ thời điểm $t$ là một biến ngẫu nhiên $G_t$:

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### 2.1. Hàm Giá trị Trạng thái (State-Value Function $V^\pi(s)$):
$$V^\pi(s) = \mathbb{E}_{\pi} \left[ G_t \mid S_t = s \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s \right]$$

### 2.2. Hàm Giá trị Hành động (Action-Value Function $Q^\pi(s, a)$):
$$Q^\pi(s, a) = \mathbb{E}_{\pi} \left[ G_t \mid S_t = s, A_t = a \right] = \mathbb{E}_{\pi} \left[ R_{t+1} + \gamma \sum_{k=0}^{\infty} \gamma^k R_{t+k+2} \;\middle|\; S_t = s, A_t = a \right]$$

Mối quan hệ giữa $V^\pi(s)$ và $Q^\pi(s, a)$:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s, a)$$
$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V^\pi(s')$$

---

## 3. Toán tử Bellman (Bellman Operators)

Biểu diễn vector của $V \in \mathbb{R}^{|\mathcal{S}|}$. Ta định nghĩa **Toán tử Kỳ vọng Bellman (Bellman Expectation Operator $\mathcal{T}^\pi$)**:

$$(\mathcal{T}^\pi V)(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V(s') \right]$$

Hoặc dưới dạng ma trận: $\mathcal{T}^\pi V = R^\pi + \gamma P^\pi V$.

Định nghĩa **Toán tử Tối ưu Bellman (Bellman Optimality Operator $\mathcal{T}^*$)**:

$$(\mathcal{T}^* V)(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}(s' \mid s, a) V(s') \right]$$

---

## 4. CHỨNG MINH TOÁN HỌC: Định lý Co Bellman & Sự Hội tụ duy nhất (Banach Fixed Point Theorem)

> [!IMPORTANT]
> **Định lý (Bellman Contraction Mapping Theorem):**
> Toán tử Tối ưu Bellman $\mathcal{T}^*$ là một **$\gamma$-Contraction Mapping (Phép co với hệ số $\gamma$)** trên Không gian Banach $\mathbb{R}^{|\mathcal{S}|}$ cùng với Chuẩn cực đại (Max-norm / Infinity-norm $\|V\|_\infty = \max_{s \in \mathcal{S}} |V(s)|$).
> Nghĩa là với mọi $V_1, V_2 \in \mathbb{R}^{|\mathcal{S}|}$:
> $$\|\mathcal{T}^* V_1 - \mathcal{T}^* V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty$$

### 🔴 CHỨNG MINH CHI TIẾT:

Với bất kỳ trạng thái $s \in \mathcal{S}$:

$$(\mathcal{T}^* V_1)(s) - (\mathcal{T}^* V_2)(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V_1(s') \right] - \max_{a' \in \mathcal{A}} \left[ R(s, a') + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a') V_2(s') \right]$$

Giả sử $a^*_1 = \arg\max_a \left[ R(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V_1(s') \right]$. Ta có bất đẳng thức:

$$(\mathcal{T}^* V_1)(s) - (\mathcal{T}^* V_2)(s) \le \left[ R(s, a^*_1) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a^*_1) V_1(s') \right] - \left[ R(s, a^*_1) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a^*_1) V_2(s') \right]$$

$$= \gamma \sum_{s'} \mathcal{P}(s' \mid s, a^*_1) \left( V_1(s') - V_2(s') \right)$$

Vì $\mathcal{P}(s' \mid s, a^*_1) \ge 0$ và $\sum_{s'} \mathcal{P}(s' \mid s, a^*_1) = 1$:

$$\gamma \sum_{s'} \mathcal{P}(s' \mid s, a^*_1) \left( V_1(s') - V_2(s') \right) \le \gamma \sum_{s'} \mathcal{P}(s' \mid s, a^*_1) \|V_1 - V_2\|_\infty = \gamma \|V_1 - V_2\|_\infty$$

Tương tự, bằng cách đổi vai trò giữa $V_1$ và $V_2$, ta thu được $(\mathcal{T}^* V_2)(s) - (\mathcal{T}^* V_1)(s) \le \gamma \|V_1 - V_2\|_\infty$. Do đó:

$$|(\mathcal{T}^* V_1)(s) - (\mathcal{T}^* V_2)(s)| \le \gamma \|V_1 - V_2\|_\infty, \quad \forall s \in \mathcal{S}$$

Lấy $\max_{s}$ ở cả 2 vế:

$$\|\mathcal{T}^* V_1 - \mathcal{T}^* V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty \quad \blacksquare$$

---

### Hệ quả của Định lý Điểm Cố định Banach (Banach Fixed Point Theorem):
1. **Tồn tại Duy nhất:** Phương trình Bellman $V^* = \mathcal{T}^* V^*$ có một nghiệm duy nhất $V^*$.
2. **Sự Hội tụ của Value Iteration:** Với bất kỳ khởi tạo ban đầu $V_0$, chuỗi lặp $V_{k+1} = \mathcal{T}^* V_k$ sẽ hội tụ về $V^*$ với tốc độ hình học:
   $$\|V_k - V^*\|_\infty \le \gamma^k \|V_0 - V^*\|_\infty \to 0 \quad \text{khi } k \to \infty$$
