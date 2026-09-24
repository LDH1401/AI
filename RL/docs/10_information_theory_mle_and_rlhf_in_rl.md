# Chương 10: Nền tảng Toán học: Maximum Entropy MLE, RLHF & CHỨNG MINH TOÁN HỌC Direct Preference Optimization (DPO)

---

## 1. Kết nối Toán học: Maximum Entropy RL & Phân phối Energy-Based (MLE)

Trong Maximum Entropy RL, ta giải bài toán tối ưu hóa có ràng buộc Entropy:

$$\max_{\pi} \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t \left( R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right) \right]$$

### 🔴 CHỨNG MINH: Policy Tối ưu là Phân phối Softmax / Boltzmann:

Bằng cách áp dụng **Nhân tử Lagrange** và tối ưu hóa hàm mục tiêu theo $\pi(a \mid s)$ sao cho $\sum_a \pi(a \mid s) = 1$:

$$\mathcal{L}(\pi, \lambda) = \sum_{a} \pi(a \mid s) Q^*(s, a) + \alpha \left( -\sum_{a} \pi(a \mid s) \log \pi(a \mid s) \right) + \lambda \left( 1 - \sum_{a} \pi(a \mid s) \right)$$

Tính đạo hàm $\frac{\partial \mathcal{L}}{\partial \pi(a \mid s)} = 0$:

$$Q^*(s, a) - \alpha \left( \log \pi(a \mid s) + 1 \right) - \lambda = 0$$

$$\log \pi(a \mid s) = \frac{Q^*(s, a)}{\alpha} - 1 - \frac{\lambda}{\alpha}$$

$$\pi^*(a \mid s) = \frac{\exp\left( \frac{1}{\alpha} Q^*(s, a) \right)}{\sum_{a'} \exp\left( \frac{1}{\alpha} Q^*(s, a') \right)} = \frac{\exp\left( \frac{1}{\alpha} Q^*(s, a) \right)}{Z(s)} \quad \blacksquare$$

---

## 2. CHỨNG MINH TOÁN HỌC CHI TIẾT: Direct Preference Optimization (DPO - Stanford 2023)

> [!IMPORTANT]
> **Định lý (Direct Preference Optimization - Rafailov et al., NIPS 2023):**
> Bài toán tối ưu hóa RLHF có phạt KL Divergence có thể được giải **trực tiếp** bằng cách biến đổi toán học loại bỏ hoàn toàn mạng Reward Model $R_\psi$ và giai đoạn huấn luyện PPO phức tạp!

---

### 🔴 CHỨNG MINH TỪ ĐẦU (Derivation from Scratch):

#### Bước 1: Mục tiêu Tối ưu hóa RLHF với KL Penalty:
Cho trước prompt $x$ và câu trả lời $y$. Ta cần tối ưu hóa Policy $\pi_\theta(y \mid x)$ theo Reward Model $r(x, y)$ có sẵn:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r(x, y) \right] - \beta \, D_{\text{KL}}\left( \pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x) \right)$$

Khai triển công thức Phân kỳ KL:

$$= \max_{\pi_\theta} \mathbb{E}_{x, y} \left[ r(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] = \max_{\pi_\theta} \beta \mathbb{E}_{x, y} \left[ \log \frac{\exp\left( \frac{1}{\beta} r(x, y) \right) \pi_{\text{ref}}(y \mid x)}{\pi_\theta(y \mid x)} \right]$$

#### Bước 2: Định nghĩa Phân phối Chuẩn hóa $Z(x)$:
Đặt $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$. Ta định nghĩa phân phối tối ưu $\pi^*(y \mid x)$:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

Thế $\pi^*(y \mid x)$ vào hàm mục tiêu:

$$\max_{\pi_\theta} \beta \mathbb{E}_{x} \left[ \log Z(x) - D_{\text{KL}}\left( \pi_\theta(y \mid x) \parallel \pi^*(y \mid x) \right) \right]$$

Do $\log Z(x)$ không phụ thuộc $\theta$, hàm mục tiêu đạt giá trị Cực đại khi $D_{\text{KL}}(\pi_\theta \parallel \pi^*) = 0 \implies \pi_\theta = \pi^*$.

#### Bước 3: Biểu diễn Ngược Reward $r(x, y)$ theo Policy $\pi_\theta$:
Từ $\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$, lấy logarit 2 vế:

$$\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

$$\implies r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

#### Bước 4: Thế vào Loss Bradley-Terry Preference:
Mô hình Bradley-Terry định nghĩa xác suất người dùng thích câu trả lời $y_w$ hơn $y_l$:

$$P(y_w \succ y_l \mid x) = \sigma\left( r(x, y_w) - r(x, y_l) \right)$$

Thế biểu diễn $r(x, y)$ ở Bước 3 vào (lưu ý đại lượng $\beta \log Z(x)$ bị triệt tiêu vì xuất hiện ở cả 2 vế!):

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

#### Kết quả: Hàm Mất mát DPO Loss ($\mathcal{L}_{\text{DPO}}$):

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right] \quad \blacksquare$$

---

## 3. Ưu điểm Đột phá của DPO trong Thực tế

1. **Không cần Reward Model:** Loại bỏ hoàn toàn công đoạn huấn luyện mạng nơ-ron Reward riêng.
2. **Không cần PPO Sampling:** Không cần chạy PPO actor-critic loop đắt đỏ.
3. **Huấn luyện Giám sát Ổn định:** DPO Loss được tính trực tiếp bằng phép huấn luyện có giám sát (Supervised Cross-Entropy-like Loss) trên cặp dữ liệu $(y_w, y_l)$ với độ ổn định tuyệt đối!
