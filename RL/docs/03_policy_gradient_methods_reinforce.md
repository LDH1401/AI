# Chương 3: Phương pháp Gradient Chiến lược & CHỨNG MINH TOÁN HỌC Định lý Policy Gradient (Policy Gradient Theorem)

---

## 1. Bản chất Toán học của Policy-Based RL

Trong Policy-Based RL, ta tham số hóa chiến lược bằng một phân phối xác suất $\pi_\theta(a \mid s)$ có thể tính đạo hàm theo trọng số $\theta \in \mathbb{R}^d$.

Hàm mục tiêu $J(\theta)$ cần tối đa hóa là tổng phần thưởng kỳ vọng trên tất cả các quỹ đạo $\tau = (s_0, a_0, s_1, a_1, \dots, s_T, a_T)$:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \int P(\tau; \theta) R(\tau) \, d\tau$$

Trong đó $R(\tau) = \sum_{t=0}^{T} \gamma^t R(s_t, a_t)$ là phần thưởng của quỹ đạo $\tau$, và $P(\tau; \theta)$ là xác suất xuất hiện quỹ đạo $\tau$ dưới policy $\pi_\theta$:

$$P(\tau; \theta) = \mu(s_0) \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) \mathcal{P}(s_{t+1} \mid s_t, a_t)$$

---

## 2. CHỨNG MINH TOÁN HỌC CHI TIẾT: Định lý Policy Gradient (Policy Gradient Theorem)

> [!IMPORTANT]
> **Định lý Policy Gradient (Sutton et al., 1999):**
> Gradient của hàm mục tiêu $J(\theta)$ theo tham số $\theta$ được tính theo công thức:
> $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) Q^{\pi_\theta}(s_t, a_t) \right]$$
> **Đặc điểm quan trọng:** Công thức này KHÔNG chứa đạo hàm của hàm xác suất chuyển môi trường $\nabla_\theta \mathcal{P}(s' \mid s, a)$, giúp ta có thể tối ưu hóa policy mà KHÔNG CẦN biết mô hình môi trường (Model-Free)!

---

### 🔴 CHỨNG MINH TỪ ĐẦU (Derivation from Scratch):

#### Bước 1: Áp dụng Đạo hàm trực tiếp vào Tích phân:

$$\nabla_\theta J(\theta) = \nabla_\theta \int P(\tau; \theta) R(\tau) \, d\tau = \int \nabla_\theta P(\tau; \theta) R(\tau) \, d\tau$$

#### Bước 2: Sử dụng Mẹo Log-Derivative (Likelihood Ratio Trick):
Ta có $\nabla_\theta \log P(\tau; \theta) = \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)}$, suy ra $\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \nabla_\theta \log P(\tau; \theta)$. Thế vào biểu thức tích phân:

$$\nabla_\theta J(\theta) = \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) \, d\tau = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) R(\tau) \right]$$

#### Bước 3: Biến đổi $\nabla_\theta \log P(\tau; \theta)$:
Lấy logarit hai vế của xác suất quỹ đạo $P(\tau; \theta)$:

$$\log P(\tau; \theta) = \log \mu(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T} \log \mathcal{P}(s_{t+1} \mid s_t, a_t)$$

Tính đạo hàm theo $\theta$: Do $\mu(s_0)$ và $\mathcal{P}(s_{t+1} \mid s_t, a_t)$ không phụ thuộc vào $\theta$, các đạo hàm của chúng bằng $0$:

$$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

#### Bước 4: Thế vàoGradient của $J(\theta)$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) \left( \sum_{t'=0}^{T} \gamma^{t'} R(s_{t'}, a_{t'}) \right) \right]$$

#### Bước 5: Loại bỏ các nhân tố Quá khứ (Causality Trick):
Một hành động tại thời điểm $t$ không thể ảnh hưởng tới các phần thưởng ở quá khứ $t' < t$. Do đó, ta có thể rút gọn tổng phần thưởng từ $t' = t \dots T$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right]$$

Trong đó $G_t = \sum_{t'=t}^{T} \gamma^{t'-t} R(s_{t'}, a_{t'})$. 

Vì $\mathbb{E}[G_t \mid s_t, a_t] = Q^{\pi_\theta}(s_t, a_t)$, ta thu được công thức Định lý Policy Gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) Q^{\pi_\theta}(s_t, a_t) \right] \quad \blacksquare$$

---

## 3. CHỨNG MINH TOÁN HỌC: Bất biến của Baseline (Baseline Invariance)

Để giảm độ biến động (Variance), ta trừ một hàm Baseline $b(s_t)$ độc lập với hành động $a_t$. Ta cần chứng minh phép trừ này **không làm lệch kỳ vọng**:

$$\mathbb{E}_{a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) b(s_t) \right] = \sum_{a_t} \pi_\theta(a_t \mid s_t) \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)} b(s_t) = b(s_t) \nabla_\theta \sum_{a_t} \pi_\theta(a_t \mid s_t)$$

Vì $\sum_{a_t} \pi_\theta(a_t \mid s_t) = 1$ (tổng xác suất bằng 1), nên $\nabla_\theta (1) = 0$. 

Do đó: $\mathbb{E}_{a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) b(s_t) \right] = 0 \quad \blacksquare$.
