# Chương 5: Từ TRPO đến PPO: Hình học Thông tin, Ma trận Fisher & Thuật toán PPO (PPO Mathematical Foundations)

---

> [!IMPORTANT]
> 📌 **TỔNG QUAN & VAI TRÒ TRONG PIPELINE AI:**
> Trong RL, nếu bước cập nhật gradient quá lớn, chiến lược (Policy) mới có thể bị suy giảm hiệu năng nghiêm trọng và mô hình sụp đổ hoàn toàn mà không thể khôi phục lại được.
> **Mục tiêu của Chương 5** là giải thích **Hình học Thông tin (Information Geometry)** và **Thuật toán PPO (Proximal Policy Optimization - OpenAI)**.
> - **Đầu vào:** Trạng thái $s_t$, Hành động $a_t$, Advantage $\hat{A}_t$ và Policy cũ $\pi_{\theta_{\text{old}}}$.
> - **Xử lý:** Probability Ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ $\to$ Clipped Surrogate Objective Loss $\mathcal{L}^{\text{CLIP}}$.
> - **Đầu ra:** Bước cập nhật Policy mới cực kỳ ổn định, nằm an toàn trong vùng tin cậy (Trust Region).
> - **Vai trò:** Là thuật toán RL tiêu chuẩn công nghiệp phổ biến nhất hiện nay (sử dụng trong ChatGPT RLHF, Dota 2 OpenAI Five và Robotics)!

---

## 1. Nền tảng Hình học Thông tin (Information Geometry & Natural Policy Gradient)

Trong không gian tham số $\theta \in \mathbb{R}^d$, một bước cập nhật Gradient Descent tiêu chuẩn $\theta_{k+1} = \theta_k + \alpha \nabla J(\theta_k)$ đo khoảng cách giữa 2 bộ tham số bằng **Khoảng cách Euclide $\|\theta_{k+1} - \theta_k\|_2$**.

Tuy nhiên, khoảng cách Euclide giữa các tham số $\theta$ không hề phản ánh đúng **sự thay đổi của phân phối xác suất hành vi $\pi_\theta(a \mid s)$**! Một sự thay đổi nhỏ về $\theta$ có thể khiến phân phối $\pi_\theta$ sụt đổ hoàn toàn.

---

### 1.1. Ma trận Thông tin Fisher (Fisher Information Matrix - FIM)

Để đo khoảng cách giữa 2 phân phối xác suất $\pi_\theta$ và $\pi_{\theta + \Delta \theta}$, ta sử dụng **Phân kỳ KL (Kullback-Leibler Divergence)**:

Khai triển Taylor bậc 2 của $D_{\text{KL}}(\pi_\theta \parallel \pi_{\theta + \Delta \theta})$ quanh $\theta$:

$$D_{\text{KL}}(\pi_\theta \parallel \pi_{\theta + \Delta \theta}) \approx \frac{1}{2} \Delta \theta^T F \Delta \theta$$

Trong đó $F \in \mathbb{R}^{d \times d}$ là **Ma trận Thông tin Fisher (Fisher Information Matrix - FIM)**:

$$F = \mathbb{E}_{s \sim \rho, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \, \nabla_\theta \log \pi_\theta(a \mid s)^T \right]$$

---

### 1.2. Natural Policy Gradient (NPG - Kakade 2002)

NPG giải bài toán tối ưu hóa có ràng buộc phân kỳ KL:

$$\max_{\Delta \theta} \nabla J(\theta)^T \Delta \theta \quad \text{sao cho } \frac{1}{2} \Delta \theta^T F \Delta \theta \le \delta$$

Áp dụng nhân tử Lagrange, ta thu được bước cập nhật **Natural Policy Gradient**:

$$\Delta \theta = \sqrt{\frac{2\delta}{g^T F^{-1} g}} \, F^{-1} g, \quad \text{với } g = \nabla_\theta J(\theta)$$

---

## 2. TRPO (Trust Region Policy Optimization - Schulman et al., 2015)

TRPO mở rộng NPG bằng cách giải bài toán tối ưu hóa trên vùng tin cậy (Trust Region):

$$\max_{\theta} \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] \quad \text{sao cho } \bar{D}_{\text{KL}}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta) \le \delta$$

Để giải bài toán này, TRPO sử dụng **Thuật toán Gradient Conjugate (Conjugate Gradient Algorithm)** để tính gần đúng $F^{-1} g$ mà không cần đảo ngược ma trận Fisher $F$ trực tiếp. Tuy nhiên, TRPO rất phức tạp và tốn chi phí tính toán lớn.

---

## 3. PPO (Proximal Policy Optimization - OpenAI 2017)

PPO xấp xỉ ràng buộc vùng tin cậy của TRPO mà không cần tính toán Ma trận Fisher $F$ hay Gradient Conjugate!

Tỷ lệ xác suất $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$.

### Hàm Mục tiêu Clipped Surrogate Loss ($\mathcal{L}^{\text{CLIP}}$):

$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

#### 💡 Giải thích Dễ hiểu Từng con số & Cơ chế Cắt (Clipping Mechanism):

Giả sử hệ số cắt $\epsilon = 0.2$ (khoảng cho phép $[0.8, 1.2]$).

1. **Tỷ lệ $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$:** 
   - Đo lường mức độ thay đổi giữa Policy mới $\pi_\theta$ và Policy cũ $\pi_{\theta_{\text{old}}}$.
   - Ban đầu khi chưa cập nhật, $r_t(\theta) = 1.0$.
2. **Trường hợp 1: Hành động tốt ($\hat{A}_t > 0$, ví dụ $\hat{A}_t = +10$):**
   - Ta muốn tăng xác suất hành động này, khiến $r_t(\theta)$ tăng lên $> 1.0$.
   - Nếu $r_t(\theta)$ tăng vọt lên $1.5$ (tăng $50\%$), phép hàm `clip(1.5, 0.8, 1.2)` ép giá trị về tối đa **$1.2$**.
   - Phép lấy $\min(1.5 \times 10, 1.2 \times 10) = 12$ sẽ triệt tiêu gradient khi $r_t > 1.2$, **loại bỏ động lực khiến mô hình thay đổi quá nhanh!**
3. **Trường hợp 2: Hành động xấu ($\hat{A}_t < 0$, ví dụ $\hat{A}_t = -10$):**
   - Ta muốn giảm xác suất hành động này, khiến $r_t(\theta)$ giảm xuống $< 1.0$.
   - Nếu $r_t(\theta)$ tụt xuống $0.5$, phép hàm `clip(0.5, 0.8, 1.2)` ép giá trị về tối thiểu **$0.8$**.
   - Phép lấy $\min(0.5 \times -10, 0.8 \times -10) = \min(-5, -8) = -8$ giúp phạt mô hình nếu nó vô tình giảm xác suất hành động quá sâu.

---

## 4. CHỨNG MINH TOÁN HỌC: Generalized Advantage Estimation (GAE)

GAE định nghĩa nhà ước lượng Advantage $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ bằng tổng có trọng số exponential của các $k$-step Advantage estimators:

$$\hat{A}_t^{(1)} = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\hat{A}_t^{(2)} = \delta_t^V + \gamma \delta_{t+1}^V = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)$$

Lấy trung bình có trọng số chiết khấu $\lambda \in [0, 1]$:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1-\lambda) \sum_{l=1}^{\infty} \lambda^{l-1} \hat{A}_t^{(l)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V \quad \blacksquare$$

- Khi $\lambda = 0$: $\hat{A}_t = \delta_t^V$ (1-step TD, Variance thấp nhất, Bias cao nhất).
- Khi $\lambda = 1$: $\hat{A}_t = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)$ (Monte Carlo, Bias bằng 0, Variance cao nhất).
