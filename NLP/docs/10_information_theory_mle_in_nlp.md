# Chương 10: Nền tảng Toán học: Lý thuyết Thông tin & MLE trong NLP (Information Theory & Perplexity)

---

Bản chất của việc huấn luyện các mô hình ngôn ngữ (từ N-gram, RNN, BERT đến các Mô hình Ngôn ngữ Lớn LLMs) được xây dựng dựa trên nền tảng của **Lý thuyết Thông tin (Information Theory)** và **Ước lượng Hợp lý Cực đại (Maximum Likelihood Estimation - MLE)**.

---

## 1. Khung Phân phối Xác suất của Mô hình Ngôn ngữ

Một văn bản $W = (w_1, w_2, \dots, w_N)$ gồm $N$ từ. Theo **Quy tắc Chuỗi Xác suất (Chain Rule of Probability)**, xác suất đồng thời xuất hiện của câu văn đó được phân tích thành:

$$P(w_1, w_2, \dots, w_N) = \prod_{i=1}^{N} P(w_i \mid w_1, w_2, \dots, w_{i-1})$$

Mô hình Ngôn ngữ (Language Model) có nhiệm vụ dự đoán phân phối xác suất của từ tiếp theo $w_i$ dựa trên chuỗi lịch sử quá khứ $w_{<i}$.

---

## 2. NLL, Cross-Entropy Loss & Perplexity (PPL)

### 2.1. Tối thiểu hóa NLL chính là Cross-Entropy Loss
Khi huấn luyện mô hình ngôn ngữ trên tập văn bản thực tế $\mathcal{D}$, hàm mất mát trung bình Negative Log-Likelihood (NLL) gán cho các từ đúng trong tập dữ liệu là:

$$\mathcal{L}_{\text{NLL}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P_\theta(w_i \mid w_{<i})$$

Đây chính là **Cross-Entropy $H(p_{\text{data}}, P_\theta)$** giữa phân phối dữ liệu ngôn ngữ thật $p_{\text{data}}$ và phân phối do mạng nơ-ron dự đoán $P_\theta$.

---

### 2.2. Perplexity (PPL - Độ phức tạp Ngôn ngữ)

**Perplexity (PPL)** là chỉ số số một để đánh giá chất lượng của một mô hình ngôn ngữ (giá trị PPL càng thấp thể hiện mô hình dự đoán càng giỏi và ít bị "bất ngờ").

#### Mối liên hệ Toán học Trực tiếp giữa Perplexity và Cross-Entropy Loss:

$$\text{Perplexity}(P_\theta) = 2^{H(p_{\text{data}}, P_\theta)} = 2^{\mathcal{L}_{\text{NLL}}} = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \ln P_\theta(w_i \mid w_{<i}) \right)$$

```
  Loss (NLL Cross-Entropy) giảm  --->  Perplexity (PPL) giảm  --->  Mô hình hiểu ngôn ngữ giỏi hơn!
```

#### Ý nghĩa Trực quan của Perplexity:
Nếu một mô hình ngôn ngữ đạt điểm $\text{Perplexity} = K$ (ví dụ $K = 10$), điều đó có nghĩa là tại mỗi bước sinh từ tiếp theo, mô hình bị phân vân tương đương với việc chọn ngẫu nhiên giữa **$K$ lựa chọn từ có khả năng xuất hiện như nhau**.
- Một mô hình đoán bừa trên từ điển 50,000 từ sẽ có $\text{PPL} = 50,000$.
- Mô hình LLM SOTA hiện đại đạt điểm $\text{PPL} \approx 8 - 15$ trên tập test!

---

## 3. Tối ưu hóa NLL buộc Mô hình phải Học Quy luật Ngôn ngữ Tự nhiên

Giả sử phân phối ngôn ngữ khách quan tự nhiên của con người là $p_{\text{data}}(w)$, và phân phối do mạng nơ-ron sinh ra là $P_\theta(w)$.

Ta có mối quan hệ giữa Cross-Entropy ($H$), Entropy nội tại của ngôn ngữ ($H(p_{\text{data}})$) và **Phân kỳ Kullback-Leibler ($D_{\text{KL}}$)**:

$$H(p_{\text{data}}, P_\theta) = H(p_{\text{data}}) + D_{\text{KL}}(p_{\text{data}} \parallel P_\theta)$$

Trong đó $H(p_{\text{data}})$ là Entropy tự nhiên cố định của ngôn ngữ loài người. Thuật toán Gradient Descent tối ưu hóa hàm loss $\mathcal{L}_{\text{NLL}}(\theta)$:

$$\arg\min_\theta \mathcal{L}_{\text{NLL}}(\theta) \iff \arg\min_\theta D_{\text{KL}}(p_{\text{data}} \parallel P_\theta)$$

> [!IMPORTANT]
> **Bất đẳng thức Gibbs:** Phân kỳ KL $D_{\text{KL}} \ge 0$, và đạt giá trị cực tiểu bằng $0$ khi và chỉ khi:
> $$P_\theta(w \mid w_{<i}) = p_{\text{data}}(w \mid w_{<i}) \quad \text{hầu khắp nơi}$$

Việc tối thiểu hóa Negative Log-Likelihood ép toàn bộ các ma trận trọng số Attention và Feed-Forward trong Transformer phải uốn nắn sao cho đường cong xác suất $P_\theta$ trùng khít hoàn toàn với quy tắc ngữ pháp, cú pháp và tri thức thực tế của ngôn ngữ con người.
