# Chương 10: Nền tảng Toán học: MLE & Bộ chỉ số Đánh giá RecSys (Metrics & Information Theory)

---

## 1. Nền tảng Toán học MLE & Information Theory trong RecSys

Huấn luyện hệ thống gợi ý bản chất là bài toán **Ước lượng Hợp lý Cực đại (Maximum Likelihood Estimation - MLE)** trên ma trận phân phối xác suất tương tác giữa Người dùng và Sản phẩm.

### 1.1. Khung Phân phối Xác suất Softmax dưới góc nhìn MLE
Coi bài toán Candidate Retrieval như việc chọn 1 sản phẩm đúng $v_+$ từ từ điển gồm $N$ sản phẩm cho trước vector người dùng $\mathbf{u}$.

$$\mathcal{L}_{\text{NLL}}(\theta) = -\sum_{u \in U} \log P_\theta(v_+ \mid u) = -\sum_{u \in U} \log \frac{\exp\left( \frac{\mathbf{u}^T \mathbf{v}_+}{\tau} \right)}{\sum_{j \in \text{Items}} \exp\left( \frac{\mathbf{u}^T \mathbf{v}_j}{\tau} \right)}$$

Hàm loss này chính là **Cross-Entropy $H(p_{\text{data}}, P_\theta)$**. Việc tối thiểu hóa Negative Log-Likelihood chính là tối thiểu hóa **Phân kỳ KL $D_{\text{KL}}(p_{\text{data}} \parallel P_\theta)$** giữa phân phối hành vi người dùng thật và phân phối do mô hình RecSys dự đoán!

---

## 2. Bộ chỉ số Đánh giá RecSys Top-K (Top-K Evaluation Metrics)

Đánh giá mô hình RecSys trên tập dữ liệu kiểm thử (Test Set) đòi hỏi các chỉ số xếp hạng Top-$K$ chuyên dụng.

---

### 2.1. Precision@K & Recall@K

- **Precision@K (Độ chính xác Top-K):** Trong $K$ sản phẩm gợi ý ra, có bao nhiêu % là sản phẩm người dùng thực sự tương tác?
  $$\text{Precision}@K = \frac{|\text{Recommended Items}@K \cap \text{Relevant Items}|}{K}$$

- **Recall@K (Độ phủ Top-K):** Trong tất cả các sản phẩm mà người dùng thích, hệ thống tìm ra được bao nhiêu % nằm trong Top $K$?
  $$\text{Recall}@K = \frac{|\text{Recommended Items}@K \cap \text{Relevant Items}|}{|\text{Relevant Items}|}$$

---

### 2.2. Hit Rate@K (HR@K) & MRR@K (Mean Reciprocal Rank)

- **Hit Rate@K (HR@K):** Bằng $1$ nếu trong Top $K$ sản phẩm gợi ý có ít nhất 1 sản phẩm đúng, ngược lại bằng $0$.

- **MRR@K (Mean Reciprocal Rank):** Đánh giá vị trí xuất hiện đầu tiên của sản phẩm đúng ($\text{rank}_u$ là vị trí của sản phẩm đúng đầu tiên trong danh sách Top-$K$):
  $$\text{MRR}@K = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{1}{\text{rank}_u}$$

---

### 2.3. NDCG@K (Normalized Discounted Cumulative Gain)

NDCG@K là chỉ số đánh giá chuẩn mực nhất trong ngành công nghiệp vì nó **phạt vị trí của sản phẩm xuất hiện ở dưới sâu và thưởng lớn cho sản phẩm xuất hiện ở các vị trí đầu tiên (Top 1-3)**.

#### 1. Discounted Cumulative Gain (DCG@K):
$$\text{DCG}@K = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

Trong đó $rel_i \in \{0, 1\}$ là mức độ liên quan của sản phẩm tại vị trí thứ $i$.

#### 2. Ideal DCG (IDCG@K):
$\text{IDCG}@K$ là điểm $\text{DCG}@K$ đạt được trong trường hợp xếp hạng hoàn hảo nhất (tất cả sản phẩm đúng đều nằm ở các vị trí đầu tiên).

#### 3. Normalized DCG (NDCG@K):
$$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K} \in [0.0, 1.0]$$

---

## 3. Bảng Tổng hợp các Indicators Đánh giá RecSys

| Chỉ số Metric | Thang giá trị | Ý nghĩa thực tế | Mục tiêu tối ưu |
| :--- | :--- | :--- | :--- |
| **Recall@K** | $[0.0, 1.0]$ | Đo độ bao phủ các sản phẩm người dùng thích | Cao nhất |
| **Hit Rate@K** | $0$ hoặc $1$ | Đo xác suất có ít nhất 1 sản phẩm đúng | Cao nhất |
| **MRR@K** | $[0.0, 1.0]$ | Đo thứ hạng của sản phẩm đúng đầu tiên | Cao nhất |
| **NDCG@K** | $[0.0, 1.0]$ | Đo chất lượng sắp xếp thứ hạng tổng thể có phạt vị trí | Cao nhất ($\rightarrow 1.0$) |
