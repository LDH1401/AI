# Chương 6: Nền tảng Toán học: Lý thuyết Thông tin & Ước lượng Hợp lý Cực đại (MLE) trong Xử lý Âm thanh

---

Bản chất toán học của việc huấn luyện các mô hình xử lý tiếng nói và âm thanh (từ mô hình phân loại tiếng nói, ASR đến các mô hình sinh âm thanh như WaveNet, AudioLM, Bark) liên kết trực tiếp với nhau thông qua **Lý thuyết Thông tin (Information Theory)** và **Ước lượng Hợp lý Cực đại (Maximum Likelihood Estimation - MLE)**.

---

## 1. Tại sao "Tối thiểu hóa NLL" chính là "Cross-Entropy Loss"?

Xét một bài toán phân loại âm thanh (Audio Classification) hoặc dự đoán nhãn âm tiết/ký tự tại một khung âm thanh, với nhãn thực tế đúng là $x_{\text{true}} \in \{0, 1, \dots, K-1\}$ (ví dụ $K$ là số lớp âm thanh hoặc số ký tự trong bảng chữ cái).

### 1. Theo góc nhìn Phân loại (Cross-Entropy Loss):
Nhãn thực tế của khung âm thanh được biểu diễn dưới dạng vector xác suất one-hot $y = [y_0, y_1, \dots, y_{K-1}]$, trong đó $y_k = 1$ nếu $k = x_{\text{true}}$, và $y_k = 0$ với mọi $k \neq x_{\text{true}}$.

Công thức Cross-Entropy giữa phân phối nhãn thực $y$ và phân phối xác suất dự đoán $P_\theta$ do mạng nơ-ron xuất ra (sau hàm Softmax) là:

$$\text{Cross-Entropy}(y, P_\theta) = -\sum_{k=0}^{K-1} y_k \log_2 P_\theta(x = k \mid X)$$

Vì $y_k$ chỉ bằng $1$ tại đúng vị trí $x_{\text{true}}$ và bằng $0$ ở $K-1$ vị trí còn lại, toàn bộ tổng trên triệt tiêu và chỉ còn lại duy nhất một số hạng:

$$\text{Cross-Entropy}(y, P_\theta) = -\log_2 P_\theta(x = x_{\text{true}} \mid X)$$

### 2. Theo góc nhìn Xác suất & Thống kê (Negative Log-Likelihood - NLL):
Chi phí âm log-xác suất (Negative Log-Likelihood) gán cho giá trị thực tế quan sát được chính là:

$$\text{NLL} = -\log_2 P_\theta(x = x_{\text{true}} \mid X)$$

> [!NOTE]
> Hai khái niệm này là **một biểu thức toán học duy nhất** được gọi bằng hai tên khác nhau tùy theo góc nhìn của kỹ sư học máy (Cross-Entropy) hay nhà lý thuyết thống kê (NLL).

---

## 2. Tại sao tối thiểu hóa NLL ép phân phối mô hình $P_\theta(x)$ phải bám sát phân phối âm thanh thật trong tự nhiên?

Giả sử phân phối xác suất thống kê tự nhiên của âm thanh thực tế ngoài đời là $p_{\text{data}}(x)$ (đây là quy luật khách quan của tự nhiên quy định âm thanh tiếng nói con người), và phân phối do mạng nơ-ron dự đoán là $P_\theta(x)$.

Khi ta huấn luyện mô hình trên tập dữ liệu âm thanh thực tế $\mathcal{D}_{\text{real}}$, hàm mất mát trung bình trên toàn bộ dữ liệu chính là **Kỳ vọng của Cross-Entropy dưới phân phối dữ liệu thật**:

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}} [-\log_2 P_\theta(x \mid X)] = H(p_{\text{data}}, P_\theta)$$

Áp dụng mối liên hệ toán học giữa Cross-Entropy ($H$) và Phân kỳ Kullback-Leibler ($D_{\text{KL}}$):

$$H(p_{\text{data}}, P_\theta) = H(p_{\text{data}}) + D_{\text{KL}}(p_{\text{data}} \parallel P_\theta)$$

Trong đó:
- $H(p_{\text{data}}) = -\sum p_{\text{data}}(x) \log_2 p_{\text{data}}(x)$ là **Entropy nội tại của tín hiệu âm thanh thật trong tự nhiên**. Đây là một **hằng số cố định**, hoàn toàn không phụ thuộc vào trọng số mạng $\theta$.
- $D_{\text{KL}}(p_{\text{data}} \parallel P_\theta) = \sum_{x} p_{\text{data}}(x) \log_2 \left( \frac{p_{\text{data}}(x)}{P_\theta(x)} \right) \ge 0$ là khoảng cách đo sự sai lệch giữa phân phối mô hình học được $P_\theta$ và phân phối tự nhiên $p_{\text{data}}$.

Khi thực hiện thuật toán Gradient Descent để tìm bộ trọng số $\theta$ sao cho hàm loss $\mathcal{L}(\theta)$ đạt giá trị nhỏ nhất:

$$\arg\min_\theta \mathcal{L}(\theta) \iff \arg\min_\theta \left[ \text{Hằng số } + D_{\text{KL}}(p_{\text{data}} \parallel P_\theta) \right] \iff \arg\min_\theta D_{\text{KL}}(p_{\text{data}} \parallel P_\theta)$$

> [!IMPORTANT]
> **Bất đẳng thức Gibbs:** Phân kỳ KL luôn lớn hơn hoặc bằng 0 ($D_{\text{KL}} \ge 0$), và đạt giá trị cực tiểu bằng $0$ **khi và chỉ khi**:
> $$P_\theta(x \mid X) = p_{\text{data}}(x \mid X) \quad \text{hầu khắp nơi}$$

Mạng nơ-ron buộc phải uốn nắn các tham số trọng số $\theta$ sao cho đường cong xác suất $P_\theta$ trùng khít với quy luật phân bổ đặc trưng âm thanh thực tế của thế giới tự nhiên để đạt điểm loss thấp nhất.

---

## 3. Trực quan hóa Cơ chế Triệt tiêu của $D_{\text{KL}}$ qua Bài toán Mã lượng tử hóa Âm thanh (Audio Codebook)

Để hiểu trực quan cơ chế tối ưu hóa này trong xử lý âm thanh:

- **Bài toán gieo xúc xắc / Lượng tử hóa phổ âm thanh (Audio Codebook):** Giả sử phổ âm thanh tự nhiên phát ra từ giọng người có xác suất xuất hiện mã codebook thứ 6 (tương ứng với âm tiết phổ biến như nguyên âm `/a/`) là $80\%$, các mã khác mỗi mã chỉ chiếm $4\%$.
- **Nếu mô hình đoán bừa (đều $16.6\%$ cho 6 mã):** Chi phí NLL trung bình trên mỗi khung thời gian sẽ rất cao vì mô hình liên tục bị "bất ngờ" khi nguyên âm `/a/` xuất hiện dồn dập trong lời nói.
- **Sau khi tối ưu hàm loss NLL:** Mô hình bị ép phải điều chỉnh phân phối dự đoán về đúng tỷ lệ tự nhiên $[4\%, 4\%, 4\%, 4\%, 4\%, 80\%]$.
- **Khi đã khớp hoàn hảo:**
  - Chi phí kỳ vọng mô hình tính trước: $H(P_\theta) = -\sum P(k) \log_2 P(k)$.
  - Chi phí thực tế quan sát qua hàng triệu khung âm thanh thật: $\langle \text{NLL} \rangle = \frac{1}{M} \sum -\log_2 P(x_{\text{thực}})$.
  - Theo **Luật số lớn (Law of Large Numbers)**, $\langle \text{NLL} \rangle$ sẽ tiến tới chính xác $H(P_\theta)$, dẫn đến hiệu số khoảng cách:
    $$D = \langle \text{NLL} \rangle - H(P_\theta) = 0$$

---

## 4. Ứng dụng trong Mô hình Sinh Âm thanh Tự hồi quy (Autoregressive Speech Models)

Trong các mô hình sinh tiếng nói hiện đại như **WaveNet**, **AudioLM**, **Bark**, hay **MusicLM**, âm thanh được mã hóa thành chuỗi các mẫu PCM hoặc các mã Discrete Audio Tokens $x = (x_1, x_2, \dots, x_T)$.

Phân phối xác suất đồng thời của toàn bộ file âm thanh $P(x)$ được tách thành tích các xác suất có điều kiện theo quy tắc chuỗi (Chain Rule):

$$P_\theta(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} P_\theta(x_t \mid x_1, x_2, \dots, x_{t-1})$$

Việc tối thiểu hóa Negative Log-Likelihood trên toàn bộ chuỗi:

$$\mathcal{L}_{\text{NLL}}(\theta) = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \dots, x_{t-1})$$

chính là ép mô hình học được quy luật nối tiếp các bước sóng âm thanh tự nhiên của con người, giúp âm thanh sinh ra tự nhiên, không bị giật hay méo tiếng.

---

## 5. Góc nhìn MLE trong CTC Loss & Giải mã ASR

Trong bài toán ASR với CTC Loss, mục tiêu huấn luyện chính là **Tối đại hóa Log-Likelihood (MLE)** của chuỗi văn bản nhãn $Y$ cho trước chuỗi khung âm thanh đầu vào $X$:

$$\theta^* = \arg\max_\theta \sum_{(X, Y) \in \mathcal{D}} \log P_\theta(Y \mid X)$$

Vì không biết chính xác căn chỉnh thời gian (alignment), CTC tính $P(Y \mid X)$ bằng cách lấy **tổng xác suất của tất cả các đường đi khả dĩ (marginalization)**:

$$P_\theta(Y \mid X) = \sum_{\pi \in \mathcal{B}^{-1}(Y)} P_\theta(\pi \mid X)$$

Hàm mất mát CTC Loss $-\log P_\theta(Y \mid X)$ chính là hình thức **Negative Log-Likelihood Marginalization** chuẩn xác theo góc nhìn của Ước lượng Hợp lý Cực đại (MLE).
