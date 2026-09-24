# Chương 9: Bộ chỉ số Đánh giá & Benchmarks trong Speech Processing (Evaluation Metrics)

---

Để đánh giá chính xác hiệu năng của các hệ thống xử lý tiếng nói, cộng đồng nghiên cứu đã xây dựng tập hợp các độ đo định lượng (Quantitative Metrics) chuẩn mực cho từng bài toán cụ thể.

---

## 1. Độ đo Đánh giá Nhận dạng Tiếng nói (ASR / STT Metrics)

### 1.1. Word Error Rate (WER - Tỷ lệ Lỗi Từ)

WER là độ đo tiêu chuẩn để đánh giá chất lượng hệ thống ASR. WER tính toán khoảng cách Levenshtein (độ dài chỉnh sửa tối thiểu) giữa chuỗi văn bản gốc (Reference $R$) và chuỗi văn bản dự đoán (Hypothesis $H$).

$$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitution} + \text{Deletion} + \text{Insertion}}{N}$$

Trong đó:
- $S$ (Substitutions): Số từ bị thay thế sai (ví dụ: "mèo" biến thành "nèo").
- $D$ (Deletions): Số từ bị bỏ sót không nhận dạng được.
- $I$ (Insertions): Số từ bị chèn thêm dư thừa.
- $N$: Tổng số từ trong văn bản gốc Reference.

> [!NOTE]
> Giá trị WER càng nhỏ càng tốt ($\text{WER} = 0\%$ là hoàn hảo). WER có thể vượt quá $100\%$ nếu mô hình bị lặp từ (Insertions quá nhiều).

### 1.2. Character Error Rate (CER - Tỷ lệ Lỗi Ký tự)
Công thức CER hoàn toàn tương tự WER nhưng được tính trên mức độ từng **ký tự (Character)** thay vì từng từ. 
- CER đặc biệt quan trọng cho các ngôn ngữ không có khoảng trắng rõ ràng (như tiếng Trung, tiếng Nhật) hoặc các từ ghép tiếng Việt.

---

## 2. Độ đo Chất lượng & Độ rõ Tiếng nói (Speech Quality & Enhancement)

### 2.1. PESQ (Perceptual Evaluation of Speech Quality - ITU-T P.862)
PESQ là tiêu chuẩn quốc tế đo chất lượng cảm nhận âm thanh của tai người bằng cách so sánh tín hiệu thoại đã qua xử lý $y(t)$ với tín hiệu thoại gốc sạch $x(t)$.

- **Thang điểm:** Từ $-0.5$ đến $4.5$ (Điểm cao hơn tương ứng với chất lượng âm thanh sạch hơn, tự nhiên hơn).
- **Mô phỏng thính giác:** PESQ tính đến độ trễ thời gian, biến đổi biên độ phi tuyến và ngưỡng ngụy trang tần số của tai người.

### 2.2. STOI (Short-Time Objective Intelligibility)
STOI đo độ **rõ lời (Intelligibility)** - khả năng người nghe hiểu được nội dung câu nói trong môi trường nhiễu mạnh hoặc có tiếng vọng.

- **Thang điểm:** Từ $0.0$ (Hoàn toàn không nghe ra tiếng gì) đến $1.0$ (Nghe rõ ràng 100%).
- STOI dựa trên hệ số tương quan giữa phổ năng lượng ngắn hạn của tín hiệu gốc và tín hiệu đã lọc nhiễu.

---

## 3. Độ đo Phân định & Nhận dạng Người nói (Speaker Verification & Diarization)

### 3.1. Equal Error Rate (EER - Tỷ lệ Lỗi Cân bằng)
Trong bài toán Xác thực Người nói (Speaker Verification):
- **False Acceptance Rate (FAR):** Tỷ lệ nhận nhầm người lạ thành người dùng chuẩn.
- **False Rejection Rate (FRR):** Tỷ lệ từ chối sai người dùng chuẩn.

**EER** là điểm làm việc của mô hình mà tại đó $\text{FAR} = \text{FRR}$. Điểm EER càng thấp thể hiện mô hình xác thực càng chính xác.

### 3.2. Diarization Error Rate (DER - Tỷ lệ Lỗi Phân định Người nói)
DER đánh giá hiệu năng bài toán "Ai nói vào thời điểm nào":

$$\text{DER} = \frac{\text{Missed Speech} + \text{False Alarm} + \text{Speaker Confusion}}{\text{Total Groundtruth Duration}}$$

- **Missed Speech:** Thời gian có người nói nhưng mô hình đoán là khoảng lặng.
- **False Alarm:** Thời gian khoảng lặng nhưng mô hình đoán có người nói.
- **Speaker Confusion:** Thời gian nhận diện đúng có người nói nhưng gán nhầm sang người khác (ví dụ: Speaker 1 bị gán thành Speaker 2).

---

## 4. Độ đo Tổng hợp Âm thanh (TTS & Vocoder Generation Metrics)

### 4.1. Fréchet Audio Distance (FAD - Kilgour et al., Google)
FAD đánh giá độ tự nhiên và đa dạng của âm thanh do mô hình AI sinh ra so với âm thanh thật ngoài đời thực.

- FAD trích xuất các vector đặc trưng tầng ẩn qua mô hình **VGGish** pre-trained, sau đó tính khoảng cách Fréchet giữa 2 phân phối Gaussian $\mathcal{N}(\mu_r, \Sigma_r)$ (âm thật) và $\mathcal{N}(\mu_g, \Sigma_g)$ (âm sinh ra):

$$\text{FAD} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g}\right)$$

- Điểm FAD càng gần 0 thể hiện chất lượng âm thanh sinh ra càng chân thực.

### 4.2. Mel-Cepstral Distortion (MCD)
MCD đo khoảng cách sai lệch giữa các vector hệ số MFCCs của âm thanh gốc và âm thanh tổng hợp:

$$\text{MCD} = \frac{10}{\ln 10} \sqrt{2 \sum_{k=1}^{K} \left( c_k^{(\text{ref})} - c_k^{(\text{syn})} \right)^2} \quad (\text{dB})$$

---

## 5. Bảng tổng hợp các Đánh giá Metrics

| Bài toán (Task) | Độ đo chính (Primary Metric) | Đơn vị / Thang điểm | Mục tiêu tối ưu |
| :--- | :--- | :--- | :--- |
| **ASR / STT** | Word Error Rate (WER) / CER | Phần trăm ($\%$) | Thấp nhất ($\rightarrow 0\%$) |
| **Speech Enhancement** | PESQ / STOI | PESQ: $[-0.5, 4.5]$, STOI: $[0.0, 1.0]$ | Cao nhất |
| **Speaker Diarization** | Diarization Error Rate (DER) | Phần trăm ($\%$) | Thấp nhất ($\rightarrow 0\%$) |
| **Speaker Verification** | Equal Error Rate (EER) | Phần trăm ($\%$) | Thấp nhất ($\rightarrow 0\%$) |
| **Neural Vocoders / TTS** | FAD / MCD / MOS score | FAD: $[0, \infty)$, MOS: $[1, 5]$ | FAD/MCD thấp nhất, MOS cao nhất |
