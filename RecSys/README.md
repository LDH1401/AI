# Comprehensive Recommendation Systems (RecSys) Repository & Documentation

Kho tài liệu lý thuyết toàn diện và Codebase chuẩn Production về **Hệ thống Gợi ý (Recommendation Systems - RecSys)**, phủ rộng từ Nền tảng Lọc cộng tác (Collaborative Filtering), Phân tích Ma trận (Matrix Factorization SVD, ALS, BPR Pairwise Loss), Lọc dựa trên nội dung (Content-Based), **Mô hình Hai tháp Candidate Retrieval (Google Two-Tower DSSM)**, **Mô hình Ranking (Google Wide & Deep, Deep & Cross DCN-v2)**, **Sequential & Session RecSys (SASRec Self-Attention, GRU4Rec)**, **Graph Neural Networks (LightGCN SOTA)**, **Multi-Task & Multi-Objective RecSys (Google MMOE, Alibaba ESMM)**, **Candidate Retrieval & FAISS ANN Search (HNSW, IVF-PQ Indexing)** đến mối liên kết toán học sâu sắc với **Lý thuyết Thông tin, MLE & Bộ chỉ số Đánh giá Top-K (NDCG@K, MRR@K, Recall@K, HitRate@K)**.

---

## 📚 Chuỗi Tài liệu Lý thuyết Toàn diện (10 Chương)

Tài liệu được thiết kế theo trình tự đầu cuối rõ ràng, chi tiết và dễ hiểu trong thư mục `docs/`:

1. [Chương 1: Nền tảng & Kiến trúc Tổng quan Hệ thống Gợi ý](docs/01_recsys_fundamentals_and_paradigms.md)
   - Khái niệm RecSys & Pipeline 2 giai đoạn tiêu chuẩn công nghiệp (Candidate Retrieval vs Ranking vs Re-ranking).
   - Explicit Feedback vs Implicit Feedback.
   - Thử thách: Cold-Start Problem (User/Item mới), Data Sparsity ($99.9\%$), Filter Bubble vs Serendipity.

2. [Chương 2: Lọc Cộng tác & Phân tích Ma trận](docs/02_collaborative_filtering_and_matrix_factorization.md)
   - User-based CF & Item-based CF (Cosine / Pearson Similarity).
   - Matrix Factorization (SVD, ALS Alternating Least Squares).
   - **BPR (Bayesian Personalized Ranking):** Pairwise Ranking Loss cho Implicit Feedback:
     $$\mathcal{L}_{\text{BPR}} = -\sum_{(u, i, j)} \ln \sigma(\hat{x}_{u,i} - \hat{x}_{u,j}) + \lambda \|\Theta\|^2$$

3. [Chương 3: Lọc Dựa trên Nội dung (Content-Based Filtering)](docs/03_content_based_filtering.md)
   - Xây dựng Item Profile (Text TF-IDF/Embeddings, Categorical One-Hot, Numerical Normalization).
   - User Profile Vector & Cosine Similarity.
   - Ưu điểm khắc phục Cold-Start item mới & Giải thích gợi ý (Explainability).

4. [Chương 4: Kiến trúc Two-Tower & Deep Learning trong Candidate Retrieval](docs/04_deep_learning_recsys_dssm_two_tower.md)
   - **Google Two-Tower DSSM Architecture:** User Tower $\mathbf{u} = \text{MLP}(x_{\text{user}})$ & Item Tower $\mathbf{v} = \text{MLP}(x_{\text{item}})$.
   - Dot-Product Relevance Scoring $\mathbf{u}^T \mathbf{v}$.
   - **In-batch Negative Softmax Loss:** Tận dụng samples cùng batch làm negatives.
   - Offline pre-computation + Online FAISS ANN Search $< 5\text{ms}$.

5. [Chương 5: Mô hình Wide & Deep và Deep & Cross trong Ranking](docs/05_wide_and_deep_deep_and_cross.md)
   - **Google Wide & Deep (2016):** Wide Component (Memorization - Cross-Product Transformations) + Deep Component (Generalization - MLP).
   - **Deep & Cross Network (DCN / DCN-v2):** Cross Network tự động tạo feature crosses bậc cao $x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$.

6. [Chương 6: Gợi ý theo Chuỗi & Session (SASRec)](docs/06_sequential_and_session_based_recommendation.md)
   - Bản chất Sequential RecSys trên chuỗi thời gian $S_u = (i_1, i_2, \dots, i_t)$.
   - **GRU4Rec:** RNN/GRU trên session clicks.
   - **SASRec (Self-Attention Based Sequential Recommendation):** Causal Masked Transformer Self-Attention bắt ngữ cảnh tương tác ngắn hạn và sở thích dài hạn.

7. [Chương 7: Mạng Nơ-ron Đồ thị trong RecSys (LightGCN)](docs/07_graph_neural_networks_in_recsys.md)
   - Biểu diễn User-Item Bipartite Graph & Message Passing $N$-hop.
   - **LightGCN (He et al., SIGIR 2020 - SOTA GNN):** Loại bỏ hoàn toàn Feature Transformation $W$ và hàm phi tuyến $\sigma$, chỉ giữ lại Weighted Neighborhood Aggregation:
     $$e_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|} \sqrt{|\mathcal{N}_i|}} e_i^{(k)}$$

8. [Chương 8: Gợi ý Đa Mục tiêu & Multi-Task Learning (MMOE & ESMM)](docs/08_multi_task_and_multi_objective_recommendation.md)
   - Multi-Objective RecSys: CTR, CVR, Watch Time, Engagement.
   - **Google MMOE (Multi-gate Mixture-of-Experts):** Khắc phục Negative Transfer bằng Shared Experts + Task Gating Networks.
   - **Alibaba ESMM (Entire Space Multi-Task Model):** Giải quyết Sample Selection Bias trong CVR qua định lý $P(\text{CTCVR}) = P(\text{CTR}) \times P(\text{CVR})$.

9. [Chương 9: Candidate Retrieval & Tìm kiếm Láng giềng Xấp xỉ (FAISS & ANN)](docs/09_candidate_retrieval_and_approximate_nearest_neighbors.md)
   - Approximate Nearest Neighbors (ANN) vs Exact KNN.
   - **HNSW (Hierarchical Navigable Small World Graphs):** Đồ thị đa tầng phân cấp $\mathcal{O}(\log N)$.
   - **FAISS IVF-PQ (Facebook AI):** Inverted File Index + Product Quantization nén $97\%$ RAM và tìm kiếm siêu tốc trên GPU/CPU.

10. [Chương 10: Nền tảng Toán học: MLE & Bộ chỉ số Đánh giá RecSys](docs/10_information_theory_mle_and_evaluation_in_recsys.md)
    - MLE & Softmax Cross-Entropy với In-batch Negatives ($D_{\text{KL}}$ minimization).
    - Top-K Metrics: Precision@K, Recall@K, Hit Rate@K, MRR@K.
    - **NDCG@K (Normalized Discounted Cumulative Gain):** Đánh giá thứ hạng có phạt vị trí:
      $$\text{DCG}@K = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

---

## 🛠️ Cấu trúc Kho lưu trữ (Project Repository Structure)

```
g:/Speech Processing/RecSys/
├── .gemini/                       # Lưu trữ context dự án & kế hoạch thực hiện
│   ├── implementation_plan.md
│   └── todo.md
├── docs/                          # Chuỗi 10 chương tài liệu lý thuyết RecSys
│   ├── 01_recsys_fundamentals_and_paradigms.md
│   ├── 02_collaborative_filtering_and_matrix_factorization.md
│   ├── 03_content_based_filtering.md
│   ├── 04_deep_learning_recsys_dssm_two_tower.md
│   ├── 05_wide_and_deep_deep_and_cross.md
│   ├── 06_sequential_and_session_based_recommendation.md
│   ├── 07_graph_neural_networks_in_recsys.md
│   ├── 08_multi_task_and_multi_objective_recommendation.md
│   ├── 09_candidate_retrieval_and_approximate_nearest_neighbors.md
│   └── 10_information_theory_mle_and_evaluation_in_recsys.md
├── src/                           # Codebase Python chuẩn Production
│   ├── models/
│   │   ├── __init__.py
│   │   ├── matrix_factorization.py# BPR Matrix Factorization với Pairwise Loss
│   │   ├── two_tower_dssm.py       # Two-Tower Candidate Retrieval Model
│   │   ├── wide_and_deep.py       # Wide & Deep Ranking Model
│   │   └── sequential_sasrec.py   # SASRec Self-Attention Sequential RecSys
│   ├── __init__.py
│   ├── config.py                 # Dataclasses cấu hình RecSys, Model & Training
│   ├── dataset.py                # PyTorch BPR Dataset & Sequential Dataset
│   ├── metrics.py                # Module tính toán NDCG@K, MRR@K, Recall@K, HitRate@K
│   ├── utils.py                  # Tiện ích FAISS Indexing & Checkpointing
│   ├── train.py                  # Script huấn luyện PyTorch chuẩn Production
│   └── infer.py                  # Script Inference cho Retrieval (FAISS) & Ranking
├── requirements.txt               # Danh sách thư viện Python cần thiết
└── README.md                      # Hướng dẫn chính của dự án RecSys
```

---

## 💻 Hướng dẫn Cài đặt & Sử dụng (Dành cho Người dùng thực thi trên Terminal/CMD)

> [!IMPORTANT]
> Theo Quy tắc 2, bạn hãy tự mở Terminal (CMD/PowerShell) tại thư mục `g:\Speech Processing\RecSys` để thực hiện các lệnh dưới đây.

### Bước 1: Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy script thử nghiệm Candidate Retrieval (Two-Tower + FAISS ANN Search)

```bash
python -m src.infer --user_id 42 --top_k 10
```
