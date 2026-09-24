# Implementation Plan - Recommendation Systems (RecSys) Repository

Xây dựng kho tài liệu và codebase chuẩn Production cho **Hệ thống Gợi ý (Recommendation Systems - RecSys)** từ nền tảng Lọc cộng tác, Phân tích Ma trận (SVD, ALS, BPR), Two-Tower DSSM, Wide & Deep, Sequential RecSys (SASRec, GRU4Rec), Graph Neural Networks (LightGCN), Multi-Task Learning (MMOE, ESMM), Candidate Retrieval (FAISS ANN, HNSW) và liên kết toán học sâu sắc với **Lý thuyết Thông tin & MLE (BPR Loss, InfoNCE, NDCG@K, MRR@K)**.

## Cấu trúc thư mục dự kiến
```
g:/Speech Processing/RecSys/
├── .gemini/
│   ├── implementation_plan.md
│   └── todo.md
├── docs/
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
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── matrix_factorization.py
│   │   ├── two_tower_dssm.py
│   │   ├── wide_and_deep.py
│   │   └── sequential_sasrec.py
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── utils.py
│   ├── train.py
│   └── infer.py
├── requirements.txt
└── README.md
```
