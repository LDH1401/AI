# Master AI Knowledge Base & Production Codebases

Chào mừng bạn đến với Kho lưu trữ Tri thức và Codebase Python chuẩn Production bao phủ **5 Lĩnh vực AI Cốt lõi**:
1. 🎙️ **Speech Processing** - Xử lý Tín hiệu & Tiếng nói
2. 🔤 **NLP** - Xử lý Ngôn ngữ Tự nhiên & Mô hình Ngôn ngữ Lớn (LLMs)
3. 👁️ **CV** - Thị giác Máy tính Nâng cao, VLM & VLA Embodied AI
4. 🎯 **RecSys** - Hệ thống Gợi ý & Generative Retrieval
5. 🎮 **RL** - Học Tăng cường & Chứng minh Toán học Chuyên sâu

---

## 📂 Danh mục Các Dự án Thành phần

### 1. 🎙️ [Speech Processing](Speech%20Processing/README.md)
- **Vị trí:** `Speech Processing/`
- **⚡ Fast Track (Mì ăn liền):** [Sổ tay Xử lý Âm thanh Mì Ăn Liền (Practical Speech Cookbook)](Speech%20Processing/docs/PRACTICAL_SPEECH_COOKBOOK.md) - Hướng dẫn ứng dụng thực tế gọn gàng.
- **Nội dung:** Chuỗi 11 chương tài liệu lý thuyết Âm thanh, STFT, Mel-Spectrogram, MFCCs, Tiền xử lý VAD/Denoising/SpecAugment, GMM-HMM, Whisper ASR, CTC/RNN-T Loss, Wav2Vec 2.0 / HuBERT SSL, FastSpeech 2 / HiFi-GAN TTS, Metrics (WER, CER, MCD), Speech LLMs (Moshi, SeamlessM4T), Speaker Diarization (ECAPA-TDNN, X-vectors).
- **Codebase:** `Speech Processing/src/` (Whisper ASR, Audio Classifier, Speaker Diarization).

### 2. 🔤 [Natural Language Processing (NLP)](NLP/README.md)
- **Vị trí:** `NLP/`
- **Nội dung:** Chuỗi 14 chương tài liệu NLP (Tokenization BPE/WordPiece, Word2Vec, Transformer Deep-Dive, BERT/PhoBERT, GPT/LLMs, Prompt Engineering, RAG, PEFT LoRA/QLoRA 4-bit, Alignment RLHF/DPO, Document QA & RAG, **MoE DeepSeek-V3**, **Multi-Head Latent Attention MLA**, **Mamba-2 SSM**, **Reasoning LLMs DeepSeek-R1 & GRPO Loss**, **LLM Quantization AWQ/GPTQ/GGUF & Distillation**).
- **Codebase:** `NLP/src/` (BPE Tokenizer, Word2Vec, Transformer, PhoBERT, RAG, LoRA, Document QA, DeepSeek GRPO Trainer, Quantization Helper).

### 3. 👁️ [Advanced Computer Vision (CV)](CV/README.md)
- **Vị trí:** `CV/`
- **Nội dung:** Chuỗi 13 chương tài liệu CV nâng cao (Bỏ qua phân loại ảnh cơ bản, tập trung vào OCR DBNet/TrOCR, Detection YOLOv8-v11/DETR/RT-DETR, Segmentation SAM/SAM 2, Video 3D CNNs/SlowFast/TimeSformer, Image/Video Captioning BLIP-2/LLaVA, Image/Video VQA, Multi-Modal CLIP Alignment, **Modern VLMs Qwen2-VL / InternVL 2.5 / Molmo**, **VLA Models & Embodied AI OpenVLA / RT-2 / ACT**, **3D Gaussian Splatting & Video Diffusion**).
- **Codebase:** `CV/src/` (OCR Pipeline, YOLO/DETR Detector, 3D Video Classifier, BLIP Captioner, VQA Pipeline, VLA Action Controller).

### 4. 🎯 [Recommendation Systems (RecSys)](RecSys/README.md)
- **Vị trí:** `RecSys/`
- **Nội dung:** Chuỗi 11 chương tài liệu RecSys (Industrial Two-Stage Pipeline Retrieval vs Ranking, Matrix Factorization BPR Loss, Google Two-Tower DSSM, Wide & Deep, Deep & Cross DCN-v2, SASRec Sequential RecSys, LightGCN SOTA GNN, Google MMOE & Alibaba ESMM Multi-Task, FAISS ANN Search HNSW/IVF-PQ, Metrics NDCG@K / MRR@K, **Generative Retrieval TIGER/DSI & LLM4Rec**).
- **Codebase:** `RecSys/src/` (BPR Matrix Factorization, Two-Tower DSSM, Wide & Deep, SASRec, FAISS Retrieval Index).

### 5. 🎮 [Reinforcement Learning (RL)](RL/README.md)
- **Vị trí:** `RL/`
- **Nội dung:** Chuỗi 11 chương tài liệu RL **CHỨNG MINH TOÁN HỌC TỪ BẢN CHẤT** (MDP, **Chứng minh Định lý Co Bellman Contraction Mapping Theorem**, **Chứng minh Định lý Policy Gradient Theorem từ đầu**, Information Geometry, Fisher Information Matrix, TRPO, PPO Clipped Loss, Continuous Control DDPG/TD3, Soft Actor-Critic SAC SOTA Maximum Entropy RL, World Models & DreamerV3, MARL MADDPG/QMIX, **Chứng minh Direct Preference Optimization DPO từ đầu mà không cần PPO**, **Offline RL Conservative Q-Learning & Decision Transformer**).
- **Codebase:** `RL/src/` (DQN / Double DQN / Dueling DQN Agent, PPO Agent với GAE, SAC Agent cho Continuous Control, Replay Buffer).

---

## 🛠️ Cấu trúc Kho lưu trữ Tổng thể (Master Structure)

```
g:/AI/
├── Speech Processing/             # DỰ ÁN XỬ LÝ ÂM THANH & TIẾNG NÓI
│   ├── docs/ (11 chương lý thuyết Speech)
│   ├── src/ (Codebase PyTorch Speech)
│   ├── requirements.txt
│   └── README.md
├── NLP/                           # DỰ ÁN NLP, LLMS, MOE, MAMBA & DEEPSEEK-R1
│   ├── docs/ (14 chương lý thuyết NLP)
│   ├── src/ (Codebase PyTorch NLP)
│   ├── requirements.txt
│   └── README.md
├── CV/                            # DỰ ÁN ADVANCED CV, VLM & VLA EMBODIED AI
│   ├── docs/ (13 chương lý thuyết CV Nâng cao, VLM, VLA & 3DGS)
│   ├── src/ (Codebase PyTorch CV)
│   ├── requirements.txt
│   └── README.md
├── RecSys/                        # DỰ ÁN HỆ THỐNG GỢI Ý (RECSYS)
│   ├── docs/ (11 chương lý thuyết RecSys)
│   ├── src/ (Codebase PyTorch RecSys)
│   ├── requirements.txt
│   └── README.md
├── RL/                            # DỰ ÁN HỌC TĂNG CƯỜNG (RL MATH PROOFS)
│   ├── docs/ (11 chương chứng minh toán học RL)
│   ├── src/ (Codebase PyTorch RL)
│   ├── requirements.txt
│   └── README.md
└── README.md                      # Master README tổng hợp tại thư mục gốc g:\AI\
```

---

## 💻 Hướng dẫn Nhanh (Quick Start cho Người dùng trên CMD)

> [!IMPORTANT]
> Theo Quy tắc 2, bạn hãy tự mở Terminal (CMD/PowerShell) tại thư mục dự án tương ứng để thực hiện các lệnh dưới đây.

```bash
# 1. Chạy dự án Speech Processing
cd "Speech Processing"
pip install -r requirements.txt
python -m src.infer --task transcribe --audio_path path/to/sample.wav

# 2. Chạy dự án NLP
cd ../NLP
pip install -r requirements.txt
python -m src.infer --task translate --text "Xin chào thế giới!"

# 3. Chạy dự án Computer Vision (CV)
cd ../CV
pip install -r requirements.txt
python -m src.infer --task caption --image_path path/to/sample.jpg

# 4. Chạy dự án Recommendation Systems (RecSys)
cd ../RecSys
pip install -r requirements.txt
python -m src.infer --user_id 42 --top_k 10

# 5. Chạy dự án Reinforcement Learning (RL)
cd ../RL
pip install -r requirements.txt
python -m src.train
```
# AI
