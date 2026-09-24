# Todo List - Speech Processing Repository (Expanded & Comprehensive)

## Đã hoàn thành (Phase 1, Phase 2 & Phase 3)
- [x] Tạo file `requirements.txt` định nghĩa các thư viện Python chuẩn Production (Bổ sung pesq, pystoi)
- [x] Xây dựng bộ tài liệu lý thuyết chi tiết tại thư mục `docs/`:
  - [x] `docs/01_audio_fundamentals.md`: Vật lý sóng âm, Tần số, Lấy mẫu (Nyquist), Lượng tử hóa, PCM, WAV structure
  - [x] `docs/02_math_and_transforms.md`: Fourier Transform, STFT, Spectrogram, Mel-scale, Mel-Spectrogram, MFCC
  - [x] `docs/03_preprocessing_and_augmentation.md`: Filtering, VAD, Noise Reduction, Data Augmentation & SpecAugment
  - [x] `docs/04_classical_speech_processing.md`: Feature engineering thủ công, Phân loại ML cổ điển (GMM, SVM), ASR HMM-GMM
  - [x] `docs/05_deep_learning_and_pretrained.md`: CNN 2D / AST, Segmentation, Diarization, CTC Loss, Fine-tuning Whisper & Wav2Vec 2.0
  - [x] `docs/06_information_theory_mle_in_speech.md`: Bản chất toán học Information Theory, NLL, Cross-Entropy, KL-Divergence trong Speech & Auto-regressive Models
  - [x] `docs/07_self_supervised_learning_ssl.md`: Self-Supervised Learning trong Speech (Wav2Vec 2.0, HuBERT, WavLM, InfoNCE Loss, Vector Quantization)
  - [x] `docs/08_speech_synthesis_and_vocoders.md`: Tổng hợp tiếng nói TTS (Tacotron 2, FastSpeech 2, VITS), Neural Vocoders (HiFi-GAN, WaveNet, BigVGAN), Diffusion Models (DiffWave)
  - [x] `docs/09_evaluation_metrics_and_benchmarks.md`: Toán học & Bộ độ đo Đánh giá Chi tiết (PESQ, STOI, DER, EER, WER, CER, FAD, MCD)
  - [x] `docs/10_specialized_audio_tasks.md`: Các bài toán mở rộng (Audio Deepfake Detection, Speech Emotion Recognition - SER, Voice Conversion - VC, Music Information Retrieval - MIR & Shazam Fingerprinting, Sound Event Detection - SED, Audio Captioning)
- [x] Xây dựng codebase Python chuẩn Production tại `src/`:
  - [x] `src/config.py`: Configuration Dataclasses
  - [x] `src/features.py`: Feature extraction & SpecAugment bằng PyTorch / Torchaudio / Librosa
  - [x] `src/dataset.py`: PyTorch Audio Dataset & DataLoader
  - [x] `src/utils.py`: Utilities cho VAD, Audio I/O, Checkpoints
  - [x] `src/metrics.py`: Module tính toán toàn bộ độ đo PESQ, STOI, DER, EER, WER, CER, MCD
  - [x] `src/models/classifier.py`: Deep Learning Audio Classifier (CNN 2D + Attention)
  - [x] `src/models/asr_whisper.py`: Transcribe pipeline với Whisper
  - [x] `src/models/ssl_wav2vec.py`: Pipeline Fine-tuning & Feature Extraction với Pretrained Wav2Vec 2.0 / HuBERT
  - [x] `src/models/tasks_demo.py`: Demo code cho Audio Fingerprinting (Shazam), Speech Emotion Recognition & Deepfake Detection
  - [x] `src/train.py`: Training script cho Audio Classifier / ASR
  - [x] `src/infer.py`: Inference script cho file âm thanh thực tế
- [x] Xây dựng `README.md` chính của repository
