# Speech & Audio Processing Repository & Documentation

Kho tài liệu lý thuyết toàn diện và Codebase chuẩn Production về **Xử lý Tín hiệu Âm thanh & Tiếng nói (Speech & Audio Processing)**, bao gồm từ Nền tảng Âm thanh Tự nhiên, Biến đổi Tín hiệu (STFT, Mel-Spectrogram, MFCCs), Tiền xử lý & Tăng cường (VAD, Denoising, SpecAugment), Xử lý Tiếng nói Cổ điển (Formant, Pitch Extraction, GMM-HMM), Nhận dạng Tiếng nói Tự động (ASR: DeepSpeech, Conformer, Whisper), Lý thuyết Thông tin & Loss Functions (CTC Loss, RNN-T Loss), Học Tự giám sát (SSL: Wav2Vec 2.0, HuBERT), Tổng hợp Tiếng nói (TTS: FastSpeech 2, HiFi-GAN Vocoder), Đánh giá chỉ số (WER, CER, MCD), Các bài toán âm thanh chuyên biệt (Audio Classification, Emotion Recognition, MIR), **Speech Language Models (Speech LLMs: Moshi, SeamlessM4T)** đến **Phân định Người nói (Speaker Diarization: ECAPA-TDNN, X-vectors)**.

---

> [!TIP]
> ⚡ **Dành cho kỹ sư cần áp dụng ngay (Mì Ăn Liền):**
> Xem hướng dẫn triển khai ứng dụng thực tế gọn gàng (Input `.wav`/`.mp3` $\to$ Tiền xử lý $\to$ Pretrained Models $\to$ Phân loại âm thanh / STT với Whisper / TTS) tại: **[Sổ tay Xử lý Âm thanh Mì Ăn Liền (Practical Speech Cookbook)](docs/PRACTICAL_SPEECH_COOKBOOK.md)**.

---

## 📚 Chuỗi Tài liệu Lý thuyết Toàn diện (11 Chương)

Tài liệu được thiết kế theo trình tự đầu cuối rõ ràng, chi tiết và dễ hiểu trong thư mục `docs/`:

1. [Chương 1: Nền tảng Âm thanh Tự nhiên & Số hóa](docs/01_audio_fundamentals.md)
   - Bản chất vật lý của sóng âm (Tần số, Biên độ, Pha).
   - Quá trình số hóa: Sampling Rate, Quantization Bit Depth, Định lý Lấy mẫu Nyquist-Shannon.
   - Định dạng âm thanh: Uncompressed (WAV, PCM) vs Lossy/Lossless Compression (MP3, FLAC).

2. [Chương 2: Cơ sở Toán học & Biến đổi Tín hiệu](docs/02_math_and_transforms.md)
   - Phân tích miền thời gian (Time-domain) vs miền tần số (Frequency-domain).
   - Biến đổi Fourier chuỗi ngắn (Short-Time Fourier Transform - STFT).
   - Biểu đồ phổ âm thanh (Spectrogram), Thang đo Tần số Mel (Mel-Scale) và ma trận Mel-Filterbank.
   - Hệ số Phổ Tần số Mel (Mel-Frequency Cepstral Coefficients - MFCCs) & Discrete Cosine Transform (DCT).

3. [Chương 3: Tiền xử lý & Tăng cường Tín hiệu Âm thanh](docs/03_preprocessing_and_augmentation.md)
   - Phát hiện Vùng Tiếng nói (Voice Activity Detection - VAD).
   - Khử nhiễu tín hiệu (Spectral Subtraction, Wiener Filter).
   - Kỹ thuật Tăng cường dữ liệu (Audio Data Augmentation): **SpecAugment** (Frequency & Time Masking), Time Stretch, Pitch Shift, Noise Injection.

4. [Chương 4: Xử lý Tiếng nói Cổ điển](docs/04_classical_speech_processing.md)
   - Đặc trưng âm thanh sinh học: Tần số F0 (Pitch Extraction qua Autocorrelation), Formants (F1, F2, F3).
   - Phân loại Âm thanh (Phonetics & Phonemes - Nguyên âm, Phụ âm).
   - Hệ thống ASR truyền thống: GMM-HMM (Gaussian Mixture Model - Hidden Markov Model) & Thuật toán Viterbi.

5. [Chương 5: Mô hình Deep Learning & Pretrained ASR](docs/05_deep_learning_and_pretrained.md)
   - Kiến trúc DeepSpeech (CNN + BiRNN).
   - **Conformer ASR (Convolution-augmented Transformer):** Kết hợp depthwise separable convolution và Multi-Head Self-Attention.
   - **Whisper ASR (OpenAI):** Encoder-Decoder Transformer đa ngữ cảnh và đa tác vụ (Multilingual Speech Recognition & Translation).

6. [Chương 6: Lý thuyết Thông tin & Ước lượng Hợp lý Cực đại trong Speech](docs/06_information_theory_mle_in_speech.md)
   - Nguyên lý MLE (Maximum Likelihood Estimation) và Cross-Entropy Loss.
   - **Connectionist Temporal Classification (CTC Loss):** Căn chỉnh tự động chuỗi thời gian không đồng bộ với ký tự blank token $\epsilon$.
   - **Recurrent Neural Network Transducer (RNN-T Loss):** Kiến trúc ASR thời gian thực (Prediction Network + Transcription Network + Joint Network).

7. [Chương 7: Học Tự giám sát trong Âm thanh (Self-Supervised Learning - SSL)](docs/07_self_supervised_learning_ssl.md)
   - Thử thách gắn nhãn dữ liệu tiếng nói.
   - **Wav2Vec 2.0 (Meta AI):** Học Tương Tương phản (Contrastive Learning) trên không gian lượng tử hóa (Quantized Vector Latents).
   - **HuBERT (Hidden-Unit BERT):** Học tự giám sát qua dự đoán các cụm ký tự ẩn (Cluster Labels từ K-Means).
   - **WavLM:** Học tự giám sát kết hợp khôi phục đoạn âm thanh bị che khuất và khử nhiễu.

8. [Chương 8: Tổng hợp Tiếng nói & Vocoders (Text-to-Speech - TTS)](docs/08_speech_synthesis_and_vocoders.md)
   - Tổng quan Pipeline TTS 2 giai đoạn: Text Embeddings $\to$ Mel-Spectrogram $\to$ Waveform.
   - **Acoustic Models:** Tacotron 2 (Autoregressive Attention) & **FastSpeech 2** (Non-autoregressive với Length Regulator).
   - **Neural Vocoders:** MelGAN, **HiFi-GAN** (Multi-Period Discriminator + Multi-Scale Discriminator) chuyển đổi Mel-Spectrogram thành sóng âm thanh sắc nét.

9. [Chương 9: Bộ chỉ số Đánh giá & Benchmarks](docs/09_evaluation_metrics_and_benchmarks.md)
   - Bộ chỉ số Đánh giá ASR: **Word Error Rate (WER)**, **Character Error Rate (CER)**, Phoneme Error Rate (PER):
     $$\text{WER} = \frac{S + D + I}{N}$$
   - Bộ chỉ số Đánh giá TTS: **Mel-Cepstral Distortion (MCD)**, Mean Opinion Score (MOS).

10. [Chương 10: Các Bài toán Xử lý Âm thanh Chuyên biệt](docs/10_specialized_audio_tasks.md)
    - Phân loại Âm thanh Môi trường (Environmental Sound Classification - ESC-50, UrbanSound8K).
    - Nhận dạng Cảm xúc qua Giọng nói (Speech Emotion Recognition - SER).
    - Trích xuất Thông tin Âm nhạc (Music Information Retrieval - MIR: Tempo, Pitch, Genre Classification).
    - Nhận dạng Từ khóa (Keyword Spotting - KWS / Wake-Word Detection).

11. [Chương 11: Mô hình Ngôn ngữ Tiếng nói & Phân định Người nói](docs/11_speech_llms_and_speaker_diarization.md)
    - **Speech LLMs & Real-Time Audio Models:** Moshi (Kyutai), SeamlessM4T (Meta AI), Audio Codecs (EnCodec, SoundStream, DAC).
    - **Speaker Diarization ("Who Spoke When?"):** VAD $\to$ Trích xuất Speaker Embeddings (**ECAPA-TDNN, X-vectors**) $\to$ Clustering.

---

## 🛠️ Cấu trúc Kho lưu trữ (Project Repository Structure)

```
g:/AI/Speech Processing/
├── .gemini/                       # Lưu trữ context dự án & kế hoạch thực hiện
│   ├── implementation_plan.md
│   └── todo.md
├── docs/                          # Chuỗi 11 chương tài liệu lý thuyết Speech & Audio Processing
│   ├── 01_audio_fundamentals.md
│   ├── 02_math_and_transforms.md
│   ├── 03_preprocessing_and_augmentation.md
│   ├── 04_classical_speech_processing.md
│   ├── 05_deep_learning_and_pretrained.md
│   ├── 06_information_theory_mle_in_speech.md
│   ├── 07_self_supervised_learning_ssl.md
│   ├── 08_speech_synthesis_and_vocoders.md
│   ├── 09_evaluation_metrics_and_benchmarks.md
│   ├── 10_specialized_audio_tasks.md
│   └── 11_speech_llms_and_speaker_diarization.md
├── src/                           # Codebase Python chuẩn Production
│   ├── models/
│   │   ├── __init__.py
│   │   ├── asr_whisper.py          # Wrapper bọc mô hình Whisper ASR (Transcribe / Translate)
│   │   ├── classifier.py           # PyTorch Audio Classifier & Emotion Recognition Model
│   │   ├── speaker_diarization.py  # ECAPA-TDNN Speaker Embedding Extractor
│   │   ├── ssl_wav2vec.py          # Wav2Vec 2.0 Self-Supervised Feature Extractor
│   │   └── tasks_demo.py           # Demo tổng hợp các tác vụ âm thanh
│   ├── __init__.py
│   ├── config.py                 # Dataclasses cấu hình Audio, Feature & Models
│   ├── dataset.py                # PyTorch AudioDataset & Feature Loader
│   ├── features.py               # Trích xuất Mel-Spectrogram, STFT, MFCCs với torchaudio/librosa
│   ├── metrics.py                # Module tính WER, CER, Accuracy, F1-Score
│   ├── utils.py                  # Tiện ích Audio IO, Resampling & Denoising
│   ├── train.py                  # Script huấn luyện PyTorch Audio Classifier chuẩn Production
│   └── infer.py                  # Script Inference cho ASR, Classification, Diarization
├── requirements.txt               # Danh sách thư viện Python cần thiết
└── README.md                      # Hướng dẫn chính của dự án Speech Processing
```

---

## 💻 Hướng dẫn Cài đặt & Sử dụng (Dành cho Người dùng thực thi trên Terminal/CMD)

> [!IMPORTANT]
> Theo Quy tắc 2, bạn hãy tự mở Terminal (CMD/PowerShell) tại thư mục `g:\AI\Speech Processing` để thực hiện các lệnh dưới đây.

### Bước 1: Cài đặt các thư viện phụ thuộc

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy script thử nghiệm Nhận dạng Tiếng nói (Whisper ASR)

```bash
python -m src.infer --task transcribe --audio_path path/to/your_speech.wav
```

### Bước 3: Chạy script thử nghiệm Phân loại Cảm xúc / Tín hiệu Âm thanh

```bash
python -m src.infer --task classify --audio_path path/to/your_sample.wav
```

### Bước 4: Chạy script thử nghiệm Trích xuất Vector Người nói (ECAPA-TDNN / Diarization)

```bash
python -m src.infer --task diarize --audio_path path/to/your_sample.wav
```

### Bước 5: Chạy script Huấn luyện PyTorch Audio Classifier Model

```bash
python -m src.train
```
