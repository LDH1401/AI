# Chương 1: Bản chất Vật lý & Số hóa Tín hiệu Âm thanh (Audio Fundamentals)

---

> [!IMPORTANT]
> 📌 **TỔNG QUAN & VAI TRÒ TRONG PIPELINE AI:**
> Máy tính hoàn toàn không thể trực tiếp xử lý các sóng âm thanh tự nhiên — máy tính chỉ hiểu các số liệu toán học.
> **Mục tiêu của Chương 1** là giải thích quy trình chuyển đổi đầu tiên: **Biến sóng rung động không khí tương tự (Analog) thành Chuỗi tín hiệu số 1D (File .wav/.mp3)**.
> - **Đầu vào:** Sóng rung động không khí Analog thu từ Microphone.
> - **Xử lý:** Lấy mẫu (Sampling 16kHz) và Lượng tử hóa (Quantization 16-bit) qua chip ADC.
> - **Đầu ra:** Chuỗi số thực biên độ 1D $x[t] \in [-1.0, 1.0]$.
> - **Vai trò:** Là bước đầu tiên bắt buộc phải thực hiện để nạp file âm thanh vào máy tính trước khi đưa vào các mô hình Deep Learning!

---

## 1. Bản chất Vật lý của Sóng Âm (Acoustic Wave Physics)

Sóng âm là loại **sóng cơ học dọc (longitudinal mechanical wave)** tạo ra bởi sự dao động của các phần tử môi trường (không khí, nước, chất rắn). Khi một vật thể dao động (ví dụ: dây thanh quản human vocal cords, màng loa), nó tạo ra các vùng **nén (compression)** và **dãn (rarefaction)** liên tục trong không khí.

### Các thuộc tính vật lý cơ bản:

1. **Tần số (Frequency - $f$):**
   - Số chu kỳ dao động hoàn chỉnh trong một giây, đơn vị là Hertz ($\text{Hz}$).
   - Công thức liên hệ với chu kỳ $T$:
     $$f = \frac{1}{T}$$
   - **Tần số quyết định Cao độ (Pitch):** Tần số cao hơn tương ứng với âm cao (treble), tần số thấp tương ứng với âm trầm (bass).
   - Dải thính giác người (Human Hearing Range): Từ $20 \text{ Hz}$ đến $20,000 \text{ Hz}$ ($20 \text{ kHz}$). Tín hiệu tiếng nói con người (Human Speech) thường tập trung chủ yếu ở dải $80 \text{ Hz} - 8,000 \text{ Hz}$.

2. **Biên độ (Amplitude - $A$):**
   - Độ chênh lệch áp suất cực đại so với áp suất khí quyển trung bình.
   - **Biên độ quyết định Âm lượng (Loudness/Volume):** Biên độ lớn hơn tương ứng với âm thanh to hơn. Tai người cảm nhận âm lượng theo thang logarit, thường sử dụng đơn vị **Decibel ($\text{dB}$)**:
     $$\text{SPL (Sound Pressure Level)} = 20 \log_{10} \left( \frac{P}{P_0} \right) \text{ dB}$$
     Trong đó $P_0 = 20 \; \mu\text{Pa}$ là ngưỡng nghe thấy của tai người tại tần số $1 \text{ kHz}$.

3. **Pha (Phase - $\phi$):**
   - Vị trí thời gian tương đối của chu kỳ dao động tại mốc $t = 0$.
   - Sóng hình sin đơn giản được biểu diễn dưới dạng:
     $$x(t) = A \cdot \sin(2\pi f t + \phi)$$

4. **Âm sắc (Timbre / Color):**
   - Đặc tính giúp phân biệt các nguồn âm thanh khác nhau (ví dụ: cùng nốt $A_4 = 440 \text{ Hz}$ phát ra từ đàn Guitar và đàn Piano).
   - Âm sắc được quyết định bởi tập hợp các **Họa âm (Harmonics / Overtones)** và hình dạng dạng sóng (Waveform shape).

---

## 2. Số hóa Tín hiệu Âm thanh (Digitalization of Sound Signals)

Âm thanh trong tự nhiên là tín hiệu **tương tự (Continuous Analog Signal)** $x(t)$. Để máy tính có thể lưu trữ và xử lý, tín hiệu này bắt buộc phải đi qua hệ thống chuyển đổi **ADC (Analog-to-Digital Converter)** qua 2 bước chính: **Lấy mẫu (Sampling)** và **Lượng tử hóa (Quantization)**.

```mermaid
graph LR
    A["Sóng âm Tương tự x(t)"] --> B["Lấy mẫu (Sampling)<br/>Tần số f_s"]
    B --> C["Tín hiệu Rời rạc x[n]"]
    C --> D["Lượng tử hóa (Quantization)<br/>Bit Depth b"]
    D --> E["Chuỗi Bit Số PCM (Digital Stream)"]
```

![Dạng sóng âm thanh 1D thô thu được sau quá trình lấy mẫu số hóa (Waveform)](../fig/waveform.png)

### 2.1. Lấy mẫu (Sampling) & Định lý Nyquist-Shannon

Lấy mẫu là quá trình biến đổi tín hiệu thời gian liên tục $x(t)$ thành tín hiệu thời gian rời rạc $x[n]$ bằng cách lấy giá trị tín hiệu tại các khoảng thời gian đều đặn $T_s$:
$$x[n] = x(n \cdot T_s), \quad n \in \mathbb{Z}$$
Trong đó $T_s$ là **chu kỳ lấy mẫu (Sampling Period)**, và $f_s = \frac{1}{T_s}$ là **tần số lấy mẫu (Sampling Rate)**.

> [!IMPORTANT]
> **Định lý Lấy mẫu Nyquist-Shannon:**
> Để có thể khôi phục hoàn toàn một tín hiệu liên tục có tần số cực đại là $f_{\text{max}}$ từ các mẫu rời rạc của nó, tần số lấy mẫu $f_s$ phải **ít nhất bằng 2 lần** tần số cực đại đó:
> $$f_s \ge 2 f_{\text{max}}$$
> Tần số $f_N = \frac{f_s}{2}$ được gọi là **Tần số Nyquist (Nyquist Frequency)**.

#### Hiện tượng Gấp băng tần (Aliasing):
Nếu $f_s < 2 f_{\text{max}}$, các tần số cao hơn $f_N$ sẽ bị "bóp méo" và xuất hiện dưới dạng các tần số ảo thấp hơn trong tín hiệu tái tạo (gọi là Aliasing).

```
Tần số thật > f_s / 2  ---> Biến thành tần số ảo f_alias = |f_real - k * f_s|
```

To giải quyết Aliasing, trước khi lấy mẫu người ta luôn dùng **Bộ lọc chống gấp băng (Anti-Aliasing Filter)** - bản chất là một bộ lọc thông thấp (Low-pass filter) để loại bỏ toàn bộ thành phần tần số $f > f_s / 2$.

#### Các chuẩn Tần số lấy mẫu phổ biến:
- **$8,000 \text{ Hz}$ (8 kHz):** Chuẩn điện thoại truyền thống (Telephony - G.711).
- **$16,000 \text{ Hz}$ (16 kHz):** Chuẩn cho các bài toán Nhận dạng tiếng nói (ASR / STT - Whisper, Wav2Vec 2.0).
- **$44,100 \text{ Hz}$ (44.1 kHz):** Chuẩn đĩa Audio CD.
- **$48,000 \text{ Hz}$ (48 kHz):** Chuẩn âm thanh Video chuyên nghiệp (Film/TV/YouTube).

---

### 2.2. Lượng tử hóa (Quantization) & Mã hóa PCM

Lượng tử hóa là quá trình làm tròn các giá trị biên độ liên tục $x[n]$ thành các mức giá trị rời rạc trong một dải số hữu hạn được quy định bởi **Bit Depth ($b$)**.

- **Số mức lượng tử (Quantization Levels):** $L = 2^b$.
- Ví dụ với Bit Depth $b = 16 \text{ bits}$, số mức lượng tử là $2^{16} = 65,536$ mức, tương ứng dải giá trị từ $-32,768$ đến $+32,767$.

#### Dynamic Range & Signal-to-Noise Ratio (SNR):
- **Dynamic Range (Dải động):** Tỷ lệ giữa âm thanh lớn nhất không bị vỡ (clipping) và âm thanh nhỏ nhất có thể ghi lại được.
- Công thức xấp xỉ cho SNR của lượng tử hóa đều:
  $$\text{SNR}_{\text{quantization}} \approx 6.02 \times b + 1.76 \text{ dB}$$
  - $b = 16 \text{ bits} \implies \text{SNR} \approx 98.08 \text{ dB}$ (Rất sạch cho thu âm tiếng nói).
  - $b = 24 \text{ bits} \implies \text{SNR} \approx 146.24 \text{ dB}$ (Chuẩn thu âm Studio chuyên nghiệp).

#### Mã hóa PCM (Pulse Code Modulation):
PCM là dạng đại diện số cơ bản nhất của âm thanh chưa nén. Các định dạng PCM chính bao gồm:
- **Int16 (Signed 16-bit PCM):** Mỗi mẫu âm thanh được biểu diễn bằng số nguyên 16-bit (`np.int16`).
- **Float32 (32-bit Floating Point):** Mỗi mẫu âm thanh được chuẩn hóa về số thực trong khoảng $[-1.0, +1.0]$ (`np.float32`). Đây là chuẩn đầu vào lý tưởng cho các mô hình Deep Learning PyTorch (`torch.FloatTensor`).

---

## 3. Định dạng File Âm thanh & Cấu trúc WAV Header

### 3.1. Phân loại định dạng File:
1. **Không nén (Uncompressed):** `.wav`, `.aiff`. Giữ nguyên dữ liệu PCM thô, chất lượng cao nhất nhưng dung lượng lớn.
2. **Nén không mất dữ liệu (Lossless Compression):** `.flac`, `.alac`. Nén dữ liệu bằng thuật toán entropy (tương tự zip), giảm khoảng 50% dung lượng mà không làm mất bit thông tin nào.
3. **Nén có mất dữ liệu (Lossy Compression):** `.mp3`, `.ogg`, `.aac`. Loại bỏ các dải tần số mà tai người khó nhận biết dựa trên mô hình tâm lý thính giác (Psychoacoustics), giảm đến 80-90% dung lượng.

### 3.2. Cấu trúc File WAV (RIFF Wave File Format)

File WAV sử dụng cấu trúc container **RIFF (Resource Interchange File Format)** bao gồm 3 chunk chính:

```
+-------------------------------------------------------+
|  RIFF Header Chunk ('RIFF', File Size, 'WAVE')        |
+-------------------------------------------------------+
|  Format Chunk ('fmt ', Chunk Size, AudioFormat,       |
|                NumChannels, SampleRate, ByteRate,     |
|                BlockAlign, BitsPerSample)             |
+-------------------------------------------------------+
|  Data Chunk ('data', Chunk Size, Raw PCM Samples...)  |
+-------------------------------------------------------+
```

#### Code Python kiểm tra cấu trúc WAV Header thủ công:

```python
import struct

def parse_wav_header(file_path: str) -> dict:
    """Đọc và phân tích trực tiếp cấu trúc WAV Header không dùng thư viện ngoài."""
    with open(file_path, "rb") as f:
        # Đọc 44 bytes đầu tiên của WAV Header chuẩn
        header = f.read(44)
        
    chunk_id, file_size, format_tag = struct.unpack("<4sI4s", header[0:12])
    subchunk1_id, subchunk1_size, audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack("<4sIHHIIHH", header[12:36])
    subchunk2_id, data_size = struct.unpack("<4sI", header[36:44])
    
    return {
        "ChunkID": chunk_id.decode("utf-8"),
        "FileSize": file_size,
        "Format": format_tag.decode("utf-8"),
        "NumChannels": num_channels,
        "SampleRate": sample_rate,
        "BitsPerSample": bits_per_sample,
        "DataSize": data_size,
        "DurationSec": data_size / (sample_rate * num_channels * (bits_per_sample // 8))
    }
```

---

## 4. Kênh âm thanh & Chuyển đổi Tần số Lấy mẫu (Resampling)

### 4.1. Mono vs Stereo (Single Channel vs Multi-Channel)
- **Mono (Monophonic):** Tín hiệu 1 kênh duy nhất $x[n]$. Các màng loa phát cùng một dạng sóng.
- **Stereo (Stereophonic):** Tín hiệu 2 kênh (Trái $L[n]$ và Phải $R[n]$). Dữ liệu PCM lưu dạng xen kẽ (Interleaved): $[L_0, R_0, L_1, R_1, \dots]$.
- Trong xử lý tiếng nói (Speech Processing), dữ liệu luôn được đưa về **Mono 1 kênh** bằng cách lấy trung bình:
  $$\text{Mono}[n] = \frac{L[n] + R[n]}{2}$$

### 4.2. Chuyển đổi Tần số Lấy mẫu (Resampling)
Khi mô hình ASR (như Whisper) yêu cầu đầu vào $16 \text{ kHz}$, nhưng file ghi âm đầu vào là $44.1 \text{ kHz}$, ta phải thực hiện **Resampling**:
- **Downsampling (Nội suy giảm):** Loại bỏ bớt mẫu bằng cách qua bộ lọc Anti-aliasing rồi lấy mẫu thưa (Decimation).
- **Upsampling (Nội suy tăng):** Chèn các số 0 giữa các mẫu cũ (Zero-padding) và qua bộ lọc thông thấp để làm mịn (Interpolation).

Sử dụng thư viện `torchaudio` để Resampling chuẩn production:

```python
import torch
import torchaudio

def load_and_resample(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load file audio, convert về Mono và Resample về target_sr."""
    waveform, sample_rate = torchaudio.load(file_path) # Tensor shape: [channels, time]
    
    # 1. Chuyển về Mono nếu là Stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # 2. Resample nếu tần số lấy mẫu không khớp
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        
    return waveform.squeeze(0) # Tra ve Tensor 1D: [time]
```
