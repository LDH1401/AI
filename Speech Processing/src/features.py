"""
Module Trích xuất Đặc trưng Âm thanh (Audio Feature Extractor) tối ưu hóa trên PyTorch GPU/CPU.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
from src.config import AudioConfig


class AudioFeatureExtractor(nn.Module):
    """
    Trích xuất Log-Mel Spectrogram và áp dụng SpecAugment cho âm thanh.
    """
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config

        # Transform trích xuất Mel-Spectrogram
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=config.power
        )

        # SpecAugment Masking Transforms
        self.freq_masking = T.FrequencyMasking(freq_mask_param=config.freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=config.time_mask_param)

    def forward(self, waveforms: torch.Tensor, augment: bool = False) -> torch.Tensor:
        """
        waveforms: Tensor shape [batch_size, time_samples] hoặc [time_samples]
        Returns: Log-Mel Spectrogram Tensor [batch_size, 1, n_mels, time_frames]
        """
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)

        # Trích xuất Mel Spectrogram: [batch_size, n_mels, time_frames]
        mel_spec = self.mel_transform(waveforms)
        
        # Chuyển sang Log-Scale (dB)
        log_mel_spec = torch.log(mel_spec + 1e-9)

        # Áp dụng SpecAugment nếu đợt huấn luyện yêu cầu
        if augment and self.config.apply_spec_augment:
            log_mel_spec = self.freq_masking(log_mel_spec)
            log_mel_spec = self.time_masking(log_mel_spec)

        # Add Channel dimension cho CNN: [batch_size, 1, n_mels, time_frames]
        return log_mel_spec.unsqueeze(1)
```
