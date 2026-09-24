"""
Module Dataset & Collate Function hỗ trợ Load và Xử lý Batching Âm thanh PyTorch.
"""

from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset
import torchaudio
from src.config import AudioConfig


class AudioClassificationDataset(Dataset):
    """
    Dataset chuẩn PyTorch cho bài toán Phân loại Âm thanh.
    Nhận đầu vào danh sách các đường dẫn file âm thanh (.wav) và nhãn tương ứng.
    """
    def __init__(self, file_paths: List[str], labels: List[int], config: AudioConfig):
        self.file_paths = file_paths
        self.labels = labels
        self.config = config

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load file audio
        waveform, sr = torchaudio.load(file_path)

        # Chuyển Stereo về Mono nếu cần
        if waveform.shape[0] > 1 and self.config.mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample về sample_rate mong muốn
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.config.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)  # Shape [time_samples]

        # Chuẩn hóa biên độ tín hiệu (Z-score)
        if self.config.normalize_signal:
            mean = waveform.mean()
            std = waveform.std() + 1e-9
            waveform = (waveform - mean) / std

        return waveform, label


def audio_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hàm Collate bọc Batch chứa các waveform có độ dài thời gian khác nhau bằng Zero-Padding.
    """
    waveforms, labels = zip(*batch)
    
    # Tìm độ dài thời gian lớn nhất trong batch
    max_len = max([w.size(0) for w in waveforms])
    
    # Pad các waveforms về cùng độ dài max_len
    padded_waveforms = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :w.size(0)] = w
        
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded_waveforms, labels_tensor
