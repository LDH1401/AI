"""
Module minh họa các thuật toán Xử lý Âm thanh Chuyên sâu:
1. Thuật toán Shazam Audio Fingerprinting (Constellation Map Matching).
2. Speech Emotion Recognition (SER Feature Extraction).
3. Audio Deepfake Detection (LFCC Feature Extractor).
"""

from typing import List, Tuple, Dict
import numpy as np
import scipy.ndimage as ndimage
import librosa


class ShazamFingerprinter:
    """
    Triển khai Thuật toán Trích xuất Hash Fingerprint kiểu Shazam (Avery Wang, 2003).
    """
    def __init__(self, fan_value: int = 15, amp_min: float = 10.0):
        self.fan_value = fan_value
        self.amp_min = amp_min

    def get_constellation_map(self, spectrogram: np.ndarray) -> List[Tuple[int, int]]:
        """
        Tìm các điểm cực đại cục bộ (Peaks) trên Spectrogram 2D.
        spectrogram: Ma trận 2D [freq_bins, time_frames]
        Returns: Danh sách các tọa độ peak (time_frame, freq_bin)
        """
        # Áp dụng Max Filter 2D để tìm điểm cực đại trong vùng lân cận
        struct = ndimage.generate_binary_structure(2, 1)
        neighborhood = ndimage.iterate_structure(struct, 5)
        
        local_max = ndimage.maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
        background = (spectrogram < self.amp_min)
        eroded_background = ndimage.binary_erosion(background, structure=neighborhood, border_value=1)
        
        detected_peaks = local_max ^ eroded_background
        freq_indices, time_indices = np.where(detected_peaks)
        
        peaks = list(zip(time_indices, freq_indices))
        peaks.sort(key=lambda x: x[0])  # Sắp xếp theo trục thời gian
        return peaks

    def generate_hashes(self, peaks: List[Tuple[int, int]]) -> List[Tuple[str, int]]:
        """
        Tạo danh sách các Hash Keys từ Constellation Map Peaks.
        Returns: Danh sách các cặp (Hash Key String, time_anchor)
        """
        hashes = []
        for i in range(len(peaks)):
            for j in range(1, self.fan_value):
                if (i + j) < len(peaks):
                    freq1 = peaks[i][1]
                    freq2 = peaks[i + j][1]
                    t1 = peaks[i][0]
                    t2 = peaks[i + j][0]
                    t_delta = t2 - t1
                    
                    if 0 <= t_delta <= 200:  # Giới hạn thời gian Target Zone
                        hash_key = f"{freq1}|{freq2}|{t_delta}"
                        hashes.append((hash_key, t1))
        return hashes


class LFCCExtractor:
    """
    Trích xuất Linear Frequency Cepstral Coefficients (LFCC) cho bài toán Audio Deepfake Detection.
    """
    def __init__(self, sample_rate: int = 16000, n_lfcc: int = 20, n_fft: int = 512, hop_length: int = 160):
        self.sample_rate = sample_rate
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """
        Trích xuất LFCCs trên dải tần số tuyến tính.
        waveform: Np.ndarray 1D
        Returns: Ma trận LFCCs shape [n_lfcc, time_frames]
        """
        # 1. Tính STFT Magnitude Spectrogram
        stft = librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length)
        mag_spec = np.abs(stft) ** 2

        # 2. Xây dựng Linear Filterbank (dải tần tuyến tính thay vì Mel scale)
        n_filters = 40
        linear_filterbank = librosa.filters.semitone_filterbank(n_notes=n_filters, sr=self.sample_rate) if hasattr(librosa.filters, 'semitone_filterbank') else None
        
        if linear_filterbank is None:
            # Fallback tạo Linear Filterbank đơn giản
            n_fft_bins = mag_spec.shape[0]
            linear_filterbank = np.eye(n_filters, n_fft_bins)

        # 3. Nhân Filterbank & lấy Logarithm
        linear_spec = np.dot(linear_filterbank, mag_spec)
        log_linear_spec = np.log(linear_spec + 1e-9)

        # 4. Discrete Cosine Transform (DCT)
        lfcc = scipy.fftpack.dct(log_linear_spec, axis=0, type=2, norm='ortho')[:self.n_lfcc]
        return lfcc


class SpeechEmotionFeatureExtractor:
    """
    Trích xuất tập đặc trưng Âm điệu (Prosodic Features - Pitch F0, Intensity, Jitter, Shimmer) cho SER.
    """
    @staticmethod
    def extract_prosodic_features(waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """
        waveform: Np.ndarray 1D
        Returns: Dictionary chứa các chỉ số F0, RMS, Zero Crossing Rate
        """
        # Pitch F0 estimation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            waveform, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'), 
            sr=sample_rate
        )
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        
        # RMS Energy (Intensity)
        rms = librosa.feature.rms(y=waveform)[0]
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=waveform)[0]

        return {
            "f0_mean": float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0,
            "f0_std": float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0,
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr))
        }
