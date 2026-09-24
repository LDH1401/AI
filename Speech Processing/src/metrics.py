"""
Module Tính toán Toàn bộ các Chỉ số Đánh giá (Evaluation Metrics) trong Speech Processing.
"""

from typing import List, Dict, Union
import numpy as np
import torch
import jiwer

try:
    import pesq as pesq_lib
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

try:
    import pystoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False


class SpeechMetricsCalculator:
    """Class tập hợp các hàm tính toán độ đo chất lượng âm thanh & ASR."""

    @staticmethod
    def compute_wer_cer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Tính Word Error Rate (WER) và Character Error Rate (CER).
        """
        wer = jiwer.wer(references, hypotheses)
        cer = jiwer.cer(references, hypotheses)
        return {
            "WER": float(wer),
            "CER": float(cer)
        }

    @staticmethod
    def compute_pesq(ref_signal: np.ndarray, deg_signal: np.ndarray, sr: int = 16000) -> float:
        """
        Tính Perceptual Evaluation of Speech Quality (PESQ).
        Đòi hỏi tần số lấy mẫu sr là 8000 Hz (nb) hoặc 16000 Hz (wb).
        """
        if not HAS_PESQ:
            print("[Warning] Thư viện 'pesq' chưa được cài đặt. Trả về 0.0.")
            return 0.0
        mode = "wb" if sr == 16000 else "nb"
        try:
            score = pesq_lib.pesq(sr, ref_signal, deg_signal, mode)
            return float(score)
        except Exception as e:
            print(f"[Error] Lỗi khi tính PESQ: {e}")
            return 0.0

    @staticmethod
    def compute_stoi(ref_signal: np.ndarray, deg_signal: np.ndarray, sr: int = 16000) -> float:
        """
        Tính Short-Time Objective Intelligibility (STOI).
        """
        if not HAS_STOI:
            print("[Warning] Thư viện 'pystoi' chưa được cài đặt. Trả về 0.0.")
            return 0.0
        try:
            score = pystoi.stoi(ref_signal, deg_signal, sr, extended=False)
            return float(score)
        except Exception as e:
            print(f"[Error] Lỗi khi tính STOI: {e}")
            return 0.0

    @staticmethod
    def compute_mcd(ref_mfcc: np.ndarray, syn_mfcc: np.ndarray) -> float:
        """
        Tính Mel-Cepstral Distortion (MCD) giữa 2 ma trận MFCCs shape [n_mfcc, time_frames].
        """
        min_frames = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
        ref_cut = ref_mfcc[:, :min_frames]
        syn_cut = syn_mfcc[:, :min_frames]

        diff = ref_cut - syn_cut
        squared_diff = np.sum(diff ** 2, axis=0)
        rmse = np.sqrt(squared_diff)
        
        # Hệ số đổi sang dB: (10 / ln(10)) * sqrt(2)
        factor = (10.0 / np.log(10.0)) * np.sqrt(2.0)
        mcd = factor * np.mean(rmse)
        return float(mcd)
