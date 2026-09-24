"""
Module Tính toán Các Chỉ số Đánh giá (Evaluation Metrics) cho NLP (BLEU, ROUGE, F1, Accuracy, PPL).
"""

from typing import List, Dict, Union
import numpy as np
import torch
import evaluate


class NLPMetricsCalculator:
    """Class tập hợp các hàm tính toán độ đo đánh giá văn bản & mô hình NLP."""

    def __init__(self):
        try:
            self.bleu_metric = evaluate.load("sacrebleu")
        except Exception:
            self.bleu_metric = None

        try:
            self.rouge_metric = evaluate.load("rouge")
        except Exception:
            self.rouge_metric = None

    @staticmethod
    def compute_accuracy_f1(predictions: np.ndarray, references: np.ndarray) -> Dict[str, float]:
        """
        Tính Accuracy và Macro F1-score cho Phân loại Văn bản.
        """
        acc = float(np.mean(predictions == references))
        
        # Calculate Macro F1
        classes = np.unique(references)
        f1_list = []
        for c in classes:
            tp = np.sum((predictions == c) & (references == c))
            fp = np.sum((predictions == c) & (references != c))
            fn = np.sum((predictions != c) & (references == c))
            
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            f1_list.append(f1)

        macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
        return {
            "Accuracy": acc,
            "Macro_F1": macro_f1
        }

    def compute_bleu(self, hypotheses: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Tính toán chỉ số BLEU score cho Dịch máy (Machine Translation).
        references: Danh sách các câu tham chiếu [[ref1_a, ref1_b], [ref2_a, ...]]
        """
        if self.bleu_metric is None:
            return {"BLEU": 0.0}
        results = self.bleu_metric.compute(predictions=hypotheses, references=references)
        return {"BLEU": float(results["score"])}

    def compute_rouge(self, hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        """
        Tính toán chỉ số ROUGE-1, ROUGE-2, ROUGE-L cho Tóm tắt Văn bản (Summarization).
        """
        if self.rouge_metric is None:
            return {"ROUGE1": 0.0, "ROUGE2": 0.0, "ROUGEL": 0.0}
        results = self.rouge_metric.compute(predictions=hypotheses, references=references)
        return {
            "ROUGE1": float(results["rouge1"]),
            "ROUGE2": float(results["rouge2"]),
            "ROUGEL": float(results["rougeL"])
        }

    @staticmethod
    def compute_perplexity(loss: float) -> float:
        """
        Tính toán Perplexity (PPL) từ Cross-Entropy Loss: PPL = exp(Loss).
        """
        return float(np.exp(loss))
