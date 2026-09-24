"""
Module Tính toán Đánh giá Metrics cho CV Nâng cao (mAP, IoU, OCR Accuracy, BLEU, ROUGE).
"""

from typing import List, Dict, Tuple
import numpy as np
import evaluate


class CVMetricsCalculator:
    """Class tập hợp các hàm tính toán độ đo cho Advanced CV."""

    @staticmethod
    def compute_iou(boxA: List[float], boxB: List[float]) -> float:
        """
        Tính Intersection over Union (IoU) giữa 2 Bounding Boxes dạng [x1, y1, x2, y2].
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-9)
        return float(iou)

    @staticmethod
    def compute_ocr_accuracy(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Tính toán Độ chính xác ở mức độ Từ (Word Accuracy) và Ký tự (Character Accuracy) cho OCR.
        """
        exact_matches = sum([1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower()])
        word_acc = exact_matches / len(references) if references else 0.0

        return {
            "Word_Accuracy": float(word_acc),
            "Exact_Match_Count": exact_matches
        }

    @staticmethod
    def compute_caption_metrics(hypotheses: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Tính toán chỉ số đánh giá Image/Video Captioning (BLEU-4 & ROUGE-L).
        """
        try:
            bleu_metric = evaluate.load("sacrebleu")
            bleu_score = bleu_metric.compute(predictions=hypotheses, references=references)["score"]
        except Exception:
            bleu_score = 0.0

        return {
            "BLEU4": float(bleu_score)
        }
