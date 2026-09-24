"""
Mô hình Extractive Question Answering (Hỏi Đáp Trích xuất) dựa trên Transformer (PhoBERT / BERT).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class ExtractiveQuestionAnsweringModel(nn.Module):
    """
    Wrapper bọc Transformer cho bài toán Hỏi Đáp Trích xuất (SQuAD style).
    Dự đoán vị trí Start Token và End Token của câu trả lời trong đoạn ngữ cảnh Context.
    """
    def __init__(self, model_name_or_path: str = "vinai/phobert-base"):
        super().__init__()
        print(f"--> Initializing Extractive QA Model from '{model_name_or_path}'...")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_positions: torch.Tensor = None,
        end_positions: torch.Tensor = None
    ):
        """
        input_ids shape: [batch_size, seq_len]
        start_positions shape: [batch_size]
        end_positions shape: [batch_size]
        """
        outputs = self.qa_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        return outputs.loss, outputs.start_logits, outputs.end_logits
