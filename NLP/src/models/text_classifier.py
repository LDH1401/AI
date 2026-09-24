"""
Mô hình Deep Learning Text Classifier (PhoBERT / BERT Backbone + Attention Classifier Head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.config import ModelConfig


class AttentionPoolingHead(nn.Module):
    """
    Cơ chế Self-Attention Pooling gom tụ tất cả các token embeddings linh hoạt theo trọng số chú ý.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state shape: [batch_size, seq_len, hidden_dim]
        # attention_mask shape: [batch_size, seq_len]
        scores = self.attention(last_hidden_state).squeeze(-1)  # [batch_size, seq_len]
        
        # Gán $-\infty$ cho các vị trí Padding
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)      # [batch_size, seq_len, 1]
        
        pooled = torch.sum(last_hidden_state * weights, dim=1)  # [batch_size, hidden_dim]
        return pooled


class TransformerTextClassifier(nn.Module):
    """
    Mô hình Phân loại Văn bản / Phân tích Cảm xúc bọc Pretrained Transformer (PhoBERT / BERT).
    """
    def __init__(self, model_name_or_path: str = "vinai/phobert-base", config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        print(f"--> Initializing Transformer Backbone from '{model_name_or_path}'...")
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_dim = self.encoder.config.hidden_size

        self.attention_pool = AttentionPoolingHead(hidden_dim)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_dim, config.num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        input_ids shape: [batch_size, seq_len]
        attention_mask shape: [batch_size, seq_len]
        Returns: Logits Tensor [batch_size, num_classes]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

        if self.config.use_attention_head:
            pooled = self.attention_pool(last_hidden_state, attention_mask)
        else:
            # Lấy vector [CLS] token làm đại diện
            pooled = last_hidden_state[:, 0, :]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
