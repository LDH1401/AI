"""
Module Trích xuất Đặc trưng & Fine-tuning với Pretrained SSL Models (Wav2Vec 2.0 / HuBERT).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2ForCTC


class SSLFeatureExtractor(nn.Module):
    """
    Trích xuất Latent Context Representations từ các mô hình Self-Supervised (Wav2Vec 2.0 / HuBERT).
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: str = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--> Loading SSL Model '{model_name}' on device '{self.device}'...")
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, waveform_16k: torch.Tensor) -> torch.Tensor:
        """
        waveform_16k: Tensor 1D hoặc 2D âm thanh 16kHz.
        Returns: Latent embeddings shape [batch_size, sequence_length, hidden_dim (768)]
        """
        if waveform_16k.dim() == 1:
            waveform_16k = waveform_16k.unsqueeze(0)

        inputs = self.processor(
            waveform_16k.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs.input_values.to(self.device)

        outputs = self.model(input_values)
        return outputs.last_hidden_state  # [batch_size, seq_len, 768]


class Wav2Vec2FineTuner(nn.Module):
    """
    Wrapper bọc Wav2Vec 2.0 cho bài toán Fine-tuning ASR với CTC Loss.
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-base", vocab_size: int = 35):
        super().__init__()
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            vocab_size=vocab_size,
            ctc_loss_reduction="mean",
            pad_token_id=0
        )

    def forward(self, input_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.wav2vec2(input_values=input_values, labels=labels)
