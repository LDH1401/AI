"""
Module Fine-tuning & Transcribe với OpenAI Whisper (Hugging Face Transformers) cho ASR tiếng Việt.
"""

from typing import List, Dict, Any
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class WhisperVietnameseASR:
    """
    Wrapper class chuẩn Production phục vụ Nhận dạng Tiếng nói (ASR) tiếng Việt sử dụng Whisper.
    """
    def __init__(self, model_name_or_path: str = "openai/whisper-small", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--> Initializing Whisper ASR Model from '{model_name_or_path}' on device '{self.device}'...")
        
        self.processor = WhisperProcessor.from_pretrained(
            model_name_or_path, 
            language="vietnamese", 
            task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def transcribe(self, waveform_16k: torch.Tensor) -> str:
        """
        waveform_16k: Tensor 1D [samples] biểu diễn âm thanh tại tần số 16,000 Hz.
        Returns: Văn bản tiếng Việt đã transcribe.
        """
        if waveform_16k.dim() > 1:
            waveform_16k = waveform_16k.squeeze()

        # Convert sang numpy array Float32
        audio_array = waveform_16k.cpu().numpy()

        # Trích xuất 80-channel Log-Mel Spectrogram
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        # Autoregressive Generation
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language="vietnamese",
                task="transcribe"
            )

        # Decode token thành chuỗi chữ tiếng Việt
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
