"""
Script Dự đoán Tín hiệu Âm thanh (Inference Script) cho Audio Classification & Speech Transcribe.
"""

import argparse
import os
import torch
import torchaudio

from src.config import AudioConfig, ModelConfig
from src.features import AudioFeatureExtractor
from src.models.classifier import AudioClassifier
from src.models.asr_whisper import WhisperVietnameseASR
from src.utils import load_checkpoint


def predict_audio_class(
    audio_path: str,
    checkpoint_path: str,
    class_names: list,
    audio_cfg: AudioConfig = AudioConfig(),
    model_cfg: ModelConfig = ModelConfig(),
    device: str = "cpu"
) -> str:
    """
    Dự đoán nhãn cho 1 file audio dùng mô hình AudioClassifier đã huấn luyện.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

    # Load audio waveform
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != audio_cfg.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=audio_cfg.sample_rate)
        waveform = resampler(waveform)

    waveform = waveform.to(device)

    # Initialize Extractor & Model
    feature_extractor = AudioFeatureExtractor(audio_cfg).to(device)
    model = AudioClassifier(model_cfg).to(device)
    load_checkpoint(model, optimizer=None, filepath=checkpoint_path, device=device)
    model.eval()

    with torch.no_grad():
        log_mel_spec = feature_extractor(waveform, augment=False)
        logits = model(log_mel_spec)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    pred_class = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)
    print(f"--> Prediction Result: '{pred_class}' (Confidence: {confidence*100:.2f}%)")
    return pred_class


def transcribe_speech_vietnamese(audio_path: str, model_name: str = "openai/whisper-small") -> str:
    """
    Transcribe file âm thanh giọng nói tiếng Việt thành văn bản bằng OpenAI Whisper.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    asr_pipeline = WhisperVietnameseASR(model_name_or_path=model_name)
    text = asr_pipeline.transcribe(waveform)
    print(f"--> Speech Transcription: \"{text}\"")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Pipeline cho Speech Processing")
    parser.add_argument("--task", type=str, choices=["classify", "transcribe"], default="transcribe", help="Tác vụ: classify hoặc transcribe")
    parser.add_argument("--audio_path", type=str, required=True, help="Đường dẫn tới file âm thanh (.wav, .mp3, .flac)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_audio_classifier.pt", help="Đường dẫn file checkpoint cho classify")
    parser.add_argument("--model_name", type=str, default="openai/whisper-small", help="Model Whisper cho transcribe")

    args = parser.parse_args()

    if args.task == "transcribe":
        transcribe_speech_vietnamese(args.audio_path, args.model_name)
    elif args.task == "classify":
        predict_audio_class(args.audio_path, args.checkpoint, class_names=["Class_0", "Class_1"])
