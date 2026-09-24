"""
Script Dự đoán Tín hiệu Văn bản (NLP Inference Script) cho Text Classification & Machine Translation.
"""

import argparse
import os
import torch

from src.config import NLPConfig, ModelConfig
from src.tokenizer import VietnameseNLPTokenizer
from src.models.text_classifier import TransformerTextClassifier
from src.models.transformer_seq2seq import Seq2SeqTransformerModel
from src.utils import load_nlp_checkpoint


def classify_text_sentiment(
    text: str,
    checkpoint_path: str,
    class_names: list = None,
    nlp_cfg: NLPConfig = NLPConfig(),
    model_cfg: ModelConfig = ModelConfig(),
    device: str = "cpu"
) -> str:
    """
    Phân loại cảm xúc / chủ đề cho 1 câu văn bản tiếng Việt.
    """
    tokenizer_wrapper = VietnameseNLPTokenizer(nlp_cfg)
    encoded = tokenizer_wrapper.encode(text)

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model = TransformerTextClassifier(nlp_cfg.model_name_or_path, model_cfg).to(device)
    if os.path.exists(checkpoint_path):
        load_nlp_checkpoint(model, optimizer=None, filepath=checkpoint_path, device=device)
    model.eval()

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    class_names = class_names or ["Tiêu cực", "Trung tính", "Tích cực"]
    pred_class = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    print(f"--> Input Text: \"{text}\"")
    print(f"--> Classification Result: '{pred_class}' (Confidence: {confidence*100:.2f}%)")
    return pred_class


def translate_text(vietnamese_text: str, model_name: str = "facebook/mbart-large-50-many-to-many-mmt") -> str:
    """
    Dịch máy câu tiếng Việt sang tiếng Anh.
    """
    translator = Seq2SeqTransformerModel(model_name_or_path=model_name)
    english_translation = translator.translate_vietnamese_to_english(vietnamese_text)
    print(f"--> Source (Vi): \"{vietnamese_text}\"")
    print(f"--> Target (En): \"{english_translation}\"")
    return english_translation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Inference Pipeline")
    parser.add_argument("--task", type=str, choices=["classify", "translate"], default="classify", help="Tác vụ: classify hoặc translate")
    parser.add_argument("--text", type=str, required=True, help="Câu văn bản đầu vào")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_nlp/best_text_classifier.pt", help="Đường dẫn file checkpoint cho classify")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base", help="Model name")

    args = parser.parse_args()

    if args.task == "classify":
        nlp_cfg = NLPConfig(model_name_or_path=args.model_name)
        classify_text_sentiment(args.text, args.checkpoint, nlp_cfg=nlp_cfg)
    elif args.task == "translate":
        translate_text(args.text)
