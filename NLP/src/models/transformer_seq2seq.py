"""
Mô hình Encoder-Decoder Seq2Seq Transformer cho Dịch máy (NMT) & Tóm tắt Văn bản.
"""

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Seq2SeqTransformerModel(nn.Module):
    """
    Wrapper bọc mô hình Encoder-Decoder (như mBART, T5) cho bài toán Dịch Máy & Tóm Tắt.
    """
    def __init__(self, model_name_or_path: str = "facebook/mbart-large-50-many-to-many-mmt", device: str = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--> Initializing Seq2Seq Model from '{model_name_or_path}' on device '{self.device}'...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

    def translate_vietnamese_to_english(self, vietnamese_text: str) -> str:
        """
        Dịch một câu tiếng Việt sang tiếng Anh.
        """
        self.tokenizer.src_lang = "vi_VN"
        encoded = self.tokenizer(vietnamese_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"]
            )
            
        translated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text.strip()
