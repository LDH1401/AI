"""
Module Tiện ích Lượng tử hóa & Load LLM 4-bit / 8-bit bằng BitsAndBytes & AutoGPTQ.
"""

from typing import Dict, Any
import torch

try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def load_quantized_llm(model_name: str, load_in_4bit: bool = True):
    """
    Nạp mô hình LLM nén 4-bit (NF4) bằng BitsAndBytesConfig chuẩn production.
    """
    if not HAS_TRANSFORMERS:
        print("[Warning] Transformers chưa cài đặt.")
        return None

    if load_in_4bit:
        print(f"--> Loading LLM '{model_name}' in 4-bit NF4 Quantization mode...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        return model
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
