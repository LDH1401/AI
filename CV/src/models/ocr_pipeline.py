"""
Module Production OCR Pipeline (Text Detection + Text Recognition).
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from PIL import Image

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


class ProductionOCRPipeline:
    """
    Pipeline bọc mô hình OCR (EasyOCR / TrOCR) cho bài toán Đọc Văn bản trên Bức ảnh.
    """
    def __init__(self, languages: List[str] = ["vi", "en"], gpu: bool = True):
        self.reader = None
        if HAS_EASYOCR:
            print(f"--> Initializing EasyOCR Reader for languages {languages}...")
            try:
                self.reader = easyocr.Reader(languages, gpu=gpu)
            except Exception as e:
                print(f"[Warning] Không thể khởi tạo EasyOCR GPU, thử CPU: {e}")
                self.reader = easyocr.Reader(languages, gpu=False)

    def read_text_from_image(self, image_input: Any) -> List[Dict[str, Any]]:
        """
        Nhận đầu vào là đường dẫn ảnh hoặc numpy array [H, W, C].
        Returns: Danh sách các Dictionary chứa Bounding Boxes, Text và Confidence Score.
        """
        if self.reader is None:
            return [{"bbox": [[0,0], [10,0], [10,10], [0,10]], "text": "OCR Engine Not Loaded", "confidence": 0.0}]

        # Run EasyOCR Pipeline (Text Detection + Recognition)
        results = self.reader.readtext(image_input)

        extracted_data = []
        for bbox, text, prob in results:
            # Convert bbox numpy types sang list Python
            bbox_list = [[int(pt[0]), int(pt[1])] for pt in bbox]
            extracted_data.append({
                "bbox": bbox_list,
                "text": text.strip(),
                "confidence": float(prob)
            })

        return extracted_data
