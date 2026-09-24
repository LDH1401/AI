"""
Script Dự đoán Tín hiệu Hình ảnh/Video (CV Inference Script) cho OCR, Detection, Captioning & VQA.
"""

import argparse
import os

from src.models.ocr_pipeline import ProductionOCRPipeline
from src.models.yolo_detr_detector import AdvancedObjectDetector
from src.models.image_captioning_blip import BLIPImageCaptioner
from src.models.visual_qa_llava import PretrainedVisualQAPipeline


def run_ocr(image_path: str):
    """Chạy OCR đọc chữ trên ảnh."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path '{image_path}' not found.")
    ocr_pipeline = ProductionOCRPipeline()
    results = ocr_pipeline.read_text_from_image(image_path)
    print(f"--> OCR Results for '{image_path}':")
    for item in results:
        print(f"  Text: \"{item['text']}\" (Conf: {item['confidence']*100:.1f}%) | BBox: {item['bbox']}")


def run_object_detection(image_path: str, model_name: str = "yolov8n.pt"):
    """Chạy Object Detection phát hiện vật thể."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path '{image_path}' not found.")
    detector = AdvancedObjectDetector(model_name=model_name)
    detections = detector.detect_objects(image_path)
    print(f"--> Detection Results for '{image_path}':")
    for det in detections:
        print(f"  Class: '{det['class_name']}' (Conf: {det['confidence']*100:.1f}%) | BBox: {det['bbox']}")


def run_image_captioning(image_path: str):
    """Chạy Image Captioning sinh câu mô tả ảnh."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path '{image_path}' not found.")
    captioner = BLIPImageCaptioner()
    captioner.generate_caption(image_path)


def run_visual_qa(image_path: str, question: str):
    """Chạy Visual Question Answering (VQA)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path '{image_path}' not found.")
    vqa_pipeline = PretrainedVisualQAPipeline()
    vqa_pipeline.answer_question(image_path, question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced CV Inference Pipeline")
    parser.add_argument("--task", type=str, choices=["ocr", "detect", "caption", "vqa"], default="caption", help="Tác vụ: ocr, detect, caption hoặc vqa")
    parser.add_argument("--image_path", type=str, required=True, help="Đường dẫn file ảnh đầu vào")
    parser.add_argument("--question", type=str, default="What is in the image?", help="Câu hỏi dành cho tác vụ vqa")
    parser.add_argument("--model_name", type=str, default="yolov8n.pt", help="Tên mô hình detection")

    args = parser.parse_args()

    if args.task == "ocr":
        run_ocr(args.image_path)
    elif args.task == "detect":
        run_object_detection(args.image_path, args.model_name)
    elif args.task == "caption":
        run_image_captioning(args.image_path)
    elif args.task == "vqa":
        run_visual_qa(args.image_path, args.question)
