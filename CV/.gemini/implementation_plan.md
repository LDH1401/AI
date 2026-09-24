# Implementation Plan - Advanced Computer Vision (CV) Repository

Xây dựng kho tài liệu và codebase chuẩn Production cho **Computer Vision nâng cao (Advanced Computer Vision)** bỏ qua Phân loại ảnh cơ bản, tập trung vào:
1. Optical Character Recognition (OCR Text Detection & Recognition).
2. Advanced Object Detection (YOLOv8-v11, DETR, RT-DETR, Anchor-free & NMS/IoU Losses).
3. Instance & Semantic Segmentation (Mask R-CNN, SAM / SAM 2 - Segment Anything).
4. Video Classification & Action Recognition (3D CNNs, SlowFast, TimeSformer, ViViT).
5. Image Captioning (BLIP-2, LLaVA, Vision-Language Encoders).
6. Video Captioning (Video-LLaVA, Temporal Attention Decoders).
7. Image Visual Question Answering (Image VQA, Qwen2-VL, Multimodal Fusion).
8. Video Visual Question Answering (Video VQA, Spatial-Temporal Reasoning).
9. Multi-Modal Alignment (CLIP, InfoNCE Loss, Vision-Text Representation Learning).

## Cấu trúc thư mục dự kiến
```
g:/Speech Processing/CV/
├── .gemini/
│   ├── implementation_plan.md
│   └── todo.md
├── docs/
│   ├── 01_cv_advanced_overview.md
│   ├── 02_ocr_text_detection_and_recognition.md
│   ├── 03_advanced_object_detection_yolo_detr.md
│   ├── 04_instance_and_semantic_segmentation_sam.md
│   ├── 05_video_classification_and_action_recognition.md
│   ├── 06_image_captioning_blip_llava.md
│   ├── 07_video_captioning_architecture.md
│   ├── 08_visual_question_answering_image_vqa.md
│   ├── 09_video_question_answering_video_vqa.md
│   └── 10_multimodal_alignment_clip_and_mle.md
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ocr_pipeline.py
│   │   ├── yolo_detr_detector.py
│   │   ├── video_classifier_3d.py
│   │   ├── image_captioning_blip.py
│   │   └── visual_qa_llava.py
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── utils.py
│   ├── train.py
│   └── infer.py
├── requirements.txt
└── README.md
```
