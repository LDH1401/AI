# Advanced Computer Vision (CV) Repository & Documentation

Kho tài liệu lý thuyết toàn diện và Codebase chuẩn Production về **Thị giác Máy tính Nâng cao (Advanced Computer Vision - CV)**, bỏ qua các bài toán phân loại ảnh đơn thuần, tập trung vào: **Optical Character Recognition (OCR)**, **Advanced Object Detection (YOLOv8-v11, DETR, RT-DETR, Anchor-free, IoU Losses)**, **Segmentation (Mask R-CNN, SAM / SAM 2)**, **Video Classification & Action Recognition (3D CNNs, SlowFast, TimeSformer)**, **Image Captioning (BLIP-2, LLaVA)**, **Video Captioning**, **Visual Question Answering (Image VQA, Video VQA)**, **Multi-Modal Alignment (CLIP & InfoNCE Contrastive Loss)**, **Vision-Language Models (Modern VLMs: Qwen2-VL, InternVL 2.5, Molmo)** và **Vision-Language-Action Models (VLA & Embodied AI: OpenVLA, RT-2, ACT)**.

---

## 📚 Chuỗi Tài liệu Lý thuyết Toàn diện (12 Chương)

1. [Chương 1: Tổng quan các Bài toán Thị giác Máy tính Nâng cao](docs/01_cv_advanced_overview.md)
2. [Chương 2: Nhận dạng Văn bản & Đọc Tài liệu (OCR)](docs/02_ocr_text_detection_and_recognition.md)
3. [Chương 3: Phát hiện Vật thể Nâng cao (YOLOv8/v11 & DETR)](docs/03_advanced_object_detection_yolo_detr.md)
4. [Chương 4: Phân đoạn Đối tượng & Mô hình SAM (Segmentation & SAM 2)](docs/04_instance_and_semantic_segmentation_sam.md)
5. [Chương 5: Phân loại Video & Nhận dạng Hành động (Video Classification)](docs/05_video_classification_and_action_recognition.md)
6. [Chương 6: Mô tả Ảnh bằng Ngôn ngữ Tự nhiên (BLIP/LLaVA)](docs/06_image_captioning_blip_llava.md)
7. [Chương 7: Mô tả Video bằng Ngôn ngữ Tự nhiên (Video Captioning)](docs/07_video_captioning_architecture.md)
8. [Chương 8: Hỏi Đáp trên Hình ảnh (Image VQA)](docs/08_visual_question_answering_image_vqa.md)
9. [Chương 9: Hỏi Đáp trên Video (Video VQA)](docs/09_video_question_answering_video_vqa.md)
10. [Chương 10: Nền tảng Toán học: Multi-Modal Alignment (CLIP & Contrastive Learning)](docs/10_multimodal_alignment_clip_and_mle.md)
11. [Chương 11: Mô hình Ngôn ngữ Thị giác Hiện đại (VLM Deep Dive: Qwen2-VL, InternVL 2.5, Molmo)](docs/11_vision_language_models_vlm_deep_dive.md)
12. [Chương 12: Mô hình Thị giác-Ngôn ngữ-Hành động cho AI Nhập thể (VLA Models & Embodied AI: OpenVLA, RT-2, ACT)](docs/12_vision_language_action_vla_embodied_ai.md)

---

## 🛠️ Cấu trúc Kho lưu trữ (Project Repository Structure)

```
g:/Speech Processing/CV/
├── docs/ (12 chương lý thuyết CV Nâng cao, VLM & VLA Embodied AI)
├── src/
│   ├── models/
│   │   ├── ocr_pipeline.py         # Production OCR Pipeline (EasyOCR / TrOCR)
│   │   ├── yolo_detr_detector.py   # Object Detector Wrapper (YOLOv8/v11 / RT-DETR)
│   │   ├── video_classifier_3d.py  # 3D ResNet Video Action Recognition Classifier
│   │   ├── image_captioning_blip.py# Pretrained BLIP Image Captioning
│   │   ├── visual_qa_llava.py      # Pretrained BLIP-VQA Pipeline
│   │   └── vla_embodied_model.py   # VLA Action Tokenizer & Policy Head Controller
```
