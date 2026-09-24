"""
Module Advanced Object Detection bọc YOLOv8/v11 & RT-DETR (Ultralytics).
"""

from typing import List, Dict, Any
import numpy as np

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class AdvancedObjectDetector:
    """
    Wrapper bọc mô hình Advanced Object Detector (YOLOv8, YOLOv11, RT-DETR).
    """
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = None
        if HAS_YOLO:
            print(f"--> Loading Object Detection Model '{model_name}'...")
            self.model = YOLO(model_name)

    def detect_objects(self, image_input: Any, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Nhận đầu vào là ảnh (Path hoặc Numpy Array).
        Returns: Danh sách các Bounding Boxes, Class IDs, Class Names và Confidence.
        """
        if self.model is None:
            return [{"bbox": [0, 0, 100, 100], "class_id": 0, "class_name": "unknown", "confidence": 0.0}]

        results = self.model.predict(image_input, conf=conf_threshold, verbose=False)[0]

        detections = []
        boxes = results.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            cls_name = results.names[cls_id]

            detections.append({
                "bbox": [round(c, 2) for c in xyxy],
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "class_name": cls_name
            })

        return detections
