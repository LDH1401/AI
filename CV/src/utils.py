"""
Module Tiện ích cho Advanced CV: Bounding Box Transformations, NMS, Save Checkpoint.
"""

import os
from typing import List, Tuple
import numpy as np
import torch


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi Bounding Boxes từ dạng [x_center, y_center, width, height] sang [x1, y1, x2, y2].
    """
    boxes_out = np.zeros_like(boxes)
    boxes_out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0
    boxes_out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0
    boxes_out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0
    boxes_out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0
    return boxes_out


def non_max_suppression_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """
    Thuật toán Non-Maximum Suppression (NMS) lọc bỏ các Bounding Box bị đè lặp.
    boxes shape: [N, 4] dạng [x1, y1, x2, y2]
    scores shape: [N]
    Returns: Danh sách các index được giữ lại
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def save_cv_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, filepath: str):
    """Lưu trọng số mô hình CV vào file checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None
    }
    torch.save(state, filepath)
    print(f"--> Saved CV checkpoint to '{filepath}'")
