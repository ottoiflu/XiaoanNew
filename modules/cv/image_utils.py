"""图像处理工具

提供图像编码、Mask 合并、IoU 计算、轮廓可视化等共用功能。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


def encode_image_to_base64(
    source: Union[str, Path, np.ndarray, Image.Image],
    max_size: tuple[int, int] = (768, 768),
    quality: int = 80,
) -> str:
    """将图像编码为 JPEG Base64 字符串

    Args:
        source: 文件路径 / numpy 数组 (RGB) / PIL Image
        max_size: 缩放上限
        quality: JPEG 压缩质量
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, np.ndarray):
        img = Image.fromarray(source)
    elif isinstance(source, Image.Image):
        img = source
    else:
        raise TypeError(f"不支持的图像类型: {type(source)}")

    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def calculate_iou_and_overlap(
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> tuple[float, float]:
    """计算两个 Mask 的 IoU 和 mask1 被 mask2 覆盖的比例"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    area1 = mask1.sum()

    iou = intersection / union if union > 0 else 0
    overlap = intersection / area1 if area1 > 0 else 0
    return round(iou, 4), round(overlap, 4)


def combine_masks(
    objects: list[dict],
    label_filter: str,
) -> Optional[np.ndarray]:
    """将特定类别的所有 mask 合并为一个"""
    combined = None
    for obj in objects:
        if obj["label"] == label_filter and obj.get("mask") is not None:
            if combined is None:
                combined = obj["mask"].copy()
            else:
                combined = np.logical_or(combined, obj["mask"])
    return combined


def draw_wireframe_visual(
    image_raw: np.ndarray,
    objects: list[dict],
    color_map: Optional[dict[str, tuple]] = None,
) -> np.ndarray:
    """绘制线框轮廓可视化图"""
    import cv2

    default_colors = {
        "Electric bike": (0, 255, 0),
        "parking lane": (255, 255, 0),
        "Curb": (0, 165, 255),
        "Tactile paving": (0, 0, 255),
        "default": (200, 200, 200),
    }
    colors = color_map or default_colors

    vis = cv2.cvtColor(image_raw.copy(), cv2.COLOR_RGB2BGR)
    for obj in objects:
        mask = obj.get("mask")
        if mask is None:
            continue
        label = obj["label"]
        color = colors.get(label, colors.get("default", (200, 200, 200)))
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
