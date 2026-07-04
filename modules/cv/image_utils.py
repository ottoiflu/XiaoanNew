"""图像处理工具

提供图像编码、Mask 合并、IoU 计算、轮廓可视化等共用功能。

2D 视觉标注升级：中文标签 + Set-of-Mark 数字 + 重叠高亮 + 主车加粗。
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────── 常量 ────────────────────────────

LABEL_MAP = {
    "Electric bike": "目标车",
    "Tactile paving": "盲道",
    "parking lane": "停车线",
    "Curb": "路缘",
}

COLOR_MAP = {
    "Electric bike": (0, 255, 0),    # 绿
    "Tactile paving": (0, 0, 255),   # 红
    "parking lane": (255, 255, 0),   # 黄
    "Curb": (0, 165, 255),           # 橙
    "default": (200, 200, 200),
}

# SoM 数字符号（①②③④...）
SOM_SYMBOLS = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]

# 中文字体路径（Noto Sans CJK）
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
_FONT_SIZE = 18
_SOM_FONT_SIZE = 16


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except (OSError, IOError):
        return ImageFont.load_default()


# ──────────────────────────── 核心可视化 ────────────────────────────


def draw_wireframe_visual(
    image_raw: np.ndarray,
    objects: list[dict],
    color_map: Optional[dict[str, tuple]] = None,
    label_map: Optional[dict[str, str]] = None,
    som_start: int = 1,
    real_contact: Optional[dict] = None,
) -> np.ndarray:
    """绘制 2D 视觉标注线框图

    功能：
    - 中文标签（白底+彩边框+黑色文字）
    - Set-of-Mark 数字（①②③...）
    - 主车（Electric bike）轮廓加粗
    - 主车与盲道/停车线的 2D 重叠高亮

    Args:
        image_raw: RGB image
        objects: [{label, mask, bbox, confidence, ...}, ...]
        color_map: 颜色映射覆盖
        label_map: 标签映射覆盖
        som_start: SoM 编号起始值

    Returns:
        RGB 图像 np.ndarray
    """
    colors = color_map or COLOR_MAP
    lbl_map = label_map or LABEL_MAP
    H, W = image_raw.shape[:2]

    # 1. 识别主车 mask（第一个 Electric bike 对象）
    main_bike_mask = None
    for obj in objects:
        if obj["label"] == "Electric bike":
            main_bike_mask = obj.get("mask")
            break

    # 2. 创建半透明叠加层 (RGBA) — 仅对真实 3D 接触做高亮
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    rc = real_contact or {"tactile": True, "parking_lane": True}  # 默认全高亮

    # 主车 ∩ 盲道 → 半透明红（仅当真实接触）
    if main_bike_mask is not None and rc.get("tactile", True):
        for obj in objects:
            if obj["label"] == "Tactile paving" and obj.get("mask") is not None:
                intersection = cv2.bitwise_and(
                    main_bike_mask.astype(np.uint8),
                    obj["mask"].astype(np.uint8),
                )
                if intersection.any():
                    overlay[intersection > 0] = (0, 0, 255, 102)

    # 主车 ∩ 停车线 → 半透明绿（仅当真实接触）
    if main_bike_mask is not None and rc.get("parking_lane", True):
        for obj in objects:
            if obj["label"] == "parking lane" and obj.get("mask") is not None:
                intersection = cv2.bitwise_and(
                    main_bike_mask.astype(np.uint8),
                    obj["mask"].astype(np.uint8),
                )
                if intersection.any():
                    overlay[intersection > 0] = (0, 255, 0, 102)

    # 3. 绘制轮廓（BGR）
    vis = cv2.cvtColor(image_raw.copy(), cv2.COLOR_RGB2BGR)
    for obj in objects:
        mask = obj.get("mask")
        if mask is None:
            continue
        label = obj["label"]
        color = colors.get(label, colors.get("default", (200, 200, 200)))
        line_width = 3 if label == "Electric bike" else 2
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, line_width)

    # 4. 融合半透明叠加层（直接 alpha 混合，避免 cv2.addWeighted 索引限制）
    for c in range(3):  # BGR channels
        vis[:, :, c] = np.where(
            overlay[:, :, 3] > 0,
            (vis[:, :, c].astype(float) * 0.6 + overlay[:, :, c].astype(float) * 0.4).astype(np.uint8),
            vis[:, :, c]
        )

    # 5. 用 PIL 添加文字标签和 SoM 数字
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(vis_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(_FONT_SIZE)
    som_font = _get_font(_SOM_FONT_SIZE)

    for idx, obj in enumerate(objects):
        bbox = obj.get("bbox")
        if bbox is None:
            continue
        try:
            bx1, by1, bx2, by2 = [int(float(v)) for v in bbox]
        except (ValueError, TypeError):
            continue
        label = obj["label"]

        som_num = idx + som_start
        som_symbol = SOM_SYMBOLS[som_num - 1] if som_num <= len(SOM_SYMBOLS) else str(som_num)
        color = colors.get(label, colors.get("default", (200, 200, 200)))
        cn_label = lbl_map.get(label, label)

        # SoM 数字（bbox 左上角）
        draw.text((bx1, by1 - 18), som_symbol, fill=(255, 255, 255), font=som_font)

        # 中文标签（bbox 上方）
        label_y = by1 - 36
        bbox_text = draw.textbbox((0, 0), cn_label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        # 白底
        draw.rectangle(
            [bx1, label_y, bx1 + tw + 4, label_y + th + 4],
            fill=(255, 255, 255),
            outline=color,
            width=2,
        )
        # 黑色文字
        draw.text((bx1 + 2, label_y + 2), cn_label, fill=(0, 0, 0), font=font)

    return np.array(pil_img)


# ──────────────────────────── 原有功能（不变） ────────────────────────────


def encode_image_to_base64(
    image: np.ndarray | str,
    max_size: tuple[int, int] = (768, 768),
    quality: int = 80,
) -> str:
    """编码图像为 base64 JPEG 字符串（保留原始功能）"""
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(image)
    pil_img.thumbnail(max_size, Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def combine_masks(
    objects: list[dict], target_label: str, image_size: tuple[int, int] | None = None
) -> np.ndarray | None:
    """合并指定类别的所有 mask（保留原有功能）"""
    masks = [obj["mask"] for obj in objects if obj["label"] == target_label and obj.get("mask") is not None]
    if not masks:
        return None
    combined = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        combined |= m.astype(bool)
    return combined


def calculate_iou_and_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    """计算两个 mask 的 IoU 和重叠率（保留原有功能）"""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = intersection / union if union > 0 else 0.0
    overlap = intersection / mask_a.sum() if mask_a.sum() > 0 else 0.0
    return float(iou), float(overlap)

# ──────────────────────────── 图片压缩 ────────────────────────────


def compress_image(img_bytes: bytes, max_size: int = 1024, quality: int = 85) -> bytes:
    """压缩图片：长边缩到 max_size（只缩不放），JPEG quality 压缩。返回 JPEG bytes。

    Args:
        img_bytes: 原始图片 bytes（任意格式）
        max_size: 长边阈值（像素），超过则等比例缩小
        quality: JPEG 压缩质量（1-100）

    Returns:
        压缩后的 JPEG bytes
    """
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = pil.size
    if max(W, H) > max_size:
        scale = max_size / max(W, H)
        pil = pil.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
