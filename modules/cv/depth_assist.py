"""深度辅助 3D 接触判断模块

加载 Depth Anything V2 估计深度图，判断主车与盲道/停车线是否真实 3D 接触。

用法：
    from modules.cv.depth_assist import DepthAssist, check_contact
    da = DepthAssist(device="cuda:0")
    depth_map = da.estimate(image_pil)
    contact = check_contact(main_bike_mask, blind_mask, parking_mask, depth_map)
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

# 延迟加载（避免模块导入时加载模型）
_DEPTH_PIPE = None
_DEVICE = None


class DepthAssist:
    """深度估计辅助器（单例模式，模型只加载一次）"""

    MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"

    def __init__(self, device: str = "cuda:0"):
        global _DEPTH_PIPE, _DEVICE
        if _DEPTH_PIPE is not None:
            return
        _DEVICE = device
        from transformers import pipeline

        print(f"[DepthAssist] 加载模型: {self.MODEL_NAME} on {device}")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        _DEPTH_PIPE = pipeline(
            "depth-estimation",
            model=self.MODEL_NAME,
            device=device,
            torch_dtype=dtype,
        )
        print("[DepthAssist] 加载完成")

    def estimate(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """对 RGB 图像执行深度估计，返回归一化 [0, 1] 深度图 (H, W)"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        result = _DEPTH_PIPE(image)
        depth = np.array(result["depth"], dtype=np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        return depth


def _get_ground_contact_region(mask: np.ndarray, depth: np.ndarray, bottom_ratio: float = 0.3) -> np.ndarray | None:
    """从主车 mask 中提取"接地部分"——mask 底部 bottom_ratio 区域 + 深度值最大（最远/贴近地面）的像素

    取 mask 垂直方向下半部分，再取其中深度值最大的 50% 像素（最远、最贴近地面）。
    """
    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        return None
    h = mask.shape[0]
    # 取 mask 垂直范围下半部分
    y_min, y_max = ys.min(), ys.max()
    y_thresh = y_min + (y_max - y_min) * (1 - bottom_ratio)
    bottom_mask = mask.copy().astype(bool)
    # 只保留下半部分的像素
    y_indices = np.arange(mask.shape[0])[:, None]
    bottom_mask = bottom_mask & (y_indices > y_thresh)
    if bottom_mask.sum() < 10:
        return None
    # 从底部像素中取深度值最大的 50%
    bottom_depths = depth[bottom_mask]
    threshold = np.percentile(bottom_depths, 50)
    contact_mask = bottom_mask.copy()
    contact_mask[contact_mask] = depth[contact_mask] >= threshold
    return contact_mask.astype(np.uint8)


def check_contact(
    main_bike_mask: np.ndarray | None,
    blind_mask: np.ndarray | None,
    parking_mask: np.ndarray | None,
    depth: np.ndarray,
    depth_threshold: float = 0.08,
) -> dict:
    """判断主车与盲道/停车线是否真实 3D 接触

    Args:
        main_bike_mask: 主车 mask (H, W) bool
        blind_mask: 盲道 mask (H, W) bool
        parking_mask: 停车线 mask (H, W) bool
        depth: 深度图 (H, W) float32, [0, 1]
        depth_threshold: 深度差阈值，低于此值视为同一平面

    Returns:
        {"tactile": bool, "parking_lane": bool}
    """
    result = {"tactile": False, "parking_lane": False}

    if main_bike_mask is None:
        return result

    ground = _get_ground_contact_region(main_bike_mask, depth)
    if ground is None or ground.sum() < 10:
        return result

    # 主车接地部分的平均深度
    ground_depth = depth[ground > 0].mean()

    # 检查盲道接触
    if blind_mask is not None and blind_mask.sum() > 10:
        blind_depth = depth[blind_mask > 0].mean()
        depth_diff = abs(ground_depth - blind_depth)
        # 深度差 < 阈值 → 同一平面 → 真实接触
        result["tactile"] = depth_diff < depth_threshold

    # 检查停车线接触
    if parking_mask is not None and parking_mask.sum() > 10:
        parking_depth = depth[parking_mask > 0].mean()
        depth_diff = abs(ground_depth - parking_depth)
        result["parking_lane"] = depth_diff < depth_threshold

    return result