"""盲道压车检测模块

2D IoU + 深度图 override，判断车辆是否压在盲道上。
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from modules.cv.depth_assist import DepthAssist
from modules.cv.image_utils import calculate_iou_and_overlap, combine_masks

# 全局 DepthAssist 单例（只在需要时惰性初始化）
_DEPTH_ASSIST: DepthAssist | None = None


def _get_depth_assist() -> DepthAssist:
    """获取 DepthAssist 单例"""
    global _DEPTH_ASSIST
    if _DEPTH_ASSIST is None:
        _DEPTH_ASSIST = DepthAssist()
    return _DEPTH_ASSIST


def check_blind_lane(
    main_bike_mask: np.ndarray,
    objects: list[dict],
    pil_image: Image.Image,
) -> dict:
    """盲道压车检测：2D IoU + 深度图验证

    Args:
        main_bike_mask: 主车二值 mask (H, W) bool
        objects: [{label, mask, ...}, ...] YOLO 检测结果
        pil_image: PIL RGB 图像

    Returns:
        {
            "iou": float,          # 主车与盲道的 2D IoU
            "depth_diff": float | None,  # 主车下半区域与盲道的深度中位数差
            "override": bool,      # 是否触发 override（深度验证确认接触）
            "reason": str,         # 文字说明
        }
    """
    # 取盲道 mask
    blind_mask = combine_masks(objects, "Tactile paving")
    if main_bike_mask is None or blind_mask is None:
        return {
            "iou": 0.0,
            "depth_diff": None,
            "override": False,
            "reason": "无盲道或无主车 mask",
        }

    # 计算 2D IoU
    iou, _ = calculate_iou_and_overlap(main_bike_mask, blind_mask)

    # IoU < 阈值，无需深度验证
    if iou < 0.01:
        return {
            "iou": round(iou, 6),
            "depth_diff": None,
            "override": False,
            "reason": f"IoU={iou:.6f} < 0.01，无接触",
        }

    # 用深度图验证
    try:
        da = _get_depth_assist()
        depth_map = da.estimate(pil_image)

        # 取主车下半区域
        car_region = np.zeros_like(main_bike_mask, dtype=bool)
        rows = np.any(main_bike_mask, axis=1)
        if rows.any():
            top = max(0, int(main_bike_mask.shape[0] * 0.5))
            car_region[top:, :] = main_bike_mask[top:, :]

        car_depth = float(np.median(depth_map[car_region & main_bike_mask]))
        blind_depth = float(np.median(depth_map[blind_mask]))
        depth_diff = abs(car_depth - blind_depth)

        override = depth_diff < 0.1
        reason = (
            f"depth_diff={depth_diff:.6f}，{'触发' if override else '未触发'}override"
        )
        return {
            "iou": round(iou, 6),
            "depth_diff": round(depth_diff, 6),
            "override": override,
            "reason": reason,
        }
    except Exception as e:
        return {
            "iou": round(iou, 6),
            "depth_diff": None,
            "override": False,
            "reason": f"深度图估计失败: {e}",
        }
