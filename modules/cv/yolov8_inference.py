"""
YOLOv8-Seg 实例分割推理模块

该模块提供统一的 YOLOv8-Seg 推理接口，支持：
1. 单张图片推理
2. 批量图片推理
3. 内存流推理（用于后端 API）
4. 返回结构化检测结果和可视化图像

类别定义：
- 0: Electric bike (电动车)
- 1: Curb (马路牙子)
- 2: parking lane (停车线)
- 3: Tactile paving (盲道)

作者: Auto-generated
日期: 2026-01-20
"""

import base64
import io
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请安装 ultralytics: pip install ultralytics")


class YOLOv8SegInference:
    """
    YOLOv8-Seg 实例分割推理类

    提供与原 MaskRCNNInference 兼容的接口，方便无缝替换
    """

    # 类别映射（与训练时一致）
    CLASS_NAMES = {0: "Electric bike", 1: "Curb", 2: "parking lane", 3: "Tactile paving"}

    # 类别颜色定义（RGBA）
    COLOR_MAP = {
        0: (0, 255, 0),  # 电动车：绿色
        1: (255, 0, 255),  # 马路牙子：紫色
        2: (255, 255, 0),  # 停车线：黄色
        3: (255, 165, 0),  # 盲道：橙色
    }

    def __init__(self, weights_path: str, device: str = None, conf_threshold: float = 0.5):
        """
        初始化 YOLOv8-Seg 模型

        Args:
            weights_path: 模型权重路径 (.pt 文件)
            device: 推理设备 ('cuda:0', 'cpu' 等)，None 则自动选择
            conf_threshold: 置信度阈值
        """
        self.conf_threshold = conf_threshold

        # 自动选择设备
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[YOLOv8-Seg] 正在加载模型: {weights_path}")
        print(f"[YOLOv8-Seg] 使用设备: {self.device}")

        # 加载模型
        self.model = YOLO(weights_path)
        self.model.to(self.device)

        # 预热模型
        self._warmup()

        print("[YOLOv8-Seg] 模型加载完成！")

    def _warmup(self):
        """模型预热，提升首次推理速度"""
        print("[YOLOv8-Seg] 预热中...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        print("[YOLOv8-Seg] 预热完成")

    def predict(
        self, source: Union[str, np.ndarray, Image.Image, bytes], conf: float = None, iou: float = 0.7, imgsz: int = 640
    ) -> Dict:
        """
        执行推理，返回结构化结果

        Args:
            source: 输入图像（路径/numpy数组/PIL图像/字节流）
            conf: 置信度阈值，None则使用初始化时的值
            iou: NMS IOU阈值
            imgsz: 推理图像尺寸

        Returns:
            {
                "image_raw": np.ndarray,      # 原始图像 (H, W, 3) RGB
                "image_visual": np.ndarray,   # 可视化图像 (H, W, 3) RGB
                "objects": List[Dict],        # 检测对象列表
                "image_size": [H, W]          # 图像尺寸
            }
        """
        conf = conf if conf is not None else self.conf_threshold

        # 统一转换为 numpy RGB
        if isinstance(source, bytes):
            pil_img = Image.open(io.BytesIO(source)).convert("RGB")
            img_array = np.array(pil_img)
        elif isinstance(source, str):
            pil_img = Image.open(source).convert("RGB")
            img_array = np.array(pil_img)
        elif isinstance(source, Image.Image):
            img_array = np.array(source.convert("RGB"))
        elif isinstance(source, np.ndarray):
            # 确保是 RGB
            if len(source.shape) == 2:
                img_array = np.stack([source] * 3, axis=-1)
            elif source.shape[2] == 4:
                img_array = source[:, :, :3]
            else:
                img_array = source
        else:
            raise ValueError(f"不支持的输入类型: {type(source)}")

        H, W = img_array.shape[:2]

        # 执行推理
        results = self.model.predict(img_array, conf=conf, iou=iou, imgsz=imgsz, verbose=False)

        result = results[0]

        # 解析结果
        objects = []
        masks_combined = np.zeros((H, W, 4), dtype=np.uint8)  # RGBA

        if result.masks is not None and len(result.masks) > 0:
            boxes = result.boxes
            masks = result.masks.data.cpu().numpy()  # (N, H, W)

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf_score = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]

                # 获取 mask 并 resize 到原图尺寸
                mask = masks[i]
                if mask.shape != (H, W):
                    mask = self._resize_mask(mask, (H, W))

                # 计算面积占比
                area_ratio = float(mask.sum()) / (H * W)

                # 添加到对象列表
                objects.append(
                    {
                        "id": i + 1,
                        "category_id": cls_id,
                        "label": self.CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                        "bbox": [round(c, 2) for c in bbox],
                        "area_ratio": round(area_ratio, 4),
                        "confidence": round(conf_score, 3),
                        "mask": mask.astype(bool),  # 二值 mask
                    }
                )

                # 叠加到可视化图
                color = self.COLOR_MAP.get(cls_id, (128, 128, 128))
                masks_combined[mask > 0.5, 0] = color[0]
                masks_combined[mask > 0.5, 1] = color[1]
                masks_combined[mask > 0.5, 2] = color[2]
                masks_combined[mask > 0.5, 3] = 120  # 半透明

        # 生成可视化图像
        visual_img = self._draw_visualization(img_array.copy(), objects, masks_combined)

        return {"image_raw": img_array, "image_visual": visual_img, "objects": objects, "image_size": [H, W]}

    def _resize_mask(self, mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """将 mask resize 到目标尺寸"""
        from PIL import Image as PILImage

        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((target_size[1], target_size[0]), PILImage.NEAREST)
        return np.array(mask_resized) / 255.0

    def _draw_visualization(self, img: np.ndarray, objects: List[Dict], masks_combined: np.ndarray) -> np.ndarray:
        """
        绘制可视化结果

        Args:
            img: 原始图像 (H, W, 3) RGB
            objects: 检测对象列表
            masks_combined: 合并的 mask 图层 (H, W, 4) RGBA

        Returns:
            可视化图像 (H, W, 3) RGB
        """
        # 叠加 mask
        mask_rgb = masks_combined[:, :, :3]
        mask_alpha = masks_combined[:, :, 3:4] / 255.0
        img = (img * (1 - mask_alpha) + mask_rgb * mask_alpha).astype(np.uint8)

        # 转为 PIL 绘制文字和边框
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # 加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except Exception:
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
            except Exception:
                font = ImageFont.load_default()

        # 绘制每个对象
        for obj in objects:
            bbox = obj["bbox"]
            label = obj["label"]
            conf = obj["confidence"]
            cls_id = obj["category_id"]

            color = self.COLOR_MAP.get(cls_id, (128, 128, 128))

            # 绘制边框
            draw.rectangle(bbox, outline=color, width=3)

            # 绘制标签
            text = f"{label}: {conf:.2f}"
            text_bbox = draw.textbbox((bbox[0], bbox[1]), text, font=font)

            # 文字背景
            draw.rectangle(
                [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=(0, 0, 0, 180)
            )
            draw.text((bbox[0], bbox[1]), text, fill=(255, 255, 255), font=font)

        return np.array(pil_img)

    # ==================== 兼容接口 ====================

    def predict_memory(self, image_bytes: bytes) -> io.BytesIO:
        """
        兼容 MaskRCNNInference.predict_memory 接口

        输入字节流，返回 PNG 格式的透明叠加层

        Args:
            image_bytes: 图像字节流

        Returns:
            io.BytesIO 包含 PNG 格式的掩码叠加层
        """
        result = self.predict(image_bytes)

        # 生成透明叠加层
        H, W = result["image_size"]
        overlay = np.zeros((H, W, 4), dtype=np.uint8)

        for obj in result["objects"]:
            mask = obj["mask"]
            cls_id = obj["category_id"]
            color = self.COLOR_MAP.get(cls_id, (128, 128, 128))

            overlay[mask, 0] = color[0]
            overlay[mask, 1] = color[1]
            overlay[mask, 2] = color[2]
            overlay[mask, 3] = 180

        # 绘制边框和标签
        pil_overlay = Image.fromarray(overlay, mode="RGBA")
        draw = ImageDraw.Draw(pil_overlay)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        for obj in result["objects"]:
            bbox = obj["bbox"]
            label = obj["label"]
            conf = obj["confidence"]
            cls_id = obj["category_id"]
            color = self.COLOR_MAP.get(cls_id, (128, 128, 128)) + (255,)

            draw.rectangle(bbox, outline=color, width=3)
            draw.text((bbox[0], bbox[1]), f"{label}: {conf:.2f}", fill=(255, 255, 255, 255), font=font)

        # 输出
        output_buffer = io.BytesIO()
        pil_overlay.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return output_buffer

    def predict_static_json(self, image_bytes: bytes) -> Dict:
        """
        兼容 MaskRCNNInference.predict_static_json 接口

        Args:
            image_bytes: 图像字节流

        Returns:
            {
                "status": "success",
                "detections": [...],
                "mask_base64": "..."
            }
        """
        result = self.predict(image_bytes)

        # 生成可视化 Base64
        H, W = result["image_size"]
        overlay = np.zeros((H, W, 4), dtype=np.uint8)

        for obj in result["objects"]:
            mask = obj["mask"]
            cls_id = obj["category_id"]
            color = self.COLOR_MAP.get(cls_id, (128, 128, 128))

            overlay[mask, 0] = color[0]
            overlay[mask, 1] = color[1]
            overlay[mask, 2] = color[2]
            overlay[mask, 3] = 120

        # 绘制边框和标签
        pil_overlay = Image.fromarray(overlay, mode="RGBA")
        draw = ImageDraw.Draw(pil_overlay)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        except Exception:
            font = ImageFont.load_default()

        for obj in result["objects"]:
            bbox = obj["bbox"]
            label = obj["label"]
            conf = obj["confidence"]
            cls_id = obj["category_id"]
            color = self.COLOR_MAP.get(cls_id, (128, 128, 128)) + (255,)

            draw.rectangle(bbox, outline=color, width=4)
            draw.text((bbox[0] + 5, bbox[1] + 5), f"{label} {conf:.2f}", fill=(255, 255, 255, 255), font=font)

        # 编码为 Base64
        buffer = io.BytesIO()
        pil_overlay.save(buffer, format="PNG")
        mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # 构造返回数据（移除 mask 字段，因为无法 JSON 序列化）
        detections = []
        for obj in result["objects"]:
            detections.append(
                {
                    "category_id": obj["category_id"],
                    "label": obj["label"],
                    "score": obj["confidence"],
                    "box": obj["bbox"],
                    "area_ratio": obj["area_ratio"],
                }
            )

        return {"status": "success", "detections": detections, "mask_base64": mask_base64}

    def run(self, img_path: str, score_thr: float = 0.5) -> Dict:
        """
        兼容 BaseInstanceSegmentor.run 接口

        Args:
            img_path: 图像路径
            score_thr: 置信度阈值

        Returns:
            {
                "image_raw": np.ndarray,
                "image_visual": np.ndarray,
                "objects": List[Dict],
                "image_size": [H, W]
            }
        """
        result = self.predict(img_path, conf=score_thr)

        # 移除 mask 字段（与原接口一致）
        for obj in result["objects"]:
            if "mask" in obj:
                del obj["mask"]

        return result


# ==================== 便捷函数 ====================


def load_yolov8_seg(weights_path: str = None, device: str = None) -> YOLOv8SegInference:
    """
    加载 YOLOv8-Seg 模型的便捷函数

    Args:
        weights_path: 权重路径，None则使用默认路径
        device: 设备，None则自动选择

    Returns:
        YOLOv8SegInference 实例
    """
    if weights_path is None:
        # 默认权重路径
        weights_path = "/root/XiaoanNew/assets/weights/best.pt"

    return YOLOv8SegInference(weights_path, device=device)


if __name__ == "__main__":
    # 简单测试
    import sys

    model = load_yolov8_seg()

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        result = model.predict(img_path)

        print(f"图像尺寸: {result['image_size']}")
        print(f"检测到 {len(result['objects'])} 个对象:")
        for obj in result["objects"]:
            print(f"  - {obj['label']}: {obj['confidence']:.2f}, bbox={obj['bbox']}")

        # 保存可视化结果
        output_path = img_path.rsplit(".", 1)[0] + "_yolov8seg_result.jpg"
        Image.fromarray(result["image_visual"]).save(output_path)
        print(f"可视化结果已保存: {output_path}")
    else:
        print("用法: python yolov8_seg_inference.py <image_path>")
