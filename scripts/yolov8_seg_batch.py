#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8-Seg 批量图片处理脚本

用法:
    python yolov8_seg_batch.py <input_folder> [options]

示例:
    python yolov8_seg_batch.py ./images
    python yolov8_seg_batch.py ./images --output ./results
    python yolov8_seg_batch.py ./images --conf 0.6 --workers 4

作者: Auto-generated
日期: 2026-01-20
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# 添加脚本目录到路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from PIL import Image

from modules.cv.yolov8_inference import load_yolov8_seg


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8-Seg 批量图片实例分割",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
类别说明:
  0: Electric bike  (电动车)
  1: Curb          (马路牙子)
  2: parking lane  (停车线)
  3: Tactile paving (盲道)

输出结构:
  <output_folder>/
  ├── visuals/           可视化结果图
  ├── masks/             透明掩码 PNG (可选)
  ├── detections.json    所有检测结果汇总
  └── summary.csv        统计摘要
        """,
    )

    parser.add_argument("input", type=str, help="输入图片文件夹路径")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="输出文件夹路径（默认：<输入文件夹>_yolov8seg_results）"
    )
    parser.add_argument(
        "-w", "--weights", type=str, default="/root/XiaoanNew/assets/weights/best.pt", help="模型权重路径"
    )
    parser.add_argument("-c", "--conf", type=float, default=0.5, help="置信度阈值 (默认: 0.5)")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IOU阈值 (默认: 0.7)")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--device", type=str, default=None, help="推理设备 (默认: auto)")
    parser.add_argument("--save-mask", action="store_true", help="保存透明掩码叠加层 PNG")
    parser.add_argument("--no-visual", action="store_true", help="不保存可视化图片")
    parser.add_argument("--workers", type=int, default=1, help="并行处理线程数 (默认: 1，GPU推理建议为1)")
    parser.add_argument("--recursive", action="store_true", help="递归处理子文件夹")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    return parser.parse_args()


def find_images(folder: str, recursive: bool = False) -> list:
    """查找文件夹中的所有图片"""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []

    folder_path = Path(folder)

    if recursive:
        for ext in extensions:
            images.extend(folder_path.rglob(f"*{ext}"))
            images.extend(folder_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            images.extend(folder_path.glob(f"*{ext}"))
            images.extend(folder_path.glob(f"*{ext.upper()}"))

    return sorted(set(images))


def process_single_image(args_tuple):
    """处理单张图片（供多线程使用）"""
    img_path, model, conf, iou, imgsz, output_visual_dir, output_mask_dir, save_mask, save_visual = args_tuple

    try:
        # 执行推理
        result = model.predict(str(img_path), conf=conf, iou=iou, imgsz=imgsz)

        stem = img_path.stem
        H, W = result["image_size"]
        objects = result["objects"]

        # 保存可视化图片
        visual_path = None
        if save_visual:
            visual_path = output_visual_dir / f"{stem}.jpg"
            visual_img = Image.fromarray(result["image_visual"])
            visual_img.save(str(visual_path), quality=95)

        # 保存透明掩码
        mask_path = None
        if save_mask:
            mask_path = output_mask_dir / f"{stem}_mask.png"
            mask_buffer = model.predict_memory(open(str(img_path), "rb").read())
            with open(mask_path, "wb") as f:
                f.write(mask_buffer.getvalue())

        # 构造返回数据
        detection_data = {
            "image_name": img_path.name,
            "image_path": str(img_path.absolute()),
            "image_size": {"width": W, "height": H},
            "num_detections": len(objects),
            "detections": [],
        }

        for obj in objects:
            detection_data["detections"].append(
                {
                    "id": obj["id"],
                    "category_id": obj["category_id"],
                    "label": obj["label"],
                    "confidence": obj["confidence"],
                    "bbox": obj["bbox"],
                    "area_ratio": obj["area_ratio"],
                }
            )

        return {"status": "success", "image_name": img_path.name, "data": detection_data}

    except Exception as e:
        return {"status": "error", "image_name": img_path.name, "error": str(e)}


def main():
    args = parse_args()

    # 检查输入文件夹
    if not os.path.isdir(args.input):
        print(f"❌ 错误: '{args.input}' 不是有效的文件夹")
        sys.exit(1)

    # 查找图片
    images = find_images(args.input, args.recursive)
    if len(images) == 0:
        print(f"❌ 错误: 在 '{args.input}' 中未找到图片文件")
        sys.exit(1)

    # 确定输出路径
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.input).parent / f"{Path(args.input).name}_yolov8seg_results"

    output_visual_dir = output_dir / "visuals"
    output_mask_dir = output_dir / "masks"

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_visual:
        output_visual_dir.mkdir(exist_ok=True)
    if args.save_mask:
        output_mask_dir.mkdir(exist_ok=True)

    # 打印配置
    print("=" * 60)
    print("🚀 YOLOv8-Seg 批量处理")
    print("=" * 60)
    print(f"📁 输入文件夹: {args.input}")
    print(f"📁 输出文件夹: {output_dir}")
    print(f"🔧 权重文件: {args.weights}")
    print(f"⚙️  置信度阈值: {args.conf}")
    print(f"📷 发现图片数: {len(images)}")
    print(f"🔄 并行线程数: {args.workers}")
    print("-" * 60)

    # 加载模型
    model = load_yolov8_seg(args.weights, device=args.device)
    model.conf_threshold = args.conf

    # 准备任务参数
    tasks = [
        (
            img,
            model,
            args.conf,
            args.iou,
            args.imgsz,
            output_visual_dir,
            output_mask_dir,
            args.save_mask,
            not args.no_visual,
        )
        for img in images
    ]

    # 统计数据
    all_results = []
    success_count = 0
    error_count = 0
    total_detections = 0
    class_counts = {"Electric bike": 0, "Curb": 0, "parking lane": 0, "Tactile paving": 0}

    # 批量处理
    print("🔍 开始批量推理...")

    if args.workers == 1:
        # 单线程处理（推荐用于 GPU）
        for task in tqdm(tasks, desc="处理进度"):
            result = process_single_image(task)
            all_results.append(result)

            if result["status"] == "success":
                success_count += 1
                data = result["data"]
                total_detections += data["num_detections"]
                for det in data["detections"]:
                    if det["label"] in class_counts:
                        class_counts[det["label"]] += 1
            else:
                error_count += 1
                if args.verbose:
                    print(f"  ⚠️ 错误: {result['image_name']} - {result['error']}")
    else:
        # 多线程处理
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_image, task): task[0] for task in tasks}

            for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
                result = future.result()
                all_results.append(result)

                if result["status"] == "success":
                    success_count += 1
                    data = result["data"]
                    total_detections += data["num_detections"]
                    for det in data["detections"]:
                        if det["label"] in class_counts:
                            class_counts[det["label"]] += 1
                else:
                    error_count += 1

    # 保存汇总 JSON
    json_output_path = output_dir / "detections.json"
    json_data = {
        "metadata": {
            "processed_time": datetime.now().isoformat(),
            "model_weights": args.weights,
            "conf_threshold": args.conf,
            "total_images": len(images),
            "successful": success_count,
            "errors": error_count,
        },
        "results": [r["data"] for r in all_results if r["status"] == "success"],
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # 保存 CSV 摘要
    csv_output_path = output_dir / "summary.csv"
    with open(csv_output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image_name", "status", "num_detections", "Electric bike", "Curb", "parking lane", "Tactile paving"]
        )

        for result in all_results:
            if result["status"] == "success":
                data = result["data"]
                counts = {"Electric bike": 0, "Curb": 0, "parking lane": 0, "Tactile paving": 0}
                for det in data["detections"]:
                    if det["label"] in counts:
                        counts[det["label"]] += 1
                writer.writerow(
                    [
                        result["image_name"],
                        "success",
                        data["num_detections"],
                        counts["Electric bike"],
                        counts["Curb"],
                        counts["parking lane"],
                        counts["Tactile paving"],
                    ]
                )
            else:
                writer.writerow([result["image_name"], "error", 0, 0, 0, 0, 0])

    # 输出统计报告
    print("-" * 60)
    print("📊 处理统计:")
    print("-" * 60)
    print(f"  总图片数: {len(images)}")
    print(f"  ✅ 成功: {success_count}")
    print(f"  ❌ 失败: {error_count}")
    print(f"  🎯 总检测数: {total_detections}")
    print("-" * 60)
    print("📈 类别统计:")
    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count}")
    print("-" * 60)
    print("💾 输出文件:")
    print(f"  📋 检测汇总: {json_output_path}")
    print(f"  📊 CSV摘要: {csv_output_path}")
    if not args.no_visual:
        print(f"  🖼️ 可视化图片: {output_visual_dir}/")
    if args.save_mask:
        print(f"  🎭 透明掩码: {output_mask_dir}/")
    print("=" * 60)
    print("✅ 批量处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
