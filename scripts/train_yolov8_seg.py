"""YOLOv8-Seg 5 类实例分割训练脚本（Electric bike / Curb / parking lane / Tactile paving / Green belt）。

基于 yolov8l-seg.pt COCO 预训练起点，imgsz=1024，batch=24，AdamW，cosine schedule。
数据集：/root/XiaoanNew/data/yolo/dataset_v5/dataset.yaml
产物：/root/XiaoanNew/train_v5_out/yolov8l_seg_v5/weights/best.pt
  （输出落在根 overlay / 上，因 /root/autodl-tmp 数据盘已 100% 满；
   原 outputs 软链指向该满盘，故改写到根 overlay 的真实目录）
"""

from pathlib import Path

from ultralytics import YOLO

PROJECT = Path("/root/XiaoanNew")


def main() -> None:
    model = YOLO(PROJECT / "assets/weights/yolov8l-seg.pt")

    results = model.train(
        data=PROJECT / "data/yolo/dataset_v5/dataset.yaml",
        epochs=200,
        imgsz=1024,
        batch=24,  # 若 OOM 降到 8；4090 48G + amp 应可承受
        device=0,
        workers=8,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # 损失权重
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # 数据增强
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,  # 分割任务 mixup mask 混乱，关闭
        copy_paste=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,  # 提一点，增强暗光鲁棒性
        # 混合精度（默认开，显式确认不关）
        amp=True,
        # 收敛控制
        close_mosaic=15,  # 最后 15 轮关 mosaic 稳定收敛
        patience=30,  # 早停
        label_smoothing=0.1,  # 小数据集防过拟合
        cache="ram",  # 911 张全缓存加速
        # 输出（写在根 overlay 上，避开已满的 autodl-tmp 数据盘）
        project=str(PROJECT / "train_v5_out"),
        name="yolov8l_seg_v5",
        exist_ok=False,
    )
    print("=== TRAIN DONE ===")
    print(results)


if __name__ == "__main__":
    main()
