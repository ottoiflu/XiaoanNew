"""YOLOv8 实例分割训练脚本。"""

from pathlib import Path

from ultralytics import YOLO


def main():
    """加载预训练模型并启动实例分割训练。"""
    project_root = Path(__file__).resolve().parents[3]
    dataset_yaml = project_root / "data" / "yolo_seg_dataset" / "dataset.yaml"
    output_dir = project_root / "outputs" / "train_outputs"

    # YOLOv8x-seg 预训练权重，精度最高
    model = YOLO("yolov8x-seg.pt")

    model.train(
        data=str(dataset_yaml),
        epochs=500,
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        project=str(output_dir),
        name="yolov8x_seg_bike",
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        close_mosaic=10,
        amp=True,
        val=True,
        save=True,
        save_period=50,
        plots=True,
        verbose=True,
    )

    best_pt = output_dir / "yolov8x_seg_bike" / "weights" / "best.pt"
    print(f"\n训练完成! 最佳权重: {best_pt}")


if __name__ == "__main__":
    main()
