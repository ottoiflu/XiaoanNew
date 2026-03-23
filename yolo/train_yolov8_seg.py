"""
YOLOv8 实例分割训练脚本 (使用 Ultralytics 官方库)
"""
from ultralytics import YOLO

def main():
    # 加载预训练的 YOLOv8-Seg 模型 (Large 版本)
    # 可选: yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt
    model = YOLO('yolov8l-seg.pt')
    
    # 训练模型
    results = model.train(
        data='data/coco/dataset.yaml',  # 数据集配置文件
        epochs=300,                      # 训练轮数
        imgsz=640,                       # 图像尺寸
        batch=16,                        # batch size
        device=0,                        # GPU 设备
        workers=8,                       # 数据加载线程数
        project='work_dirs',             # 项目目录
        name='yolov8l_seg',              # 实验名称
        exist_ok=True,                   # 覆盖已有实验
        pretrained=True,                 # 使用预训练权重
        optimizer='SGD',                 # 优化器
        lr0=0.01,                        # 初始学习率
        lrf=0.01,                        # 最终学习率 (lr0 * lrf)
        momentum=0.937,                  # SGD 动量
        weight_decay=0.0005,             # 权重衰减
        warmup_epochs=3,                 # 预热轮数
        warmup_momentum=0.8,             # 预热动量
        warmup_bias_lr=0.1,              # 预热 bias 学习率
        close_mosaic=10,                 # 最后 N 轮关闭 mosaic
        amp=True,                        # 混合精度训练
        val=True,                        # 训练期间验证
        save=True,                       # 保存检查点
        save_period=50,                  # 每 N 轮保存一次
        plots=True,                      # 保存训练曲线图
        verbose=True,                    # 详细输出
    )
    
    print("\n训练完成!")
    print(f"最佳模型保存在: work_dirs/yolov8l_seg/weights/best.pt")

if __name__ == '__main__':
    main()
