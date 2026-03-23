# 项目结构说明

本文档详细描述了 XiaoanNew 项目的目录结构及各文件的作用。

## 根目录

```
XiaoanNew/
├── app.py                  # Flask 后端 API 服务入口
├── mask_inference.py       # MaskRCNN 推理模块
├── hh.py                   # 临时测试脚本
├── rtmdet_tiny_8xb32-300e_coco.py  # RTMDet 配置文件
├── CHANGELOG.md            # 版本变更日志
├── .gitignore              # Git 忽略规则
└── .github/
    └── AGENTS.md           # 项目工作指南
```

### 核心文件说明

| 文件 | 功能 | 备注 |
|------|------|------|
| `app.py` | Flask 后端服务，提供 5 个 API 端点 | 入口文件，依赖 scripts/yolov8_seg_inference.py |
| `mask_inference.py` | MaskRCNN 实例分割推理器 | YOLOv8 不可用时的备用方案 |
| `CHANGELOG.md` | 记录版本变更历史 | 遵循 Keep a Changelog 规范 |

---

## scripts/ - 脚本工具目录

核心推理和实验脚本存放位置。

```
scripts/
├── yolov8_seg_inference.py     # YOLOv8-Seg 推理模块（核心）
├── yolov8_seg_batch.py         # 批量图片处理脚本
├── contrast_VLM_CV_test.py     # VLM+CV 联合测试（掩膜版）
├── contrast_VLM_CV_test_v2.py  # VLM+CV 联合测试（轮廓版）
├── contrast_VLM_test.py        # 纯 VLM 测试脚本
├── outdate/                    # 已废弃的旧版脚本
│   ├── VLM_test.py
│   ├── compare.py
│   ├── test_Vlm_v2.py
│   └── test_prompt.py
└── tool/                       # 数据处理工具
    ├── batch_rotate_images.py   # 批量旋转图片
    ├── copy_sample_view.py      # 复制采样视图
    ├── debug_viewer.py          # 调试可视化工具
    ├── sample_view.py           # 采样查看器
    ├── split_yes_dataset.py     # 数据集划分
    └── view_result_nolabel.py   # 无标签结果查看
```

### 核心模块详解

#### yolov8_seg_inference.py
- **类**: `YOLOv8SegInference`
- **功能**: YOLOv8 实例分割统一推理接口
- **主要方法**:
  - `predict()`: 返回结构化字典（含 mask、bbox、置信度）
  - `predict_memory()`: 返回 PNG 字节流（用于 API）
  - `predict_static_json()`: 返回 JSON 格式结果
- **依赖**: ultralytics, torch, PIL

#### contrast_VLM_CV_test_v2.py
- **功能**: VLM + CV 联合测试的主脚本
- **流程**: 图片 → YOLOv8分割 → 几何计算(IoU) → VLM判定 → CSV输出
- **输出**: experiment_outputs/ 目录下的 CSV 文件

---

## yolo/ - YOLO 训练目录

```
yolo/
├── train_yolov8_seg.py    # YOLOv8 实例分割训练脚本
└── data/
    └── coco/
        └── dataset.yaml   # 数据集配置文件
```

### train_yolov8_seg.py
- **功能**: 使用 Ultralytics 库训练 YOLOv8-Seg 模型
- **预训练模型**: yolov8l-seg.pt (Large)
- **输出路径**: work_dirs/yolov8l_seg/weights/best.pt
- **主要参数**: epochs=300, imgsz=640, batch=16

---

## weights/ - 模型权重目录

存放训练好的模型权重文件。

```
weights/
└── best.pt    # YOLOv8-Seg 训练好的最佳模型权重 (~88MB)
```

### 说明

- 模型使用 Ultralytics YOLOv8l-seg 架构
- 训练数据集：4 类实例分割（电动车、马路牙子、停车线、盲道）
- 建议使用 GPU 推理以获得最佳性能

---

## App_collected_dataset/ - 数据集目录

采集的数据存放位置。

```
App_collected_dataset/
├── record.md                  # 数据采集记录
├── bad/labels.txt             # 不合格样本标注
├── Campus_val/labels.txt      # 校园验证集
├── dark_label/labels.txt      # 暗光场景标注
├── split_data/labels.txt      # 划分后数据
├── test/                      # 测试数据
├── Xiaoan_datasets/           # 主数据集
│   ├── Readme.md              # 数据集说明
│   ├── yes_val/               # 合规样本
│   └── no_val/                # 不合规样本
├── yk_dark/                   # 夜间样本
├── yk01/labels.txt            # yk01 批次
├── zz01_rotate/labels.txt     # zz01 旋转增强
├── zz02_rotate/labels.txt     # zz02 旋转增强
├── zz03/                      # zz03 批次
└── zz03_rotate/labels.txt     # zz03 旋转增强
```

### labels.txt 格式
每行格式: `<文件名>, <标签>`
```
20260120_141520_image.jpg, yes
20260120_141535_image.jpg, no
```

---

## experiment_outputs/ - 实验输出

VLM+CV 联合测试的结果保存位置。

```
experiment_outputs/
├── all_experiments_summary.csv           # 所有实验汇总
├── qwen3-vl-30b_visual_prompting_v2.csv  # Qwen3-VL 30B 实验
├── qwen3-vl-30b-a3b_contours_iou_fix.csv # 轮廓版实验
├── results_qwen2.5-vl-72b-instruct_*.csv # Qwen2.5-VL 72B 结果
├── results_qwen3-vl-235b-a22b-instruct_*.csv
└── results_ernie-4.5-vl-424b-a47b_*.csv  # 文心 4.5 结果
```

### CSV 字段说明

| 字段 | 说明 |
|------|------|
| image_name | 图片文件名 |
| folder | 数据来源文件夹 |
| ground_truth | 真实标签 (yes/no) |
| vlm_result | VLM 预测结果 |
| composition_status | 构图合规性 |
| angle_status | 角度合规性 |
| distance_status | 距离合规性 |
| context_status | 环境合规性 |
| raw_response | VLM 原始响应 |

---

## 可视化输出目录

```
yolov8seg_visuals/          # 掩膜可视化结果
yolov8seg_visuals_contours/ # 轮廓可视化结果
yolov8seg_visuals_test/     # 测试可视化
yolov8seg_visuals_v2/       # V2 版本可视化
temp_processing/            # 临时处理文件
```

---

## 依赖关系图

```
app.py
  └── scripts/yolov8_seg_inference.py (主推理模块)
        └── ultralytics (YOLO)
  └── mask_inference.py (备用)
        └── torchvision (MaskRCNN)

scripts/contrast_VLM_CV_test_v2.py
  └── scripts/yolov8_seg_inference.py
  └── openai (VLM API)
  └── cv2 (OpenCV)

yolo/train_yolov8_seg.py
  └── ultralytics
  └── yolo/data/coco/dataset.yaml
```

---

## 配置文件优先级

1. **模型权重**: `weights/best.pt`
2. **数据集配置**: `yolo/data/coco/dataset.yaml`
3. **API 配置**: 各脚本顶部的 `CONFIG` 字典
