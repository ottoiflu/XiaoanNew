# XiaoanNew - 共享单车停放检测系统

## 项目概述

本项目是一个基于深度学习的共享单车智能停放检测系统，主要功能包括：

1. 实时掩膜分割：检测电动车、停车线、马路牙子、盲道等目标
2. 停车合规性判断：结合 CV 检测结果与云端 VLM 进行综合判定
3. 数据采集与管理：支持通过 API 上传图片并记录标注

## 项目结构

```
XiaoanNew/
├── app.py                    # Flask 后端 API 服务入口
├── mask_inference.py         # MaskRCNN 推理模块（备用）
├── scripts/                  # 脚本工具目录
│   ├── yolov8_seg_inference.py    # YOLOv8-Seg 推理模块（主用）
│   ├── yolov8_seg_batch.py        # 批量处理脚本
│   ├── contrast_VLM_CV_test.py    # VLM+CV 联合测试脚本
│   └── contrast_VLM_CV_test_v2.py # 联合测试脚本（轮廓版）
├── yolo/                     # YOLO 训练相关
│   ├── train_yolov8_seg.py        # YOLOv8 实例分割训练脚本
│   └── data/coco/                 # YOLO 格式数据集
├── weights/                  # 模型权重目录
│   └── best.pt               # YOLOv8-Seg 训练好的模型
├── Compliance_test_data/     # 清洗后的测试数据集
│   ├── yes_val/              # 验证集正样本 (428张)
│   ├── no_val/               # 验证集负样本 (408张)
│   ├── positive/             # 扩展正样本 (1194张，去重)
│   └── negative/             # 扩展负样本 (258张，去重)
├── App_collected_dataset/    # App 采集的原始数据（待清洗）
├── experiment_outputs/       # 实验结果输出（CSV 格式）
└── test_outputs/             # 测试输出目录
    └── seg_visuals/          # 分割可视化结果
```

## 类别定义

| ID | 名称 | 描述 |
|----|------|------|
| 0 | Electric bike | 电动车 |
| 1 | Curb | 马路牙子 |
| 2 | parking lane | 停车线 |
| 3 | Tactile paving | 盲道 |

---

## API 接口详细文档

### 1. 数据采集上传

**端点**: `POST /api/collect/upload`

**请求格式**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |
| label | String | 否 | 数据标签（默认: unknown） |
| date | String | 否 | 日期（默认: 当天，格式: YYYY-MM-DD） |
| custom_path | String | 否 | 自定义存储路径 |
| ground_truth | String | 否 | 真实标注（yes/no） |

**响应示例**:
```json
{
  "status": "success",
  "path": "/root/XiaoanNew/App_collected_dataset/yes/2026-03-23/20260323_141520_image.jpg"
}
```

### 2. 实时掩膜分割

**端点**: `POST /api/segmentation/detect`

**请求格式**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |

**响应**: 返回 PNG 格式的透明掩码叠加层（`image/png`），客户端可直接叠加在原图上显示。

### 3. 静态图片分析

**端点**: `POST /api/segmentation/detect_static`

**请求格式**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |

**响应示例**:
```json
{
  "status": "success",
  "data": {
    "detections": [
      {
        "id": 1,
        "label": "Electric bike",
        "confidence": 0.92,
        "bbox": [100, 200, 400, 500],
        "area_ratio": 0.15
      }
    ],
    "mask_base64": "iVBORw0KGgo..."
  }
}
```

### 4. 停车检测

**端点**: `POST /api/test/check_parking`

**请求格式**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 |

**处理流程**:
1. 裁剪图片下方 30% 区域
2. 调用云端 OCR 识别车牌
3. AI 模型检测停车线/马路牙子/盲道
4. 综合判断停车合规性

**响应示例**:
```json
{
  "is_valid": true,
  "plate_number": "京A12345",
  "confidence": 0.95,
  "message": "规范停车 (检测到停车线)",
  "detections": {
    "parking_lane": true,
    "curb": false,
    "tactile_paving": false
  }
}
```

### 5. 健康检查

**端点**: `GET /api/health`

**响应示例**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "YOLOv8SegInference",
  "ocr_available": true
}
```

---

## 模型训练指南

### 数据集准备

1. **目录结构** (COCO 格式):
```
data/coco/
├── images/
│   ├── train2017/     # 训练图片
│   └── val2017/       # 验证图片
├── labels/
│   ├── train2017/     # 训练标注 (YOLO 格式)
│   └── val2017/       # 验证标注
└── dataset.yaml       # 数据集配置
```

2. **标注格式** (YOLO Segmentation):
```
# labels/train2017/image001.txt
# <class_id> <x1> <y1> <x2> <y2> ... (归一化多边形坐标)
0 0.123 0.456 0.234 0.567 0.345 0.678 ...
```

3. **dataset.yaml 配置**:
```yaml
path: /root/XiaoanNew/MMLab/mmyolo/data/coco
train: images/train2017
val: images/val2017

names:
  0: Electric bike
  1: Curb
  2: parking lane
  3: Tactile paving
```

### 训练配置

**默认训练参数** (yolo/train_yolov8_seg.py):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model | yolov8l-seg.pt | 预训练模型 |
| epochs | 300 | 训练轮数 |
| imgsz | 640 | 图像尺寸 |
| batch | 16 | 批大小 |
| optimizer | SGD | 优化器 |
| lr0 | 0.01 | 初始学习率 |
| lrf | 0.01 | 最终学习率因子 |
| momentum | 0.937 | 动量 |
| weight_decay | 0.0005 | 权重衰减 |
| warmup_epochs | 3 | 预热轮数 |
| close_mosaic | 10 | 最后 N 轮关闭 mosaic |
| amp | True | 混合精度训练 |

### 训练命令

```bash
cd /root/XiaoanNew/yolo
python train_yolov8_seg.py
```

### 模型权重

训练完成后，最佳模型保存在:
```
work_dirs/yolov8l_seg/weights/best.pt
```

---

## VLM 实验流程

### 实验架构

```
图片输入 → YOLOv8-Seg 分割 → 几何计算(IoU/重叠率) → VLM 判定 → 结果输出
                ↓                        ↓
           轮廓可视化图              结构化 JSON
```

### Prompt 设计原则

1. **角色设定**: 专业共享单车运维质检员
2. **输入描述**: 原始图片 + 可视化增强图 + CV 结构化数据
3. **判定维度**:
   - 图像构图合规性
   - 摆放角度合规性（与停车线垂直，偏差 < 30°）
   - 停放距离合规性（车身中点位置）
   - 路面环境合规性（盲道占用检测）

### 实验配置

**脚本**: scripts/contrast_VLM_CV_test_v2.py

| 配置项 | 说明 |
|--------|------|
| exp_name | 实验名称 |
| model | VLM 模型名称 |
| max_size | 图片最大尺寸 (768, 768) |
| quality | JPEG 压缩质量 (80) |
| prompt_id | 使用的 Prompt ID |
| conf_threshold | 分割置信度阈值 (0.6) |

### 几何计算

1. **IoU 计算**: 车辆 mask 与停车线 mask 的交并比
2. **重叠率**: 车辆被停车区域覆盖的比例
3. **盲道占用**: 车辆 mask 与盲道 mask 的重叠比例

### 输出格式

CSV 文件保存在 `experiment_outputs/` 目录，包含:
- 图片名称
- 真实标签
- VLM 预测结果
- 四维度状态判定
- 详细分析文本

---

## 部署运维指南

### 环境配置

1. **基础环境**:
```bash
# 创建 Conda 环境
conda create -n xiaoan python=3.10
conda activate xiaoan

# 安装 PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install ultralytics flask opencv-python openai pillow numpy
```

2. **模型准备**:
   - 下载 YOLOv8 预训练权重或使用自训练模型
   - 配置模型路径: `YOLO_SEG_WEIGHTS` 变量

3. **API Key 配置**:
   - OCR/VLM 服务需要有效的 API Key
   - 建议通过环境变量管理敏感信息

### 启动服务

```bash
cd /root/XiaoanNew
python app.py
```

服务默认监听 `0.0.0.0:5000`

### 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| Model not loaded | 模型权重路径错误 | 检查 YOLO_SEG_WEIGHTS 路径 |
| CUDA out of memory | GPU 显存不足 | 降低 batch size 或使用更小模型 |
| OCR 调用失败 | API Key 无效或超限 | 检查 API Key 和调用频率 |
| 推理速度慢 | 未使用 GPU | 确认 CUDA 环境正确配置 |

### 日志输出

服务启动时会打印:
- 存储根目录路径
- AI 引擎类型
- OCR 服务状态

---

## 代码规范

1. Python 代码遵循 PEP8 规范
2. 函数和类需包含中文文档字符串
3. 重要的配置参数应集中在脚本顶部的「配置区域」中定义
4. 模型权重路径使用绝对路径，便于跨目录调用

## 开发约定

1. 推理模块应提供统一接口：`predict()` 返回结构化字典，`predict_memory()` 返回 PNG 字节流
2. 新增功能时优先扩展 `YOLOv8SegInference` 类，保持接口向后兼容
3. 实验脚本命名格式：`{功能}_{模型}_{版本}.py`
4. 实验输出 CSV 命名格式：`results_{model}_{size}_q{quality}_p{prompt_id}_detailed.csv`

## 注意事项

1. 模型权重文件较大，不纳入 Git 版本控制
2. API Key 等敏感信息应通过环境变量或独立配置文件管理
3. 实验输出 CSV 保留作为性能对比参考，修改实验配置前先备份
4. 云端 VLM 调用有频率限制，批量测试时注意控制并发数
