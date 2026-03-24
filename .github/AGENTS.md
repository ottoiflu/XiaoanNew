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
├── pyproject.toml            # 项目配置（ruff / 构建 / 元数据）
├── requirements.txt          # pip 依赖清单
├── .env.example              # 环境变量模板
│
├── modules/                  # 核心 Python 包
│   ├── config/               #   统一配置管理
│   │   └── settings.py
│   ├── vlm/                  #   VLM 客户端与响应解析
│   │   ├── client.py
│   │   └── parser.py
│   ├── cv/                   #   计算机视觉推理
│   │   ├── yolov8_inference.py
│   │   ├── mask_inference.py
│   │   └── image_utils.py
│   ├── experiment/           #   实验管理
│   │   ├── config.py
│   │   ├── io.py
│   │   ├── metrics.py
│   │   └── scoring.py
│   ├── prompt/               #   提示词管理
│   │   └── manager.py
│   └── train/                #   训练脚本
│       └── yolo/
│
├── scripts/                  # 可执行实验脚本
│   ├── contrast_VLM_CV_test_v2.py  # CV+VLM 联合测试（主用）
│   ├── contrast_VLM_test.py        # 纯 VLM 测试
│   ├── contrast_VLM_CV_test.py     # 旧版 CV+VLM 测试
│   ├── yolov8_seg_batch.py         # YOLOv8 批量推理
│   └── tool/                       # 辅助工具脚本
│
├── assets/                   # 静态资源
│   ├── configs/              #   实验配置 YAML
│   ├── prompts/              #   提示词配置 YAML
│   └── weights/              #   模型权重（best.pt）
│
├── data/                     # 数据集
│   ├── Compliance_test_data/ #   测试数据集
│   │   ├── yes_val/          #     正样本测试集（50 张）
│   │   ├── no_val/           #     负样本测试集（50 张）
│   │   ├── yes_val_all/      #     正样本完整集（440 张）
│   │   └── no_val_all/       #     负样本完整集（421 张）
│   └── App_collected_dataset/#   采集数据
│
├── outputs/                  # 实验输出
│   ├── test_outputs/         #   实验结果目录
│   └── experiment_outputs/   #   汇总 CSV 输出
│
├── docs/                     # 项目文档
└── .github/                  # GitHub 配置
```

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

1. Python 代码遵循 PEP8 规范，使用 ruff 作为唯一的 lint + format 工具
   - 提交前必须通过 `ruff check .` 和 `ruff format --check .`
   - 配置集中在 pyproject.toml `[tool.ruff]` 段
   - 禁止使用 bare except，必须捕获具体异常类型
2. 函数和类需包含中文文档字符串
3. 重要的配置参数应集中在脚本顶部的「配置区域」中定义
4. 模型权重路径使用绝对路径，便于跨目录调用

## 实验配置系统



v1.0.0 引入了解耦的实验配置系统，支持通过 YAML 文件管理实验参数。



### 目录结构

```
XiaoanNew/
├── assets/
│   ├── configs/              # 实验配置 YAML
│   │   ├── default.yaml
│   │   └── *.yaml
│   └── prompts/              # 提示词配置 YAML
│       └── cv_enhanced_p4.yaml
└── modules/
    ├── experiment/config.py  # 配置管理模块
    └── prompt/manager.py     # 提示词管理模块
```



### 使用方式



```bash

# 列出可用配置

python scripts/contrast_VLM_CV_test_v2.py --list-configs



# 使用配置文件运行实验

python scripts/contrast_VLM_CV_test_v2.py --config assets/configs/default.yaml



# 无参数运行（使用脚本内置默认配置）

python scripts/contrast_VLM_CV_test_v2.py

```



### 配置字段说明



| 字段 | 类型 | 说明 |

|------|------|------|

| exp_name | string | 实验名称，用于生成输出目录 |

| model | string | VLM 模型路径 |

| prompt_id | string | 提示词文件名（无扩展名） |

| max_size | [int, int] | 图像最大尺寸 |

| quality | int | JPEG 压缩质量 |

| data_folders | list | 数据目录列表 |

| max_workers | int | 并发线程数 |



### 配置备份



每次实验运行时，配置文件会自动备份到实验输出目录：

```

outputs/test_outputs/exp_{timestamp}_{name}/

├── experiment_config.yaml    # 配置快照

├── {exp_name}.csv           # 实验结果

└── visuals/                 # 可视化输出

```



---


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
