# 项目结构说明

本文档详细描述了 XiaoanNew 项目的目录结构及各文件的作用。

## 根目录

```
XiaoanNew/
├── app.py                  # Flask 后端 API 服务入口
├── mask_inference.py       # MaskRCNN 推理模块（备用）
├── requirements.txt        # pip 依赖清单
├── pyproject.toml          # 项目配置（PEP 621）
├── .env.example            # 环境变量模板
├── AGENTS.md               # 项目工作指南
├── CHANGELOG.md            # 版本变更日志
├── config/                 # Python 配置模块
├── configs/                # 实验配置 YAML
├── prompts/                # 提示词配置 YAML
├── scripts/                # 脚本工具目录
├── utils/                  # 公共工具模块
├── yolo/                   # YOLO 训练相关
├── Compliance_test_data/   # 清洗后的测试数据集
├── test_outputs/           # 测试输出目录
├── weights/                # 模型权重文件
├── docs/                   # 项目文档
└── App_collected_dataset/  # 原始采集数据
```

### 核心文件说明

| 文件 | 功能 | 备注 |
|------|------|------|
| `app.py` | Flask 后端服务，提供 5 个 API 端点 | 入口文件 |
| `requirements.txt` | pip 依赖声明 | `pip install -r requirements.txt` |
| `pyproject.toml` | 现代 Python 项目配置 | 包含 black/isort/mypy 配置 |
| `.env.example` | 环境变量模板 | 复制为 .env 并填入实际值 |

---

## config/ - Python 配置模块

统一的环境变量和配置加载模块。

```
config/
├── __init__.py
└── settings.py    # 统一配置加载器
```

**使用方式**:
```python
from config.settings import settings

api_key = settings.VLM_API_KEY
model = settings.VLM_MODEL
```

---

## configs/ - 实验配置目录

实验参数的 YAML 配置文件。

```
configs/
├── default.yaml              # 默认实验配置
└── test_config_system.yaml   # 测试用配置
```

**运行方式**:
```bash
python scripts/contrast_VLM_CV_test_v2.py --config configs/default.yaml
```

---

## prompts/ - 提示词配置目录

VLM 提示词的 YAML 配置文件。

```
prompts/
└── cv_enhanced_p4.yaml    # 当前使用的提示词
```

**使用方式**:
```python
from scripts.prompt_manager import load_prompt
prompt = load_prompt("cv_enhanced_p4")
```

---

## utils/ - 公共工具模块

通用工具函数和类。

```
utils/
├── __init__.py
└── metrics.py    # 评估指标计算模块
```

**主要功能**:
- `calculate_metrics()`: 计算准确率、召回率、F1 等指标
- `print_metrics_report()`: 输出格式化的评估报告

---

## scripts/ - 脚本工具目录

核心推理和实验脚本存放位置。

```
scripts/
├── yolov8_seg_inference.py     # YOLOv8-Seg 推理模块（核心）
├── yolov8_seg_batch.py         # 批量图片处理脚本
├── contrast_VLM_CV_test_v2.py  # VLM+CV 联合测试（主用）
├── experiment_config.py        # 实验配置管理模块
├── prompt_manager.py           # 提示词管理模块
├── outdate/                    # 已废弃的旧版脚本
└── tool/                       # 数据处理工具
```

### 核心模块详解

#### yolov8_seg_inference.py
- **类**: `YOLOv8SegInference`
- **功能**: YOLOv8 实例分割统一推理接口
- **主要方法**:
  - `predict()`: 返回结构化字典
  - `predict_memory()`: 返回 PNG 字节流

#### experiment_config.py
- **功能**: 实验配置管理
- **主要函数**:
  - `load_config(path)`: 加载 YAML 配置
  - `save_config(config, path)`: 保存配置到实验目录

#### contrast_VLM_CV_test_v2.py
- **功能**: VLM + CV 联合测试主脚本
- **命令行参数**:
  - `--config`: 指定配置文件
  - `--list-configs`: 列出可用配置
- **输出**: test_outputs/exp_{timestamp}_{name}/

---

## Compliance_test_data/ - 测试数据集

```
Compliance_test_data/
├── yes_val/           # 正样本测试集 (50张)
├── no_val/            # 负样本测试集 (50张)
├── yes_val_all/       # 正样本完整集 (440张)
└── no_val_all/        # 负样本完整集 (421张)
```

每个测试目录需包含 `labels.txt` 文件，格式：`文件名,标签`

---

## test_outputs/ - 测试输出目录

```
test_outputs/
├── exp_{timestamp}_{name}/   # 每次实验独立目录
│   ├── experiment_config.yaml  # 配置快照
│   ├── {exp_name}.csv          # 详细结果
│   └── visuals/                # 可视化图片
└── archived_experiments/       # 历史实验存档
```

---

## yolo/ - YOLO 训练目录

```
yolo/
├── train_yolov8_seg.py    # YOLOv8 实例分割训练脚本
└── data/coco/
    └── dataset.yaml       # 数据集配置文件
```

---

## weights/ - 模型权重

```
weights/
└── best.pt    # 训练好的 YOLOv8-Seg 模型
```

---

## 更新日志

- **2026-03-24**: 重组配置目录，将 configs/ 和 prompts/ 移至根目录
- **2026-03-24**: 添加 config/settings.py 统一配置管理
- **2026-03-24**: 添加 utils/metrics.py 评估指标模块
- **2026-03-24**: 添加 requirements.txt 和 pyproject.toml
- **2026-03-24**: 重构提示词管理，添加 prompt_manager.py
- **2026-03-23**: 移除 MMLab 依赖，简化项目结构
