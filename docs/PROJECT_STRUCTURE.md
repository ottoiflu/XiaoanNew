# 项目结构说明

本文档详细描述了 XiaoanNew 项目的目录结构及各文件的作用。

## 根目录

```
XiaoanNew/
├── app.py                  # Flask 后端 API 服务入口
├── mask_inference.py       # MaskRCNN 推理模块（备用）
├── AGENTS.md               # 项目工作指南
├── CHANGELOG.md            # 版本变更日志
├── .gitignore              # Git 忽略规则
├── scripts/                # 脚本工具目录
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
| `mask_inference.py` | MaskRCNN 实例分割推理器 | 备用方案 |
| `AGENTS.md` | 项目指南和约定 | 最重要的参考文档 |
| `CHANGELOG.md` | 记录版本变更历史 | 遵循 Keep a Changelog 规范 |

---

## scripts/ - 脚本工具目录

核心推理和实验脚本存放位置。

```
scripts/
├── yolov8_seg_inference.py     # YOLOv8-Seg 推理模块（核心）
├── yolov8_seg_batch.py         # 批量图片处理脚本
├── contrast_VLM_CV_test.py     # VLM+CV 联合测试（掩膜版）
├── contrast_VLM_CV_test_v2.py  # VLM+CV 联合测试（轮廓版，主用）
├── contrast_VLM_test.py        # 纯 VLM 测试脚本
├── prompt_manager.py           # 提示词管理模块
├── prompts/                    # 提示词配置目录
│   └── cv_enhanced_p4.yaml     # 当前使用的提示词
├── outdate/                    # 已废弃的旧版脚本
└── tool/                       # 数据处理工具
    ├── debug_viewer.py         # 调试可视化工具
    ├── sample_view.py          # 采样查看器
    └── ...
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

#### prompt_manager.py
- **功能**: 提示词管理模块
- **主要函数**:
  - `load_prompt(name)`: 加载指定名称的提示词
  - `list_prompts()`: 列出所有可用提示词
- **命令行工具**: `python prompt_manager.py list/show/info`

#### contrast_VLM_CV_test_v2.py
- **功能**: VLM + CV 联合测试的主脚本
- **流程**: 图片 → YOLOv8分割 → 几何计算(IoU) → VLM判定 → CSV输出
- **输出**: test_outputs/exp_{timestamp}_{name}/ 目录

---

## Compliance_test_data/ - 测试数据集

```
Compliance_test_data/
├── yes_val/           # 正样本测试集 (50张，精简)
├── no_val/            # 负样本测试集 (50张，精简)
├── yes_val_all/       # 正样本完整集 (440张)
├── no_val_all/        # 负样本完整集 (421张)
├── positive_extra/    # 去重正样本
└── negative_extra/    # 去重负样本
```

每个测试目录需包含 `labels.txt` 文件，格式：`文件名,标签`

---

## test_outputs/ - 测试输出目录

```
test_outputs/
├── exp_{timestamp}_{name}/   # 每次实验独立目录
│   ├── {exp_name}.csv        # 详细结果
│   ├── all_experiments_summary.csv
│   └── visuals/              # 可视化图片
└── archived_experiments/     # 历史实验存档
```

---

## yolo/ - YOLO 训练目录

```
yolo/
├── train_yolov8_seg.py    # YOLOv8 实例分割训练脚本
└── data/
    └── coco/
        └── dataset.yaml   # 数据集配置文件
```

---

## weights/ - 模型权重

```
weights/
└── best.pt    # 训练好的 YOLOv8-Seg 模型
```

---

## 更新日志

- **2026-03-24**: 重构提示词管理，添加 prompt_manager.py
- **2026-03-24**: 重组测试输出目录结构
- **2026-03-23**: 移除 MMLab 依赖，简化项目结构
