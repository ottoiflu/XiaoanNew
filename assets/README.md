# assets/ 静态资产说明

> 包含配置、提示词、模型权重。严禁存放可执行代码。

## 目录总览

```
assets/
├── configs/          # 实验参数配置 YAML
│   ├── default.yaml  # 默认实验配置模板
│   ├── scoring_optimized_cv_p4.yaml  # 优化版评分（当前最优，cv_enhanced_p4 用）
│   ├── scoring_default.yaml          # 默认评分配置
│   ├── scoring_optimized_vlm_p4.yaml # VLM 版优化评分
│   ├── template.yaml  # 空白实验配置模板
│   └── _archived/     # 已弃用配置
│
├── prompts/          # 提示词模板 YAML
│   ├── standard_p4.yaml ~ standard_p8.yaml   # 纯 VLM 提示词（在用）
│   ├── cv_enhanced_p4.yaml ~ cv_enhanced_p8.yaml  # CV 增强提示词（在用）
│   ├── standard_p4_1/2/3.yaml, cv_enhanced_p4_1/2/3.yaml  # 消融实验版
│   └── _archived/     # 已淘汰提示词（p2/p3/p9-p15 系列）
│
└── weights/          # 模型权重
    ├── best.pt       # 当前 YOLOv8-Seg 生产权重（~88MB）
    ├── best_v1.pt    # v1 旧权重备份
    ├── yolo26n.pt    # YOLO 轻量版（~5MB）
    ├── yolov8l-seg.pt  # YOLOv8l 官方预训练（~88MB）
    ├── yolov8x-seg.pt  # YOLOv8x 官方预训练（~139MB）
    └── archive/      # 旧版本权重备份
```

## configs/ 配置组织

| 类型 | 文件 | 用途 |
|------|------|------|
| 实验 | `default.yaml` | 通用实验参数 |
| 实验 | `template.yaml` | 空白模板 |
| 评分 | `scoring_optimized_cv_p4.yaml` | CV 增强版优化评分（weights + threshold + score_map） |
| 评分 | `scoring_optimized_vlm_p4.yaml` | 纯 VLM 版优化评分 |
| 评分 | `scoring_default.yaml` | 默认评分配置 |

实验配置通过 `run_contrast_batch_v2.py --config <path>` 或 `contrast_VLM_CV_test_v2.py --config <path>` 加载。评分配置由 `ScoringEngine` 从与实验同名的 scoring YAML 读取。

## prompts/ 组织规则

| 前缀 | 类型 | 说明 |
|------|------|------|
| `standard_p*` | 纯 VLM | 不包含 CV 几何数据，VLM 自行视觉判断 |
| `cv_enhanced_p*` | CV 增强 | 包含 IoU、重叠率等结构化 CV 数据注入 |

在用版本：p4-p8 系列（共 10 个文件，含 `_1/_2/_3` 消融变体）。
已归档：p2、p3、p9-p15 系列（移入 `_archived/`）。

## weights/ 说明

| 文件 | 来源 | 使用场景 |
|------|------|----------|
| `best.pt` | 自行训练/微调 | 生产推理，`INFERENCE_DEVICE` 指定设备 |
| `best_v1.pt` | 旧版训练 | 回归对比参考 |
| `yolo26n.pt` | 官方 | 移动端/轻量实验 |
| `yolov8l-seg.pt` | 官方 | 预训练权重，训练起点 |
| `yolov8x-seg.pt` | 官方 | 超大模型实验 |

## 注意事项

1. 运行中的评分配置（`scoring_optimized_cv_p4.yaml`）正在 benchmark 使用，修改前需确认不影响测试
2. 提示词修改后需同步更新 CHANGELOG
3. 权重路径由 `config/settings.py` 管理，不要硬编码
4. `_archived/` 目录仅供回溯参考，不再维护
