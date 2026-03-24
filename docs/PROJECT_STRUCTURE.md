# 项目结构说明

本文档详细描述 XiaoanNew 项目的目录结构及各文件的作用。

## 目录总览

```
XiaoanNew/
├── app.py                    # Flask 后端 API 服务入口
├── pyproject.toml            # 项目配置（ruff/构建/元数据）
├── requirements.txt          # pip 依赖清单
├── .env.example              # 环境变量模板
├── AGENTS.md                 # 项目工作指南
├── CHANGELOG.md              # 版本变更日志
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
│   ├── contrast_VLM_CV_test_v2.py   # CV+VLM 联合测试（主用）
│   ├── contrast_VLM_test.py         # 纯 VLM 测试
│   ├── contrast_VLM_CV_test.py      # 旧版 CV+VLM 测试
│   ├── yolov8_seg_batch.py          # YOLOv8 批量推理
│   └── tool/                        # 数据处理辅助工具
│
├── assets/                   # 静态资源
│   ├── configs/              #   实验配置 YAML
│   ├── prompts/              #   提示词配置 YAML
│   └── weights/              #   模型权重文件
│
├── data/                     # 数据集
│   ├── Compliance_test_data/ #   测试数据集
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

## modules/ — 核心 Python 包

所有可复用的业务逻辑以 `modules.*` 包形式组织，供 `app.py` 和 `scripts/` 统一导入。

### modules/config/

```python
from modules.config.settings import settings
```

- `settings.py`：基于 `@dataclass` 的统一配置加载器，从 `.env` 读取环境变量
- 关键字段：`VLM_API_KEYS`、`VLM_MODEL`、`YOLO_WEIGHTS`、`PROJECT_ROOT`

### modules/vlm/

```python
from modules.vlm.client import create_client_pool, distribute_tasks
from modules.vlm.parser import parse_vlm_response, normalize_label
```

- `client.py`：OpenAI 兼容的 API 客户端池管理，支持多 Key 轮询
- `parser.py`：VLM 响应 JSON 解析与标签标准化

### modules/cv/

```python
from modules.cv.yolov8_inference import YOLOv8SegInference
from modules.cv.image_utils import encode_image_to_base64
```

- `yolov8_inference.py`：YOLOv8 实例分割统一推理接口（`predict()` / `predict_memory()`）
- `mask_inference.py`：MaskRCNN 推理（备用）
- `image_utils.py`：图像编码、IoU 计算、轮廓可视化

### modules/experiment/

```python
from modules.experiment.config import load_config, save_config, list_configs
from modules.experiment.io import load_labels, ResultWriter, append_summary
from modules.experiment.metrics import calculate_metrics, update_leaderboard
from modules.experiment.scoring import ScoringEngine
```

- `config.py`：YAML 实验配置的加载/保存/列举
- `io.py`：标签加载、CSV 结果写入、汇总追加
- `metrics.py`：二分类指标计算、排行榜更新
- `scoring.py`：加权评判引擎

### modules/prompt/

```python
from modules.prompt.manager import load_prompt, list_prompts
```

- `manager.py`：从 `assets/prompts/*.yaml` 加载提示词文本

---

## scripts/ — 实验脚本

| 脚本 | 功能 | 命令示例 |
|------|------|----------|
| `contrast_VLM_CV_test_v2.py` | CV+VLM 联合实验（主用） | `python scripts/contrast_VLM_CV_test_v2.py --config assets/configs/default.yaml` |
| `contrast_VLM_test.py` | 纯 VLM 基准测试 | `python scripts/contrast_VLM_test.py` |
| `yolov8_seg_batch.py` | YOLOv8 批量推理 | `python scripts/yolov8_seg_batch.py -i input_dir -o output_dir` |

### scripts/tool/

辅助数据处理工具：`debug_viewer.py`、`sample_view.py`、`batch_rotate_images.py`、`split_yes_dataset.py` 等。

---

## assets/ — 静态资源

### assets/configs/

实验参数的 YAML 配置文件：

| 文件 | 说明 |
|------|------|
| `default.yaml` | 默认实验配置模板 |
| `scoring_default.yaml` | 加权评分默认配置 |
| `v2_optimized_p5.yaml` | p5 优化实验配置 |
| `v2_optimized_p6.yaml` | p6 优化实验配置 |

### assets/prompts/

VLM 提示词 YAML 文件，命名规则 `{类型}_p{版本}.yaml`：
- `standard_p2` ~ `standard_p8`：标准提示词系列
- `cv_enhanced_p3` ~ `cv_enhanced_p6`：CV 增强提示词系列

### assets/weights/

- `best.pt`：训练好的 YOLOv8l-Seg 模型权重

---

## data/ — 数据集

### data/Compliance_test_data/

```
Compliance_test_data/
├── yes_val/         # 正样本测试集（50 张）
├── no_val/          # 负样本测试集（50 张）
├── yes_val_all/     # 正样本完整集（440 张）
├── no_val_all/      # 负样本完整集（421 张）
├── positive_extra/  # 正样本扩展集
└── negative_extra/  # 负样本扩展集
```

每个目录需包含 `labels.txt`，格式：`文件名,标签`

---

## outputs/ — 实验输出

```
outputs/
├── test_outputs/
│   ├── exp_{timestamp}_{name}/     # 每次实验独立目录
│   │   ├── experiment_config.yaml  #   配置快照
│   │   ├── {exp_name}.csv          #   详细结果
│   │   └── visuals/                #   可视化图片
│   ├── leaderboard_top20.csv       # 排行榜
│   └── archived_experiments/       # 历史存档
└── experiment_outputs/
    └── all_experiments_summary.csv  # 全局汇总
```
