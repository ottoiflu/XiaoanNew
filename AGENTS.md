# XiaoanNew AI 指导手册 (Master Agent Guide)

你是 XiaoanNew 项目的资深架构师。在执行任何任务前，请务必阅读本指南。

## 1. 项目核心定位
基于 **YOLOv8-Seg + VLM (Qwen-VL)** 的共享单车智能停放检测系统。
- **目标**：通过 CV 几何计算与大模型逻辑推理，判定车辆是否规范停放在指定区域。

## 2. 核心技术栈 (不可违背)
- **环境管理**: 纯 `uv` 环境（禁用 conda/pip）。详见 `docs/ENV.md`。
- **推理引擎**: Ultralytics (YOLOv8), OpenAI SDK (VLM).
- **后端框架**: Flask (Python 3.10).
- **代码规范**: Ruff (Lint/Format). 详见下文。

## 3. 目录地图 (关键路径)
- `src/` -> `modules/`: 所有核心逻辑（cv, vlm, experiment）。
- `scripts/`: 入口脚本。**主实验脚本**: `scripts/contrast_VLM_CV_test_v2.py`。
- `assets/`: 所有的静态配置（configs, prompts）和模型权重（weights）。
- `data/`: 结构化测试集。
- `docs/`: 详细分项文档。

## 4. 详细文档索引 (遇到特定任务必读)
| 任务类型 | 参考文档 |
| :--- | :--- |
| **环境配置/增加依赖** | [环境规范](docs/ENV.md) |
| **修改后端接口/API** | [API 接口文档](docs/API.md) |
| **模型训练/权重更新** | [训练指南](docs/TRAINING.md) |
| **实验复现/Prompt 调整** | [实验流程说明](docs/EXPERIMENTS.md) |

## 5. AI 执行原则 (Highest Priority)
1. **环境操作**: 任何安装包的操作必须使用 `uv add`。执行脚本必须前缀 `uv run`。
2. **配置解耦**: 严禁在代码中硬编码超参数，必须通过 `assets/configs/*.yaml` 读取。
3. **Prompt 管理**: 修改 Prompt 请前往 `assets/prompts/*.yaml`。
4. **代码风格**: 
   - 必须通过 Ruff 校验。
   - 函数需包含中文 Docstring。
   - 严禁使用 bare `except:`，必须显式捕获异常。
5. **路径处理**: 统一使用绝对路径（基于项目根目录）或通过 `modules/config/settings.py` 获取。

## 6. 实验工作流 (Workflow)
1. 修改 `assets/configs/` 中的 YAML。
2. 执行 `uv run scripts/contrast_VLM_CV_test_v2.py --config assets/configs/xxx.yaml`。
3. 检查 `outputs/test_outputs/` 下生成的实验目录。