# XiaoanNew 项目结构与职责定义

> 在新增功能或修改代码前，必须查阅此地图。严禁跨越职责边界编写代码（例如在 app.py 中编写复杂的几何算法）。

## 1. 核心分层架构

项目遵循 **驱动-逻辑-核心** 三层架构：

| 层级 | 对应目录 | 职责 |
|------|----------|------|
| 驱动层 | `app.py`, `scripts/` | 外部接口、实验启动、CLI 入口 |
| 逻辑层 | `modules/experiment/`, `modules/prompt/` | 业务编排、提示词组合、实验评分 |
| 核心层 | `modules/cv/`, `modules/vlm/` | 模型原子级推理，不处理业务判定 |

## 2. 目录详细职责

### `modules/config/` - 配置中心

唯一的环境变量读取入口。所有 `os.getenv` 必须在此聚合。

| 文件 | 说明 |
|------|------|
| `settings.py` | `Settings` 单例类，管理 VLM 密钥、YOLO 权重路径、推理设备等。提供 `get_env()` / `get_env_bool()` / `get_env_int()` / `get_env_list()` 四个读取函数。加载优先级：`.env` < `.env.{stage}` < `.env.local` < 系统环境变量。 |

### `modules/cv/` - 计算机视觉

负责返回 masks、boxes、labels 等原始检测数据，不处理业务判定。

| 文件 | 说明 |
|------|------|
| `yolov8_inference.py` | `YOLOv8SegInference` 类。4 个检测类别：`Electric bike`(0)、`Curb`(1)、`parking lane`(2)、`Tactile paving`(3)。`predict()` 返回 `image_raw`、`image_visual`、`objects`、`image_size`。对外工厂函数 `load_yolov8_seg()`。 |
| `mask_inference.py` | `MaskRCNNInference` 类，作为 YOLOv8 的回退方案。提供 `predict_memory()`、`predict_static_json()`、`get_geometry_lines()` 接口。 |
| `image_utils.py` | 工具函数集：`encode_image_to_base64()`（图像 Base64 编码）、`calculate_iou_and_overlap()`（掩膜 IoU 计算）、`combine_masks()`（掩膜合并）、`draw_wireframe_visual()`（线框可视化）。 |

### `modules/vlm/` - 视觉语言模型

负责与云端大模型的请求构造与响应解析。

| 文件 | 说明 |
|------|------|
| `client.py` | `create_client_pool()` 创建 OpenAI 客户端池，`distribute_tasks()` 将推理任务轮询分配给多个客户端。 |
| `retry.py` | `chat_completion_with_retry()` 封装，处理超时、连接错误、限流三类可重试异常。 |
| `parser.py` | `VLMResult` 数据类，`parse_vlm_response()` 从 VLM 的 JSON 输出中提取构图、角度、距离、环境四个维度状态。`normalize_label()` 将 VLM 返回值标准化为 `yes` / `no`。 |

### `modules/experiment/` - 实验框架

负责实验全生命周期管理：配置加载、数据读写、指标计算、评分引擎。

| 文件 | 说明 |
|------|------|
| `config.py` | `ExperimentConfig` 类，管理实验名称、模型、数据路径等参数。`load_config()` 从 YAML 加载，`create_experiment_dirs()` 创建输出目录。 |
| `io.py` | `load_labels()` / `load_all_labels()` 读取标注文件，`collect_image_tasks()` 收集图片-标签对，`ResultWriter` 上下文管理器写 CSV 结果，`append_summary()` 追加实验摘要。 |
| `metrics.py` | `BinaryMetrics` 数据类封装 TP/TN/FP/FN 及衍生指标（Precision、Recall、F1、Accuracy）。`calculate_metrics()` 从预测和真值列表计算指标，`update_leaderboard()` 更新全局排行榜。 |
| `scoring.py` | `ScoringEngine` 引擎，支持加权评分（`score()`）和一票否决（`veto_judge()`）两种模式，`ScoringConfig` 从 YAML 加载四维度权重和阈值。提供 `sweep_threshold()` 和 `grid_search()` 自动调参功能。 |

### `modules/prompt/` - 提示词工厂

负责动态组合图片信息与文本模板。

| 文件 | 说明 |
|------|------|
| `manager.py` | `PromptManager` 类，从 `assets/prompts/*.yaml` 目录加载提示词。`get()` 返回 `Prompt` 对象，`list_prompts()` 列出可用模板，`reload()` 热重载。对外快捷函数 `load_prompt()` / `list_prompts()`。 |

### `modules/train/` - 训练工具

| 目录 | 说明 |
|------|------|
| `yolo/` | YOLO 模型训练相关脚本和配置，当前为预留目录。 |

### `scripts/` - 入口脚本与工具

| 文件 | 说明 |
|------|------|
| `contrast_VLM_CV_test_v2.py` | **主实验入口**。YOLOv8-Seg + VLM 联合判定，输入线框轮廓图，Python 端预计算 IoU/重叠率。通过 `--config` 指定 YAML 配置文件。 |
| `contrast_VLM_CV_test.py` | v1 版联合测试脚本，将原图 + 分割可视化图 + 结构化检测信息一起发给 VLM。 |
| `contrast_VLM_test.py` | 纯 VLM 测试脚本，不使用 CV 预处理，支持加权评分和一票否决两种评判模式。 |
| `depth_pointcloud_demo.py` | 展示脚本：YOLO 分割 + Depth Anything V2 深度估计 + Open3D 点云可视化。输出原图、分割图、深度图、单车点云、全场景点云。 |
| `yolov8_seg_batch.py` | YOLOv8-Seg 批量处理脚本，支持自定义置信度、输出目录、并行线程数。 |

| 工具脚本 (`scripts/tool/`) | 说明 |
|------|------|
| `batch_rotate_images.py` | 批量旋转图片 |
| `split_yes_dataset.py` | 拆分合规数据集 |
| `copy_sample_view.py` | 复制样本用于预览 |
| `sample_view.py` | 样本可视化预览 |
| `debug_viewer.py` | 调试用图片查看器 |
| `view_result_nolabel.py` | 查看无标签结果 |

### `assets/` - 静态资产

严禁存放可执行代码。

| 子目录 | 说明 |
|------|------|
| `configs/` | 实验参数配置 YAML。`default.yaml`（默认配置）、`scoring_default.yaml`（评分权重）、`template.yaml`（空白模板）、`v2_optimized_p5.yaml` / `v2_optimized_p6.yaml`（优化后配置）。 |
| `prompts/` | 提示词模板 YAML。`standard_p*` 系列为纯 VLM 提示词，`cv_enhanced_p*` 系列为 CV 增强提示词（包含 IoU、重叠率等结构化数据）。 |
| `weights/` | 模型权重文件。`best.pt` 为当前 YOLOv8-Seg 生产权重。 |

### `data/` - 数据中心

| 子目录 | 说明 | 文件数 |
|------|------|--------|
| `Compliance_test_data/yes_val/` | 合规验证集 | 51 |
| `Compliance_test_data/no_val/` | 不合规验证集 | 51 |
| `Compliance_test_data/yes_val_all/` | 合规完整集 | 441 |
| `Compliance_test_data/no_val_all/` | 不合规完整集 | 422 |
| `Compliance_test_data/positive_extra/` | 额外合规样本 | 1229 |
| `Compliance_test_data/negative_extra/` | 额外不合规样本 | 279 |
| `App_collected_dataset/zz01_rotate/` | 采集数据（旋转校正） | 30 |
| `App_collected_dataset/zz02_rotate/` | 采集数据（旋转校正） | 53 |
| `App_collected_dataset/zz03/` | 采集数据（原始） | 54 |
| `App_collected_dataset/zz03_rotate/` | 采集数据（旋转校正） | 49 |

### `tests/` - 测试套件

共 14 个测试文件，200+ 个测试函数，覆盖全部核心模块。

| 文件 | 覆盖模块 |
|------|----------|
| `test_config_settings.py` | `modules/config/settings.py` |
| `test_cv_image_utils.py` | `modules/cv/image_utils.py` |
| `test_vlm_client.py` | `modules/vlm/client.py` |
| `test_vlm_parser.py` | `modules/vlm/parser.py` |
| `test_vlm_retry.py` | `modules/vlm/retry.py` |
| `test_experiment_config.py` | `modules/experiment/config.py` |
| `test_experiment_io.py` | `modules/experiment/io.py` |
| `test_experiment_metrics.py` | `modules/experiment/metrics.py` |
| `test_experiment_scoring.py` | `modules/experiment/scoring.py` |
| `test_prompt_manager.py` | `modules/prompt/manager.py` |
| `test_flask_api.py` | `app.py` |
| `test_inprocess_cli.py` | CLI 入口（scoring/config/metrics） |
| `test_cli_and_branches.py` | CLI 子进程 + 分支覆盖 |
| `test_coverage_boost.py` | 边界条件补充 |
| `test_network_resilience.py` | 网络异常场景 |

### `outputs/` - 实验输出

| 子目录 | 说明 |
|------|------|
| `test_outputs/` | 实验结果目录，每次实验生成 `exp_YYYYMMDD_HHMMSS_<实验名>/`，包含 CSV 结果、可视化图片、配置快照。`leaderboard_top20.csv` 为全局排行榜。 |
| `depth_demo/` | 深度估计展示脚本输出目录。 |

## 3. 开发约束

1. **绝对路径**：所有文件读取基于项目根目录，使用 `Path(__file__).parents[n]` 或通过 `modules/config/settings.py` 获取。
2. **依赖隔离**：`modules/` 内部严禁循环引用。依赖方向为 `experiment/ -> cv/ + vlm/ + prompt/`，核心层之间互不依赖。
3. **配置解耦**：超参数通过 `assets/configs/*.yaml` 读取，严禁硬编码。
4. **环境管理**：仅使用 `uv`，禁用 conda/pip。安装依赖用 `uv add`，执行脚本用 `uv run`。
