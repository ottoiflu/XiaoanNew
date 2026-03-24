# Changelog

本文件记录项目的所有版本变更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [2.0.0] - 2026-03-24

### Changed
- 项目级包重构: 引入 modules/ 顶层包，按领域划分子包
  - modules/config/: 配置管理 (原 config/)
  - modules/vlm/: VLM 客户端与响应解析 (原 utils/vlm_client.py, utils/vlm_parser.py)
  - modules/cv/: 计算机视觉 (原 utils/image_utils.py, scripts/yolov8_seg_inference.py, mask_inference.py)
  - modules/experiment/: 实验管理 (原 utils/experiment_io.py, utils/metrics.py, utils/scoring.py, scripts/experiment_config.py)
  - modules/prompt/: 提示词管理 (原 scripts/prompt_manager.py)
- 所有入口脚本和 app.py 的 import 路径统一迁移至 modules.*
- pyproject.toml 包发现配置更新

### Removed
- config/ 旧目录 (迁移至 modules/config/)
- utils/ 旧目录 (迁移至 modules/ 各子包)
- 根目录 mask_inference.py (迁移至 modules/cv/)
- scripts/ 中的库模块文件 (迁移至 modules/ 各子包)

## [1.6.1] - 2026-03-24

### Added
- pyproject.toml 新增 ruff 配置（lint + format），替代 black/isort
- ruff 加入 dev 依赖

### Changed
- 全项目 ruff check 零违规（修复 E722 bare-except、F401 unused-import、F541 f-string、W293 空白行等 444 处问题）
- ruff format 统一代码风格（22 文件重格式化）
- pyproject.toml 移除 [tool.black] 和 [tool.isort] 配置
- 所有 bare except 改为 except Exception

### Removed
- dev 依赖中移除 black、isort（ruff 完全替代）

## [1.6.0] - 2026-03-24

### Added
- utils/vlm_parser.py: VLM 响应解析与标签标准化模块
  - normalize_label() 统一标签归一化（合并原先 3 处重复实现）
  - VLMResult 数据类，结构化承载四维度解析结果
  - parse_vlm_response() 从 VLM 文本中提取 JSON 四维度状态
- utils/vlm_client.py: API 客户端池管理模块
  - create_client_pool() 从环境变量加载 API 密钥创建客户端
  - distribute_tasks() 轮询分配任务到客户端
- utils/image_utils.py: 图像处理工具模块
  - encode_image_to_base64() 支持文件路径 / ndarray / PIL Image 输入
  - calculate_iou_and_overlap() Mask IoU 与覆盖率计算
  - combine_masks() 按类别合并 Mask
  - draw_wireframe_visual() 线框轮廓可视化
- utils/experiment_io.py: 实验 IO 工具模块
  - load_labels() / load_all_labels() 标签加载
  - collect_image_tasks() 图片文件收集
  - ResultWriter 上下文管理器，CSV 流式写入
  - append_summary() 汇总指标追加
- BinaryMetrics.from_confusion_matrix() 类方法，支持从混淆矩阵直接构建指标
- 5 个 YAML 提示词文件迁移自 v1 内联字典
  - standard_p2, standard_p3, standard_p4, standard_p5, cv_enhanced_p3_compare

### Changed
- scripts/contrast_VLM_test.py: 全面重构（557 行 -> 189 行）
  - 移除硬编码 API 密钥，改用 config.settings 环境变量管理
  - 移除 328 行内联 PROMPT_LIB 字典，统一使用 prompt_manager
  - 所有工具函数替换为共享模块调用
- scripts/contrast_VLM_CV_test_v2.py: 全面重构（662 行 -> 319 行）
  - 消除 main() 与 _run_experiment() 的 140 行重复代码
  - 替换内联几何计算、图像编码、标签加载为共享模块
  - 统一使用 vlm_parser + ScoringEngine 进行判定
- utils/metrics.py: normalize_label 改为复用 vlm_parser 统一实现
- utils/scoring.py: _calc_metrics 改为委托 BinaryMetrics.from_confusion_matrix
- utils/__init__.py: 导出所有新增模块的公共接口

### Removed
- v1 脚本中的硬编码 API 密钥（安全风险消除）
- v1 脚本中的 328 行内联提示词字典
- v2 脚本中的重复 main() 函数
- 4 处 norm_yesno / normalize_label 重复实现（统一为 1 处）
- 4 处 calculate_and_report 重复实现（统一为 metrics 模块）


## [1.5.0] - 2026-03-24

### Added
- utils/scoring.py 加权评判引擎模块
  - ScoringConfig / ScoringResult 数据类
  - ScoringEngine：score() 单条评判、batch_evaluate() 批量重评估
  - sweep_threshold() 阈值扫描、grid_search() 权重网格搜索
  - 一票否决兼容方法 veto_judge()
  - CLI 入口：支持 evaluate / sweep / grid 三种子命令
  - 模糊匹配容错，支持状态标签格式差异
- configs/scoring_default.yaml 评判配置（网格搜索最优参数）

### Changed
- scripts/contrast_VLM_test.py 集成加权评判引擎
  - parse_vlm_response 评判逻辑从硬编码一票否决改为可配置引擎
  - CONFIG 新增 scoring_config 字段，设为 None 可回退到一票否决
  - CSV 输出新增 weighted_score 列，记录加权得分数值
  - 启动时打印评判模式和阈值信息

### Improved
- 加权评判 F1=0.73 (对比一票否决 F1=0.71)，FP 30->29，FN 6->5
- 网格搜索最优参数：comp=0.05 angle=0.25 dist=0.40 ctx=0.30 threshold=0.60
- 关键发现：[基本合规-压线] 分值设为 0.0 比 0.5 效果更优（F1 0.74->0.75 on static CSV）
- 角度单维度不合规可被其他合规维度补偿（FN 降低的核心机制）


## [1.4.0] - 2026-03-24

### Added
- 评估标准文档 docs/idea.md，定义四维度停车合规判定规范
- v1 优化提示词系列：standard_p6 / standard_p7 / standard_p8
- v2 优化提示词系列：cv_enhanced_p5 / cv_enhanced_p6
- v1 基线对比实验 (standard_p4) 验证优化效果
- v2 实验配置：configs/v2_optimized_p5.yaml, configs/v2_optimized_p6.yaml

### Changed
- scripts/contrast_VLM_test.py 新增 prompt_manager 回退加载支持，可使用外部 YAML 提示词

### Improved
- v1 最优提示词 standard_p6 相比基线 standard_p4 F1 提升 48% (0.48 -> 0.71)
- 修正 cv_enhanced_p4 中的角度规则错误（标线类参照物允许平行或垂直）
- 简化距离判定标准，移除不准确的 Df/Dr 深度估算，改用视觉观察

### Analysis
- v1 (纯 VLM) 方案在当前测试集上显著优于 v2 (CV+VLM)
- v2 的 CV 轮廓可视化导致 VLM 在距离维度产生系统性误判
- 提示词严格度与召回率呈负相关，需在精确率和召回率间权衡


## [1.3.0] - 2026-03-24

### Added
- 实验排行榜功能
  - 在 utils/metrics.py 中新增 update_leaderboard()，自动汇总所有实验记录
  - 按 F1 降序、Accuracy 降序排序，保留 Top 20 条记录
  - 输出到 test_outputs/leaderboard_top20.csv
  - 每次实验结束后自动刷新排行榜
  - 终端格式化打印排行榜表格


## [1.2.0] - 2026-03-24

### Changed
- 重组配置目录结构
  - 将 `scripts/configs/` 移至项目根目录 `configs/`
  - 将 `scripts/prompts/` 移至项目根目录 `prompts/`
  - 更新 experiment_config.py 和 prompt_manager.py 路径引用
- 更新 .gitignore 配置规则，确保 YAML 文件正确追踪
- 同步 AGENTS.md 和 PROJECT_STRUCTURE.md 文档

### Fixed
- 修复配置目录迁移后的 Git 追踪问题

## [1.1.0] - 2026-03-24

### Security
- 从 app.py 和测试脚本中移除硬编码 API Key
- 所有 API Key 现在从 .env 文件加载




### Added

- 新增依赖管理文件

  - `requirements.txt`: pip 依赖清单

  - `pyproject.toml`: 现代 Python 项目配置

- 新增环境变量管理系统

  - `.env.example`: 环境变量模板

  - `config/settings.py`: 统一配置加载模块

- 新增公共工具模块 `utils/`

  - `metrics.py`: 评估指标计算和报告生成



### Changed

- 更新 .gitignore 支持新增文件类型



### Security

- API Key 从代码中移除，改为环境变量管理




## [1.0.0] - 2026-03-24

### Added
- 新增实验配置管理系统 (`scripts/experiment_config.py`)
  - 支持从 YAML 文件加载实验配置
  - ExperimentConfig dataclass 统一管理配置参数
  - 配置自动备份到实验输出目录
- 创建 `scripts/configs/` 目录存放实验配置
  - `default.yaml`: 默认实验配置模板
  - `test_config_system.yaml`: 配置系统测试用配置

### Changed
- 重构测试脚本 (`contrast_VLM_CV_test_v2.py`)
  - 添加 `--config` 命令行参数支持配置文件驱动
  - 添加 `--list-configs` 列出可用配置
  - 保持向后兼容：无参数时使用内置默认配置
  - 每次实验自动保存配置快照到输出目录

### Technical Details
- 配置字段: exp_name, model, prompt_id, max_size, quality, segmentor, data_folders, output_root, max_workers
- 运行方式: `python contrast_VLM_CV_test_v2.py --config configs/default.yaml`


## [0.9.1] - 2026-03-24

### Changed
- 更新 AGENTS.md 文档
  - 添加 prompt_manager.py 和 prompts/ 目录说明
  - 添加提示词管理使用指南
  - 更新实验输出格式说明
- 重写 docs/PROJECT_STRUCTURE.md
  - 同步所有目录结构变更
  - 添加 Compliance_test_data 和 test_outputs 说明
  - 移除已删除文件的引用


## [0.9.0] - 2026-03-24

### Added
- 新增提示词管理模块 (`scripts/prompt_manager.py`)
  - 支持从 YAML 文件加载提示词
  - 提供 `load_prompt()` 便捷函数
  - 命令行工具: `python prompt_manager.py list/show/info`
- 创建 `scripts/prompts/` 目录存放提示词配置

### Changed
- 重构测试脚本的提示词加载方式
  - 移除内嵌 PROMPT_LIB 字典
  - 改为从外部 YAML 文件动态加载
- 首个提示词配置: `cv_enhanced_p4.yaml`


## [0.8.3] - 2026-03-24

### Changed
- 将测试集从各200张缩减为各50张
  - yes_val/: 50张 (从200张中随机保留)
  - no_val/: 50张 (从200张中随机保留)
- 更新 AGENTS.md 测试集数量说明


## [0.8.2] - 2026-03-24

### Added
- 创建精简测试集 (各200张随机抽样)
  - yes_val/: 正样本测试集 (从440张中抽取)
  - no_val/: 负样本测试集 (从421张中抽取)

### Changed
- 原验证集重命名为 yes_val_all 和 no_val_all
- 原 positive/negative 重命名为 positive_extra/negative_extra
- 更新 AGENTS.md 目录结构说明


## [0.8.1] - 2026-03-24

### Changed
- 更新 AGENTS.md 以反映最新项目结构
- 同步文档中的路径引用（dataset.yaml, test_outputs）

### Fixed
- 修正 dataset.yaml 示例路径为 yolo/data/coco

## [0.8.0] - 2026-03-24

### Changed
- 重组实验输出结构
  - 每次测试创建独立目录 `exp_{timestamp}_{name}/`
  - 目录内包含 results.csv 和 visuals/ 子目录
- 历史实验数据迁移至 archived_experiments/

## [0.7.0] - 2026-03-24

### Changed
- 统一测试输出目录结构
- 脚本可视化输出路径更新为 test_outputs/seg_visuals

## [0.6.0] - 2026-03-24

### Changed
- 重新组织数据目录结构
- 创建 Compliance_test_data/ 存放清洗后的测试数据
  - yes_val/: 正样本验证集 (428张)
  - no_val/: 负样本验证集 (408张)
  - positive/: 去重正样本 (1228张)
  - negative/: 去重负样本 (278张)

### Removed
- 移除 yolov8seg_visuals_* 可视化目录 (约830M)

## [0.5.0] - 2026-03-24

### Added
- 添加数据集分析报告 docs/DATASET_ANALYSIS.md
  - 总计分析 3349 张图片
  - 发现 47.9% 重复率 (1603张)
  - 唯一图片 1746 张



## [0.4.0] - 2026-03-23

### Changed
- 重构项目结构，移除 MMLab 依赖
  - 模型权重移至 `weights/best.pt`
  - 更新所有代码中的路径引用
- 删除临时测试文件：`hh.py`, `rtmdet_tiny_*.py`, `*.pth`

### Removed
- 移除 `MMLab/` 目录（包含 mmdetection 和 mmyolo submodules）

### Notes
- 如需重新训练模型，需单独安装 ultralytics 库
- 推理功能不受影响，仅使用 weights/best.pt

## [0.3.0] - 2026-03-23

### Added
- 新增 `docs/PROJECT_STRUCTURE.md` 项目结构详细说明文档

### Changed
- 更新 .gitignore 以允许追踪 docs/ 目录

## [0.2.0] - 2026-03-23

### Added
- API 接口详细文档
- 模型训练指南
- VLM 实验流程文档
- 部署运维指南

## [0.1.0] - 2026-03-23

### Added
- 创建 `.github/AGENTS.md` 项目工作指南
- 创建 `CHANGELOG.md` 变更日志
