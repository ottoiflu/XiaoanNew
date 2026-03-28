# Changelog

本文件记录项目的所有版本变更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]
## [0.7.0] - 2025-03-28

### Added
- 提示词迭代版本：cv_enhanced_p4_1/2/3.yaml, standard_p4_1/2.yaml
- Phase 3 误判归因分析与三轮消融实验结果写入 EXPERIMENT_REPORT.md
- 实验批次脚本新增 Group F/G/H (p4.1~p4.3) 共 10 个实验配置

### Changed
- 角度判定标准从"夹角 > 60 度"改为区间制 (60-120 度 + 容差 10 度)
- 修复 VLM 对角度判据的系统性误读（FN 角度触发从 5 降至 0）

### Fixed
- p4 原版角度准则歧义导致 VLM 逻辑反转问题

### Experimental Results
- p4.1 (角度修复+IoU强否决): FP 17->5 但 FN 9->25, F1 降至 62.50%
- p4.2 (角度修复+IoU弱化): FN 回落至 16, F1=68.69%
- p4.3 (仅角度修复): F1=75.21%, FN 降至 6, 召回率 88% (最高)
- 结论：IoU 规则改动弊大于利, 原版 p4+最优加权 (F1=77.42%) 仍为综合最优


## [0.6.0] - 2025-03-28

### Added
- 实验可视化图表（4 张）：F1 对比、PR 分布、p4 四维指标、混淆矩阵
- 阶段性实验报告 `outputs/contrast_experiments/EXPERIMENT_REPORT.md`

### Changed
- 加权评分网格搜索脚本 `scripts/scoring_grid_search.py`：覆盖权重、阈值、分数映射三层搜索
- 最优评分配置 `scoring_optimized_cv_p4.yaml` / `scoring_optimized_vlm_p4.yaml`
- 发现原始 threshold=0.60 过高，最优阈值为 0.35，cv_p4 F1 从 75.93% 提升至 77.42%

## [0.5.0] - 2026-03-29

### Added
- 批量对比实验运行器 `scripts/run_contrast_batch.py`，支持 18 组实验矩阵
  - 纯 VLM / VLM+CV 两种工作流模式
  - 一票否决 / 加权评分两种评判方式
  - strip_geometry 选项：移除 CV 几何数据中的 IoU/重叠率数值
  - 命令行参数 `--list` 列出实验、`-e` 选择实验子集
- 新提示词文件
  - `standard_p4.yaml`：纯 VLM 版本，接地点锚定 + 融合验证法
  - `cv_enhanced_p7.yaml`：VLM+CV 版本，CV 数据仅辅助定位不改变判定标准
- 实验分析报告 `outputs/contrast_experiments/ANALYSIS.md`

### Changed
- 对比实验验证 cv_enhanced_p4 + 一票否决为当前最优方案 (F1=0.76, Acc=74%)



### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- 重构 depth_pointcloud_demo.py：支持电动车、马路牙子、车道线三类别分别输出掩膜图和点云
- 深度估计模型升级为 Depth Anything V2 Large（原 Small）
- 每张图片的输出文件统一存放在以图片名命名的子目录中
- 新增 render_mask_image、merge_class_masks、process_single_class 等函数

### Added
- 为单车点云新增 PCA 方向分析功能 (scripts/depth_pointcloud_demo.py)
  - compute_pca: 对点云执行 PCA 分解，输出质心、特征值、特征向量
  - create_pca_arrows: 生成三轴方向 LineSet（红/绿/蓝对应 PC1/PC2/PC3）
  - 可视化时在点云中叠加 PCA 方向轴和质心球标记
  - 控制台输出方差贡献率及主方向向量
- 新增点云 GUI 可视化脚本 (scripts/visualize_pointcloud_gui.py)
  - 在有 X11 显示环境的服务器上交互式查看 PLY 点云
  - 自动对单车点云叠加 PCA 三轴方向和质心标记
  - 支持 --ply 单文件模式和 --dir 目录批量模式

### Docs
- 补全 docs/PROJECT_STRUCTURE.md：基于仓库扫描填入三层架构、模块职责、数据统计、测试覆盖等实际内容
- 补全 docs/API.md：编写完整的 Flask 后端接口文档（5 个端点的请求/响应格式、处理流程）
- 整理 docs/IDEA.md：合并合规判定标准与创新点记录，补充评分权重、线框输入、深度辅助判距等实际内容

## [2.2.0] - 2026-03-27

### Added
- 新增电动车掩膜深度估计与点云可视化展示脚本 (scripts/depth_pointcloud_demo.py)
  - 同一张图片生成 5 个输出：原图、分割图、深度图、单车点云、全场景点云
  - 调用现有 YOLOv8-Seg 模型检测电动车并提取掩膜
  - 集成 Depth Anything V2 单目深度估计模型
  - RGB + Depth 转 3D 点云，支持 Open3D 交互式可视化
  - 支持 --no-gui 无界面模式（适配无显示器服务器）

### Dependencies
- 新增 open3d (点云可视化)
- 新增 transformers (Depth Anything V2 模型加载)

## [2.1.3] - 2026-03-25

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- 重写 docs/env.md: 基于实际部署环境审计，修正全部 7 处文档与环境不一致的问题
  - 文件名 environment.yaml -> environment.yml
  - 环境名 XiaoanNewtest1 -> XiaoanNew
  - uv compile 命令移除多余的 --extra-index-url 参数
  - 补充 env_setup.sh 激活脚本说明
  - 补充实际环境快照（Python 3.10.20 / PyTorch 2.5.1+cu121 / CUDA 12.1）
  - 补充 [tool.uv.index] 双索引源机制说明
  - 修正 requirements.txt 中 -e . 的职责分离说明

### Fixed
- 修正 environment.yml 中 name 字段从 XiaoanNewtest1 改为 XiaoanNew，与实际环境一致
- 修正 environment.yml 头部注释中过时的 uv compile 命令
- .gitignore 添加 .yml 白名单，使 environment.yml 可被版本控制
- docs/env_manage.md: 新增依赖工作流改为 uv add/remove --frozen 方式，取代手动编辑 pyproject.toml
- docs/env.md 重命名为 docs/env_manage.md


## [2.1.2] - 2026-03-24

### Added
- 新增 tests/test_inprocess_cli.py: 进程内 CLI 入口覆盖测试（14 tests）
  - scoring.py main() 的 evaluate / sweep / grid / 无子命令四个分支
  - scoring.py batch_evaluate() 的 fn / fp 分支覆盖
  - config.py / prompt/manager.py / metrics.py 的 __main__ 入口块（runpy）
- 新增 tests/test_cli_and_branches.py: CLI 子进程 + 剩余分支测试（18 tests）
  - scoring CLI 的 evaluate / sweep / grid 端到端验证
  - config CLI 的 list / show / create 端到端验证
  - prompt/manager CLI 的 list / show / info 端到端验证
  - settings.py dotenv 降级路径 + .env 文件加载
  - scoring batch_evaluate 中文 ground_truth 解析

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- 测试规模 315 -> **347**（+32 tests）
- 模块覆盖率大幅提升:
  - experiment/config.py: 73% -> **100%**
  - experiment/scoring.py: 79% -> **99%** (仅剩 `if __name__` 1 行)
  - prompt/manager.py: 81% -> **100%**
  - experiment/metrics.py: 95% -> **98%**
  - 总体: 65% -> **72%** (排除 GPU 推理模块后有效覆盖率更高)


## [2.1.1] - 2026-03-24

### Added
- 新增 tests/test_flask_api.py: Flask API 端点集成测试（20 tests）
  - 覆盖全部 5 个 API 端点（health / upload / detect / detect_static / check_parking）
  - 包含路径遍历安全测试、模型未加载异常路径、OCR 故障降级等场景
- 新增 tests/test_coverage_boost.py: 模块覆盖率补充测试（26 tests）
  - scoring.py: 缺失 score_map 校验、中文 ground_truth 解析、grid_search 优化目标切换等
  - experiment/config.py: 默认配置目录发现、实验目录备份生成、完整字段初始化
  - prompt/manager.py: 单例模式、模块级快捷函数、默认目录解析
  - metrics.py: 排行榜去重、零值实验过滤、自定义标题报告

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- tests/conftest.py: 新增 flask_app fixture（mock 模型 + Flask 测试客户端）
- 测试规模从 269 增至 315（+46 tests），模块覆盖率从 61% 提升至 65%
- 排除 GPU 推理模块后可测试模块覆盖率均在 73% 以上


## [2.1.0] - 2026-03-24

### Added
- 新增 modules/vlm/retry.py: 基于 tenacity 的 API 调用重试机制
  - 对 APITimeoutError / APIConnectionError / RateLimitError 自动指数退避重试（最多 3 次, 2s -> 4s -> 8s）
  - 不可恢复异常（认证失败等）立即抛出，不浪费重试次数
- 新增 tests/test_vlm_retry.py: 重试机制单元测试（19 tests）
  - 可恢复异常触发重试并最终成功
  - 重试耗尽后正确抛出最终异常
  - 不可恢复异常立即抛出不重试
  - 参数透传验证
  - 脚本集成断言

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- contrast_VLM_test.py: VLM 调用改用 chat_completion_with_retry
- contrast_VLM_CV_test_v2.py: VLM 调用改用 chat_completion_with_retry
- app.py: OCR 调用改用 chat_completion_with_retry
- modules/vlm/__init__.py: 导出 chat_completion_with_retry
- requirements.txt: 新增 tenacity>=8.0 依赖

## [2.0.3] - 2026-03-24

### Added
- 新增 pytest 单元测试套件（250 tests），覆盖全部 9 个核心模块
- 新增 tests/conftest.py 共享 fixtures（图片、掩码、标签目录、评分配置等）
- 新增 tests/test_network_resilience.py 网络异常 mock 测试（38 tests）：
  - VLM API：连接超时、服务不可达、速率限制、响应截断、空 choices
  - OCR API：超时/连接失败/限频均优雅降级返回 None
  - 客户端池：部分/全部 key 失效、round-robin 容错
  - 畸形响应：截断 JSON、HTML 错误页、BOM、二进制乱码、超大响应

### Documented
- 测试明确记录了当前系统无重试机制的现状（TestVLMNoRetryBehavior）


## [2.0.2] - 2026-03-24

### Security
- 移除 scripts/contrast_VLM_CV_test.py 中硬编码的 3 个 VLM API Key，改为从 settings 读取
- 使用 git-filter-repo 清除 Git 历史中全部 5 个泄露的 API Key（替换为 REDACTED_API_KEY_*）

## [2.0.1] - 2026-03-24

### Fixed
- 修复 modules/config/settings.py 中 PROJECT_ROOT 路径计算深度错误（.parent.parent -> .parent.parent.parent）
- 修复 modules/prompt/manager.py 提示词目录指向 prompts/ 而非 assets/prompts/
- 修复 modules/experiment/config.py 配置目录指向 configs/ 而非 assets/configs/
- 修复 .env 和 .env.example 中 YOLO_WEIGHTS 路径未更新为 assets/weights/
- 修复全部 YAML 实验配置（default.yaml 等）中的数据集、权重、输出路径
- 修复 app.py、scripts/ 全部入口脚本中的硬编码路径（weights/、Compliance_test_data/、test_outputs/、experiment_outputs/）
- 修复 scripts/contrast_VLM_CV_test.py 和 yolov8_seg_batch.py 的 sys.path 指向 scripts/ 而非项目根
- 移除 app.py 和 contrast_VLM_CV_test_v2.py 中多余的 scripts/ sys.path

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- 全部模块 docstring 中的导入示例更新为 modules.* 风格
- 重写 docs/PROJECT_STRUCTURE.md，反映 assets/data/outputs 三层目录布局
- 更新 AGENTS.md 项目结构树和实验配置系统段落

## [2.0.0] - 2026-03-24

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数

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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- 重构测试脚本的提示词加载方式
  - 移除内嵌 PROMPT_LIB 字典
  - 改为从外部 YAML 文件动态加载
- 首个提示词配置: `cv_enhanced_p4.yaml`


## [0.8.3] - 2026-03-24

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- 原验证集重命名为 yes_val_all 和 no_val_all
- 原 positive/negative 重命名为 positive_extra/negative_extra
- 更新 AGENTS.md 目录结构说明


## [0.8.1] - 2026-03-24

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- 更新 AGENTS.md 以反映最新项目结构
- 同步文档中的路径引用（dataset.yaml, test_outputs）

### Fixed
- 修正 dataset.yaml 示例路径为 yolo/data/coco

## [0.8.0] - 2026-03-24

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- 重组实验输出结构
  - 每次测试创建独立目录 `exp_{timestamp}_{name}/`
  - 目录内包含 results.csv 和 visuals/ 子目录
- 历史实验数据迁移至 archived_experiments/

## [0.7.0] - 2026-03-24

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
- 统一测试输出目录结构
- 脚本可视化输出路径更新为 test_outputs/seg_visuals

## [0.6.0] - 2026-03-24

### Changed
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
- 修复掩膜因 YOLO 原型分辨率不足导致的断裂（resolve_mask_priority 优先级冲突解决）
- yolov8_inference.py predict() 方法新增 retina_masks 参数
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
