# Changelog

本文件记录项目的所有版本变更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

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
