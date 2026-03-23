# Changelog

本文件记录项目的所有版本变更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.3.0] - 2026-03-23

### Added
- 新增 `docs/PROJECT_STRUCTURE.md` 项目结构详细说明文档
  - 根目录核心文件说明
  - scripts/ 脚本工具目录详解
  - yolo/ 训练目录说明
  - MMLab/ OpenMMLab 生态介绍
  - App_collected_dataset/ 数据集目录结构
  - experiment_outputs/ 实验输出字段说明
  - 依赖关系图

### Changed
- 更新 .gitignore 以允许追踪 docs/ 目录

## [0.2.0] - 2026-03-23

### Added
- API 接口详细文档：包含请求参数、响应格式、示例代码
- 模型训练指南：数据集准备、目录结构、训练配置参数说明
- VLM 实验流程文档：实验架构、Prompt 设计原则、几何计算说明
- 部署运维指南：环境配置、启动命令、常见问题排查表

### Changed
- 重构 AGENTS.md 结构，按功能模块划分章节
- 优化文档表格格式，提高可读性

## [0.1.0] - 2026-03-23

### Added
- 创建项目工作指南文件 `.github/AGENTS.md`
- 创建 `CHANGELOG.md` 变更日志
- 更新 `.gitignore` 以允许追踪 .md 文件

### Notes
- 这是首个版本，基于现有代码库的状态进行文档化
