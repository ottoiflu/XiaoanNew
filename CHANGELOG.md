# Changelog

本文件记录项目的所有版本变更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

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
- 创建项目工作指南文件 `.github/AGENTS.md`，包含：
  - 项目概述与核心功能说明
  - 项目结构文档
  - 类别定义（电动车、马路牙子、停车线、盲道）
  - 代码规范与开发约定
  - 运行环境配置
  - 常用命令参考
  - API 端点列表
  - 注意事项
- 创建 `CHANGELOG.md` 变更日志
- 更新 `.gitignore` 以允许追踪 .md 文件

### Notes
- 这是首个版本，基于现有代码库的状态进行文档化
