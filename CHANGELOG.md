# Changelog

本文件记录项目的所有版本变更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

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
