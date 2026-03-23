# App_collected_dataset 数据集分析报告

## 概述

本报告对 `App_collected_dataset` 目录下的图片数据进行了全面分析，包括目录结构、重复检测、正负样本分布等。

---

## 1. 数据集统计

### 1.1 目录分布

| 目录 | 图片数量 | 说明 |
|------|---------|------|
| `Xiaoan_datasets/y` | 983 | 正样本训练集 |
| `Xiaoan_datasets/yes_val` | 440 | 正样本验证集 |
| `Xiaoan_datasets/no_val` | 421 | 负样本验证集 |
| `Xiaoan_datasets/yes_reserve_val` | 400 | 正样本保留验证集 |
| `Xiaoan_datasets/n` | 392 | 负样本训练集 |
| `Campus_val` | 156 | 校园场景验证集 |
| `Xiaoan_datasets/label` | 90 | 标注数据 |
| `Xiaoan_datasets/blind` | 81 | 盲道场景数据 |
| `zz01_rotate` | 63 | 场景1旋转增强数据 |
| `Xiaoan_datasets/trash` | 56 | 废弃/低质量数据 |
| `zz03` | 54 | 场景3原始数据 |
| `yk_dark` | 53 | 夜间/暗光数据 |
| `zz02_rotate` | 52 | 场景2旋转增强数据 |
| `zz03_rotate` | 48 | 场景3旋转增强数据 |
| `yk01` | 41 | 场景数据 |
| `dark_label` | 8 | 暗光标注数据 |
| `split_data` | 6 | 分割测试数据 |
| `bad` | 3 | 有问题的数据 |
| `test/2026-01-17` | 2 | 测试数据 |
| **总计** | **3349** | |

### 1.2 去重后统计

| 指标 | 数值 |
|------|------|
| 原始图片总数 | 3349 |
| 唯一图片数量 | 1746 |
| 重复图片数量 | 1603 |
| 重复率 | 47.9% |

---

## 2. 正负样本分布

### 2.1 基于目录名的分类

| 分类 | 图片数量 | 占比 |
|------|---------|------|
| 正样本 (停车规范) | 1823 | 54.4% |
| 负样本 (停车违规) | 813 | 24.3% |
| 特殊类别 | 235 | 7.0% |
| 未分类 | 478 | 14.3% |

**正样本目录**: yes_val, y, yes_reserve_val
**负样本目录**: no_val, n
**特殊类别**: trash, blind, label, dark_label

### 2.2 标签文件统计

| 目录 | 条目数 | yes | no |
|------|-------|-----|-----|
| Xiaoan_datasets/yes_val | 438 | 438 | 0 |
| Xiaoan_datasets/no_val | 421 | 0 | 421 |
| Xiaoan_datasets/yes_reserve_val | 400 | 395 | 5 |
| Campus_val | 156 | 112 | 44 |
| Xiaoan_datasets/blind | 90 | 27 | 60 |
| Xiaoan_datasets/label | 90 | 58 | 32 |
| zz01_rotate | 63 | 42 | 21 |
| Xiaoan_datasets/trash | 56 | 52 | 4 |
| zz02_rotate | 52 | 32 | 20 |
| test/2025-12-31 | 51 | 32 | 19 |
| zz03_rotate | 48 | 34 | 14 |
| yk01 | 41 | 32 | 9 |
| dark_label | 8 | 6 | 2 |
| bad | 7 | 4 | 3 |
| split_data | 6 | 2 | 4 |
| test/2026-01-17 | 2 | 2 | 0 |

---

## 3. 重复分析

### 3.1 重复概况

- **同名文件组数**: 1586 组
- **内容相同组数**: 1532 组
- **总重复数量**: 1603 张

### 3.2 主要重复来源

以下目录之间存在大量重复：

1. Campus_val <-> zz01_rotate / zz02_rotate / zz03_rotate
2. Xiaoan_datasets/* 各子目录之间互相重复
3. test/* 与其他目录重复

---

## 4. 目录用途说明

### 4.1 核心数据集 (Xiaoan_datasets/)

| 子目录 | 用途 | 建议 |
|--------|------|------|
| y | 正样本训练集 | 保留，主训练数据 |
| n | 负样本训练集 | 保留，主训练数据 |
| yes_val | 正样本验证集 | 保留，验证用 |
| no_val | 负样本验证集 | 保留，验证用 |
| yes_reserve_val | 备用正样本验证集 | 可合并到yes_val |
| label | 待标注/已标注数据 | 处理后移入y或n |
| trash | 低质量数据 | 可删除 |
| blind | 盲道场景数据 | 特殊场景，单独管理 |

### 4.2 场景数据

| 目录 | 说明 | 建议 |
|------|------|------|
| Campus_val | 校园场景验证集 | 与其他目录有大量重复，建议去重 |
| zz01_rotate | 场景1旋转增强 | 已包含在其他目录，可删除 |
| zz02_rotate | 场景2旋转增强 | 已包含在其他目录，可删除 |
| zz03 | 场景3原始数据 | 评估后决定保留 |
| zz03_rotate | 场景3旋转增强 | 已包含在其他目录，可删除 |
| yk01 | 场景数据 | 保留或合并 |
| yk_dark | 夜间数据 | 特殊场景，建议保留 |
| dark_label | 暗光标注 | 特殊场景，建议保留 |

---

## 5. 清理建议

### 5.1 可删除的重复目录

- zz01_rotate/ (与 Campus_val 重复)
- zz02_rotate/ (与 Campus_val 重复)
- zz03_rotate/ (与 zz03 重复)

### 5.2 可删除的低质量数据

- Xiaoan_datasets/trash/
- bad/

### 5.3 去重后预估

去重后唯一图片约 1746 张，其中：
- 正样本约 1100 张
- 负样本约 450 张
- 特殊场景约 200 张

---

*报告生成时间: 2025-03-24*
