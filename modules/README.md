# modules/ 模块分层说明

> 核心库包，按领域划分子包。严禁循环引用。

## 依赖方向

```
驱动层（app.py / scripts/）
    │
    ▼
┌─────────────────────────────────────┐
│  experiment/（业务编排、评分、指标）   │
│    ↓         ↓         ↓            │
│  cv/      vlm/     prompt/          │
└──────┬──────────────────────────────┘
       │ 共享
       ▼
    config/（配置中心，全局唯一 settings）
```

核心层（cv, vlm）之间互不依赖，仅通过 experiment/ 编排。

## 子包职责

### `config/` - 配置中心

唯一的环境变量和路径常量入口。

| 文件 | 职责 |
|------|------|
| `settings.py` | `Settings` 单例，管理 VLM 密钥、YOLO 权重路径、推理设备。加载优先级：`.env` < `.env.{stage}` < `.env.local` < 系统环境变量 |

### `cv/` - 计算机视觉（核心层）

YOLOv8-Seg 推理、掩膜处理、图像工具。不处理业务判定。

| 文件 | 职责 |
|------|------|
| `yolov8_inference.py` | `YOLOv8SegInference` 类。4 类别检测：Electric bike(0)/Curb(1)/parking lane(2)/Tactile paving(3) |
| `mask_inference.py` | `MaskRCNNInference`，YOLOv8 的回退方案 |
| `image_utils.py` | 工具集：Base64 编码、IoU 计算、掩膜合并、线框可视化 |

检测类别：
| category_id | label | 说明 |
|-------------|-------|------|
| 0 | Electric bike | 电动车（主目标） |
| 1 | Curb | 马路牙子/路缘石 |
| 2 | parking lane | 停车线 |
| 3 | Tactile paving | 盲道 |

### `vlm/` - 视觉语言模型（核心层）

云端大模型请求构造与响应解析。

| 文件 | 职责 |
|------|------|
| `client.py` | OpenAI 客户端池管理，轮询分配任务 |
| `retry.py` | `chat_completion_with_retry()`，对超时/连接错误/限流指数退避重试 |
| `parser.py` | `VLMResult` 数据类 + `parse_vlm_response()` 解析四维度 JSON + `normalize_label()` 标准化 |

### `experiment/` - 实验框架（逻辑层）

实验配置、IO、指标、评分引擎。

| 文件 | 职责 |
|------|------|
| `config.py` | `ExperimentConfig` 数据类，YAML 配置加载 |
| `io.py` | 标签加载、图片收集、CSV 结果写入 |
| `metrics.py` | `BinaryMetrics`：TP/TN/FP/FN + Precision/Recall/F1/Accuracy |
| `scoring.py` | `ScoringEngine`：加权评分 + 一票否决 + 阈值扫描 + 网格搜索 |

### `prompt/` - 提示词管理（逻辑层）

| 文件 | 职责 |
|------|------|
| `manager.py` | `PromptManager` 单例，从 `assets/prompts/*.yaml` 加载提示词，支持热重载 |

### `train/` - 训练工具

当前为预留目录，训练脚本位于 `modules/train/yolo/`。已废弃的死代码已清理。

## 命名规则

- 包名：全小写英文单数
- 文件名：下划线分隔，与类名/函数名对应
- 公共接口：在各子包的 `__init__.py` 中显式导出

## 注意事项

1. **禁止循环引用**：`cv/` 和 `vlm/` 互不引用；`experiment/` 可引用 cv+vlm+prompt；`config/` 可被任何包引用但不反向引用
2. **不硬编码路径**：所有路径通过 `config/settings.py` 获取
3. **新增子包**：需更新 `pyproject.toml` 包发现配置并在 `__init__.py` 中导出接口
