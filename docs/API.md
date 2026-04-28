# XiaoanNew 后端 API 接口文档

> 后端基于 Flask 构建，默认监听 `0.0.0.0:5000`。启动命令：`uv run python app.py`。

## 服务架构

启动时自动加载以下组件：
- **YOLOv8-Seg 模型**：从 `assets/weights/best.pt` 加载，使用 CUDA 推理。加载失败时回退到 Mask R-CNN。
- **云端 OCR 客户端**：通过 OpenAI SDK 连接 VLM 服务，用于车牌识别。API 密钥和地址从环境变量读取（`modules/config/settings.py`）。

---

## 接口列表

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/collect/upload` | POST | 数据采集上传 |
| `/api/segmentation/detect` | POST | 实时掩膜分割 |
| `/api/segmentation/detect_static` | POST | 静态图片分析 |
| `/api/test/check_parking` | POST | 停车合规判定 |

---

## GET /api/health

健康检查，返回模型加载状态。

**响应示例**：

```json
{
    "status": "ok",
    "model_loaded": true,
    "model_type": "YOLOv8SegInference",
    "ocr_available": true
}
```

---

## POST /api/collect/upload

数据采集接口，上传标注图片并保存到本地。

**请求格式**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 图片文件 |
| `label` | String | 否 | 标签分类，默认 `unknown` |
| `date` | String | 否 | 日期字符串，默认当天（`YYYY-MM-DD`） |
| `custom_path` | String | 否 | 自定义存储子路径（相对于 `App_collected_dataset/`） |
| `ground_truth` | String | 否 | 真值标签，写入 `labels.txt` |

**存储路径规则**：
- 指定 `custom_path` 时：`App_collected_dataset/{custom_path}/{timestamp}_{filename}`
- 未指定时：`App_collected_dataset/{label}/{date}/{timestamp}_{filename}`

**成功响应** (200)：

```json
{
    "status": "success",
    "path": "/root/XiaoanNew/App_collected_dataset/unknown/2026-03-27/20260327_120000_photo.jpg"
}
```

**失败响应** (400/500)：

```json
{
    "status": "error",
    "message": "No file"
}
```

---

## POST /api/segmentation/detect

实时掩膜分割接口，返回 PNG 格式的透明掩码叠加层。客户端可直接将返回的 PNG 叠加在原图上显示。

**请求格式**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 图片文件 |

**成功响应** (200)：返回 `image/png` 二进制流。

**失败响应** (400/500)：纯文本错误信息。

---

## POST /api/segmentation/detect_static

静态图片分析接口，返回 JSON 格式的检测结果。

**请求格式**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 图片文件 |

**成功响应** (200)：

```json
{
    "status": "success",
    "data": {
        "detections": [
            {
                "id": 0,
                "label": "Electric bike",
                "category_id": 0,
                "confidence": 0.976,
                "bbox": [120, 80, 450, 600],
                "area_ratio": 0.1116
            },
            {
                "id": 1,
                "label": "parking lane",
                "category_id": 2,
                "confidence": 0.85,
                "bbox": [0, 500, 800, 700],
                "area_ratio": 0.042
            }
        ],
        "mask_base64": "iVBORw0KGgo..."
    }
}
```

检测类别说明：

| category_id | label | 说明 |
|-------------|-------|------|
| 0 | Electric bike | 电动车 |
| 1 | Curb | 马路牙子（路缘石） |
| 2 | parking lane | 停车线 |
| 3 | Tactile paving | 盲道 |

**失败响应** (400/500)：

```json
{
    "status": "error",
    "message": "Model not loaded"
}
```

---

## POST /api/test/check_parking

停车合规综合判定接口。完整流程：裁剪下方区域 → 车牌识别 → YOLOv8-Seg 分割 + 几何指标计算 → CV+VLM 联合合规判断。

**请求格式**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 图片文件 |
| `plate_number` | String | 否 | 客户端 OCR 识别的车牌号，有值时跳过云端 OCR（节省约 500-1000ms） |

**处理流程**：

1. **预处理**：对图片 `center_y = h * 0.7` 位置附近按宽度 1/3 裁剪，聚焦地面停车区域。
2. **步骤 A - 车牌识别**：优先使用客户端传入的 `plate_number`，无则调用云端 Qwen-VL-8B OCR。未识别到车牌则直接返回不通过。
3. **步骤 B - YOLOv8-Seg 分割与几何计算**：检测四类目标（电动车、停车线、马路牙子、盲道），提取主车辆 Mask，计算与停车线/盲道的 IoU 和重叠率，构造结构化 `geometry_analysis` 数据。
4. **步骤 C - CV+VLM 联合判断**：将原始图片、线框轮廓可视化图和结构化 CV 数据注入 Qwen-VL-30B（Prompt `cv_enhanced_p5`），获取四维度状态标签，经 `ScoringEngine` 加权评分（阈值 0.60）得出合规结论。若 VLM 不可用则降级为规则判断。
5. **步骤 D - 证据保存**：将原图保存至对应合规/违规目录。

**四维度评分规则**（`ScoringEngine` 默认配置）：

| 维度 | 权重 | 合规状态 | 得分 |
|------|------|----------|------|
| 构图 | 5% | `[合规]` / `[基本合规]` | 1.0 / 0.7 |
| 角度 | 25% | `[合规]` | 1.0 |
| 距离 | 40% | `[完全合规]` | 1.0 |
| 环境 | 30% | `[合规]` | 1.0 |

综合分数 ≥ 0.60 判定为合规；构图门控触发（画面质量不合格）时直接否决。

**成功响应** (200) - VLM 判断：

```json
{
    "is_valid": true,
    "plate_number": "沪A12345",
    "confidence": 0.82,
    "message": "合规停车（综合评分 0.82）",
    "detections": {
        "parking_lane": true,
        "curb": false,
        "tactile_paving": false,
        "objects": [
            {"id": 0, "label": "Electric bike", "confidence": 0.97, "bbox": [120, 80, 450, 600]},
            {"id": 1, "label": "parking lane", "confidence": 0.85, "bbox": [0, 500, 800, 700]}
        ]
    },
    "vlm_analysis": {
        "composition": "[合规]",
        "angle": "[合规]",
        "distance": "[完全合规]",
        "context": "[合规]",
        "final_score": 0.82,
        "dimension_scores": {"composition": 1.0, "angle": 1.0, "distance": 1.0, "context": 1.0},
        "reason": "..."
    }
}
```

**成功响应** (200) - 规则降级判断（VLM 不可用）：

```json
{
    "is_valid": true,
    "plate_number": "沪A12345",
    "confidence": 0.7,
    "message": "规范停车（检测到停车线）",
    "detections": {
        "parking_lane": true,
        "curb": false,
        "tactile_paving": false,
        "objects": [...]
    }
}
```

未识别到车牌时：

```json
{
    "is_valid": false,
    "message": "未检测到清晰车牌，请对准车牌重拍",
    "confidence": 0.0,
    "plate_number": "未识别"
}
```

**证据保存路径**：
- 合规：`App_collected_dataset/evidence/parking_success/{timestamp}_{plate}.jpg`
- 不合规：`App_collected_dataset/evidence/parking_violation/{timestamp}_{plate}.jpg`

---

## 错误处理

所有接口在内部异常时返回 HTTP 500，响应体为 JSON：

```json
{
    "status": "error",
    "message": "具体错误信息"
}
```

或（`check_parking` 接口）：

```json
{
    "code": 500,
    "message": "具体错误信息"
}
```

---

## 环境变量依赖

以下环境变量需在启动前配置（通过 `.env` 文件或系统环境变量）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VLM_API_KEYS` | VLM 合规分析 API 密钥（逗号分隔多个） | 无，必填 |
| `API_BASE_URL` | VLM/OCR 服务基础 URL | `https://api.ppinfra.com/openai` |
| `OCR_API_KEY` | OCR 调用密钥 | 无，必填 |
| `VLM_MODEL` | VLM 合规分析模型 | `qwen/qwen3-vl-30b-a3b-instruct` |
| `OCR_MODEL` | OCR 模型 | `qwen/qwen3-vl-8b-instruct` |
| `YOLO_WEIGHTS` | YOLOv8 权重路径 | `assets/weights/best.pt` |
| `INFERENCE_DEVICE` | 推理设备 | `cuda:0` |
