# 端云协同还车业务流程详解

> 本文档描述共享单车还车合规检测的完整业务流程，覆盖 Android 客户端五状态机、端云职责划分、API 交互时序和服务端判断逻辑。

---

## 1. 整体架构

系统采用"CV 为骨，VLM 为肉"的端云协同设计：

- **端侧（Android App）**：MLKit OCR 车牌识别、CameraX 帧捕获、IoU 端侧计算、状态机驱动的 UI
- **云端（Flask + GPU 实例）**：YOLOv8-Seg 实例分割、Qwen-VL 多模态大模型合规判断、ScoringEngine 加权评分

```
手机摄像头
    │
    ├── MLKit OCR（端侧）─── 车牌 3帧防抖 ──→ 确认车牌号
    │
    ├── CameraX ImageAnalysis ──→ 500ms节流上传 ──→ /api/segmentation/detect_static
    │                                              │
    │                              返回检测框坐标（bbox）
    │                                              │
    │                          端侧计算 IoU + 质心偏移
    │                                              │
    │                           IoU≥0.40 连续3帧？
    │                              ├── 否 → 方向箭头引导
    │                              └── 是 → 缓存帧，进入 ReadyToCapture
    │
    └── 用户确认还车 ──→ 复用缓存帧 + 车牌号 ──→ /api/test/check_parking
                                                  │
                                    步骤A：车牌识别（跳过 OCR）
                                    步骤B：YOLOv8-Seg 分割 + IoU
                                    步骤C：VLM 四维度合规判断
                                                  │
                                       返回 is_valid + vlm_analysis
```

---

## 2. Android 客户端状态机

客户端使用 sealed class `ReturnBikeState` 驱动，共五个状态，通过 `ReturnBikeViewModel` 管理，绝不在 Composable 中直接发起网络请求。

### 2.1 状态定义

| 状态 | 含义 | 触发条件 |
|------|------|----------|
| `PlateScanning` | 车牌扫描中 | 初始状态 |
| `GuidanceLoop(iou, offsetX, offsetY, bikeDetected)` | 引导取景 | 车牌确认后自动进入 |
| `ReadyToCapture(iou)` | 就绪等待确认 | IoU ≥ 0.40 连续 3 帧 |
| `Uploading` | 上传合规判断中 | 用户点击"确认还车" |
| `ResultReady(result)` | 结果展示 | 服务端返回判断结果 |
| `Error(message)` | 错误 | 网络异常或业务错误 |

### 2.2 状态转换图

```
初始
  │
  ▼
PlateScanning
  │ MLKit OCR 连续 3 帧识别到相同车牌
  ▼
GuidanceLoop ←──────────────────────────────┐
  │ 上传帧至 /detect_static（500ms节流）      │
  │ 端侧计算 IoU + 质心偏移                   │
  │                                          │
  ├── IoU < 0.40 → UI 显示方向箭头 ──────────┘
  │
  │ IoU ≥ 0.40 连续 3 帧，缓存当前帧
  ▼
ReadyToCapture
  │ 绿色边框 + "确认还车"按钮
  │
  ├── 用户点击"重新扫描" → 回到 PlateScanning
  │
  └── 用户点击"确认还车"
        ▼
      Uploading
        │ 复用缓存帧 + 端侧车牌号
        │ POST /api/test/check_parking
        ▼
      ResultReady
        │ is_valid=true → 还车成功弹窗
        │ is_valid=false → 还车失败弹窗（附违规原因）
        └── 用户关闭弹窗 → 回到 PlateScanning
```

### 2.3 关键优化点

| 优化 | 说明 | 实现位置 |
|------|------|----------|
| 分割帧复用 | ReadyToCapture 缓存的帧直接用于最终上传，无需二次快门 | `cachedGoodBitmap` in `ReturnBikeViewModel` |
| 500ms 节流 | GuidanceLoop 阶段限制帧上传频率，避免服务器过载 | `AtomicBoolean isProcessingFrame` + 时间戳检查 |
| 端侧 IoU 计算 | 云端只返回检测框坐标，IoU 在端侧完成，减少数据传输 | `ReturnBikeViewModel.computeIoU()` |
| 车牌传服务端 | 直接将端侧 OCR 结果传给服务端，跳过云端 OCR，节省 500-1000ms | `ApiService.checkParking(plate_number=...)` |

---

## 3. API 交互时序

### 3.1 引导阶段（GuidanceLoop）

```
Android                              服务端
  │                                    │
  │─── POST /api/segmentation/detect_static ──→│
  │         file: 当前预览帧 (JPEG)             │
  │                                    │
  │                      YOLOv8-Seg 推理 (~50ms)
  │                                    │
  │←── 200 OK ─────────────────────────│
  │    detections: [{label, bbox, score, ...}]  │
  │    mask_base64: 可视化掩码图               │
  │                                    │
  │  端侧：computeIoU(bikeBox, parkingLaneBox) │
  │  端侧：计算质心偏移 (offsetX, offsetY)     │
  │                                    │
  更新 GuidanceLoop 状态，刷新 UI
```

**检测类别与 IoU 意义**：

| label | category_id | 用途 |
|-------|-------------|------|
| `Electric bike` | 0 | 主目标，用于 IoU 基准框 |
| `Curb` | 1 | 辅助参照物（马路牙子） |
| `parking lane` | 2 | 停车线，与电动车 IoU ≥ 0.40 视为对准 |
| `Tactile paving` | 3 | 盲道，重叠则警告违规风险 |

**IoU 阈值**：端侧计算电动车检测框与停车线检测框的 IoU，≥ 0.40 且连续 3 帧稳定则判定对准。

### 3.2 还车判断阶段（Uploading）

```
Android                              服务端
  │                                    │
  │─── POST /api/test/check_parking ──→│
  │    file: 缓存的就绪帧 (JPEG)       │
  │    plate_number: "沪A12345"        │
  │                                    │
  │                      预处理：裁剪 center_y=0.7h 区域
  │                                    │
  │                      步骤A：直接使用客户端传入的车牌号
  │                             （跳过 Qwen-VL-8B OCR）
  │                                    │
  │                      步骤B：YOLOv8-Seg 完整分割
  │                             提取 Electric bike Mask
  │                             计算与 parking lane 的 IoU、重叠率
  │                             计算与 Tactile paving 的重叠率
  │                             构造 geometry_analysis 结构体
  │                                    │
  │                      步骤C：CV+VLM 联合判断
  │                        ├── 生成线框轮廓可视化图
  │                        ├── Base64 编码原图 + 轮廓图
  │                        ├── 构造结构化 CV 检测数据 JSON
  │                        ├── 拼接 Prompt（cv_enhanced_p5）
  │                        └── 调用 Qwen-VL-30B，获取四维度状态
  │                                    │
  │                      ScoringEngine 加权评分（阈值 0.60）
  │                                    │
  │                      步骤D：保存证据图片
  │                                    │
  │←── 200 OK ─────────────────────────│
  │    is_valid: true/false            │
  │    confidence: 0.0~1.0             │
  │    message: "合规停车（综合评分 0.82）"    │
  │    vlm_analysis: {四维度状态, 评分, 原因} │
```

---

## 4. 服务端合规判断详解

### 4.1 预处理（自适应裁剪）

对上传图片按以下规则裁剪，聚焦地面停车细节：

```
center_y = h * 0.7        # 图片高度 70% 位置
box_h    = w / 3.0        # 裁剪高度为图片宽度的 1/3
y1 = center_y - box_h/2
y2 = center_y + box_h/2
```

### 4.2 CV 几何指标计算

```python
# 找到置信度最高的 Electric bike 作为主车辆
main_bike_mask = ...

# 计算与停车线的几何关系
parking_mask = combine_masks(objects, "parking lane")
iou, overlap = calculate_iou_and_overlap(main_bike_mask, parking_mask)

# 计算盲道重叠
tactile_mask = combine_masks(objects, "Tactile paving")
_, overlap_t = calculate_iou_and_overlap(main_bike_mask, tactile_mask)

# 初步推断
if overlap > 0.8:
    status_inference = "Likely Compliant (High Overlap)"
elif overlap < 0.1:
    status_inference = "Likely Out of Bounds"
```

以上数值以结构化 JSON 注入 VLM Prompt，作为辅助参考。

### 4.3 VLM 四维度合规判断

Prompt 模板 `cv_enhanced_p5` 要求 VLM 按四个维度输出状态标签：

| 维度 | 评估对象 | 可能的状态标签 |
|------|----------|----------------|
| 构图（composition） | 画面质量、车辆完整性、参照物存在 | `[合规]` / `[基本合规]` / `[不合规-构图]` / `[不合规-无参照]` |
| 角度（angle） | 车辆长轴与停车线/路缘的夹角 | `[合规]` / `[不合规-角度]` |
| 距离（distance） | 前后轮与停车边线的垂直距离 | `[完全合规]` / `[基本合规-压线]` / `[不合规-超界]` |
| 环境（context） | 停放介质（盲道、绿化带等禁停区） | `[合规]` / `[不合规-环境]` |

VLM 输出的 JSON 格式：

```json
{
  "step_by_step_analysis": {
    "cv_reference": "CV检测到parking lane（IoU=0.72），无盲道",
    "composition_check": "车辆完整可见，背景有停车线",
    "angle_analysis": "车辆长轴与停车线平行，偏差约10度",
    "distance_analysis": "前后轮均在停车线内，未压线",
    "context_analysis": "地面为硬化路面，无禁停特征"
  },
  "scores": {
    "composition_status": "[合规]",
    "angle_status": "[合规]",
    "distance_status": "[完全合规]",
    "context_status": "[合规]"
  }
}
```

### 4.4 ScoringEngine 加权评分

```
final_score = Σ (weight_i × score_i)

其中：
  构图权重 = 5%，  角度权重 = 25%
  距离权重 = 40%， 环境权重 = 30%

判定规则：
  - final_score ≥ 0.60 → 合规
  - final_score < 0.60 → 违规
  - 构图得分 = 0（门控触发）→ 直接判违规，不计算加权分

分数映射（部分）：
  构图：[合规]=1.0，[基本合规]=0.7，不合规=0.0
  角度：[合规]=1.0，[不合规-角度]=0.0
  距离：[完全合规]=1.0，[基本合规-压线]=0.0，[不合规-超界]=0.0
  环境：[合规]=1.0，[不合规-环境]=0.0
```

### 4.5 降级策略

VLM 不可用（未配置 API 密钥或调用失败）时，自动降级为规则判断：

| 条件 | 结论 | 置信度 |
|------|------|--------|
| 检测到盲道 | 违规 | 0.30 |
| 检测到停车线 | 合规 | 0.70 |
| 检测到马路牙子 | 合规 | 0.65 |
| 仅有车牌 | 合规（默认通过） | 0.50 |

---

## 5. 端云职责划分

| 职责 | 执行位置 | 说明 |
|------|----------|------|
| 车牌 OCR | **端侧**（MLKit） | 实时识别，3 帧防抖，无需网络 |
| 目标检测（引导阶段） | **云端**（YOLOv8-Seg） | 返回 bbox 坐标，轻量传输 |
| IoU 计算（引导阶段） | **端侧** | 基于云端返回的 bbox 在本地完成，节省传输 |
| 帧节流控制 | **端侧** | 500ms + AtomicBoolean，避免重叠请求 |
| 目标检测（判定阶段） | **云端**（YOLOv8-Seg） | 返回完整 Mask，用于几何计算 |
| 几何指标计算（判定阶段） | **云端** | IoU、重叠率，注入 VLM Prompt |
| 四维度合规判断 | **云端**（Qwen-VL-30B） | 结合原图 + 轮廓图 + CV 数据综合判断 |
| 加权评分 | **云端**（ScoringEngine） | 阈值 0.60 |
| 证据存储 | **云端** | 按合规/违规分目录存储原图 |

---

## 6. 关键设计决策

### 6.1 为何引导阶段只传 bbox，判定阶段才用完整 Mask？

引导阶段（每 500ms 一帧）追求低延迟，完整 Mask 序列化开销大；`/detect_static` 的 JSON 响应中 bbox 足够端侧计算 IoU 引导对准。判定阶段只触发一次，服务端直接调用 `predict()` 获取原始 Mask 对象进行精确几何计算。

### 6.2 为何不直接端侧运行 VLM？

30B 参数量的 Qwen-VL 无法在手机端运行。但 CV 几何预计算能消除 VLM 的视觉幻觉（直接告诉模型"IoU=0.72"比让它自己估算更准确），实验数据显示 CV+VLM 方案 F1 约 0.75，高于纯 VLM 的 0.71。

### 6.3 为何选择 cv_enhanced_p5？

实验中对比了多版 Prompt（standard_p*、cv_enhanced_p*），`cv_enhanced_p5` 明确了"CV 数据是辅助参考，以原图视觉事实为准"的优先级原则，避免 VLM 过度依赖可能不精确的几何数值，在 F1 和鲁棒性上综合表现最佳。
