# XiaoAnEbike 项目技术架构总结

## 1. 项目概述
本项目是一个基于 **Android (端)** + **Flask/PyTorch (云)** 的端云协同 AI 应用。主要用于电动车场景下的数据采集、实时掩膜分割检测以及停车规范性验证。

*   **核心架构**: Client-Server (C/S) 架构。
*   **通信协议**: HTTP (RESTful API), Multipart/form-data 文件传输。
*   **开发标准**: Modern Android Development (MAD), Python Flask 后端。

---

## 2. 服务端设计 (Server-Side)

### 2.1 技术栈
*   **语言**: Python 3.x
*   **Web 框架**: Flask (轻量级，易于部署)
*   **AI 框架**: PyTorch, Torchvision
*   **图像处理**: PIL (Pillow), OpenCV (辅助), Numpy
*   **部署环境**: Linux 服务器 (推荐使用 `tmux` 守护进程)

### 2.2 文件结构
```text
server/
├── app.py                # 核心入口，路由定义，业务逻辑
├── inference.py          # AI 推理引擎封装 (Mask R-CNN)
├── MaskRCNN_Xiaoan.pth   # 训练好的模型权重
├── App_collected_dataset/# [功能1] 存储采集的数据集
│   ├── test/             # 标签文件夹
│   ├── data/             # 普通文件夹
│   └── custom_path/      # 自定义文件夹
└── temp_processing/      # [临时] 调试用的中间文件
```

### 2.3 核心 API 接口

| 功能模块 | 接口路径 (Endpoint) | 请求方式 | 参数 (Multipart) | 返回类型 | 核心逻辑 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. 数据采集** | `/api/collect/upload` | POST | `file`, `label`, `date`, `custom_path`, `ground_truth` | JSON | 保存图片至指定目录；若为验证模式，追加写入 `labels.txt`。 |
| **2. 掩膜分割** | `/api/segmentation/detect` | POST | `file` | Image Stream (PNG) | **全内存处理**。不存盘。返回包含 Mask+Box+Text 的**透明 PNG**。 |
| **3. 停车检测** | `/api/test/check_parking` | POST | `file` | JSON | 模拟/推理停车规范。返回 `{is_valid, confidence, message}`。保存证据图。 |

### 2.4 关键优化技术
1.  **全局模型单例**: 在 `app.py` 启动时初始化 `inference.py`，避免每次请求重新加载模型（耗时从 2s -> 0.2s）。
2.  **全内存操作**: 使用 `io.BytesIO` 替代磁盘读写，极大降低 I/O 延迟。
3.  **透明图层回传**: 在掩膜检测中，只回传“透明背景+掩膜”的 PNG 图片，降低带宽占用，便于前端叠加显示。
4.  **容错机制**: 针对 Android 端可能漏传的参数（如 `ground_truth`）做了宽容处理和详细日志记录。

---

## 3. 客户端设计 (Android App)

### 3.1 技术栈
*   **语言**: Kotlin
*   **UI 框架**: Jetpack Compose (Material Design 3)
*   **架构模式**: MVVM (Model-View-ViewModel)
*   **网络库**: Retrofit 2 + OkHttp 3
*   **相机库**: CameraX (ImageCapture & ImageAnalysis)
*   **异步处理**: Kotlin Coroutines (协程) + Flow

### 3.2 模块详细设计

#### 模块一：数据采集与打标 (Data Collection)
*   **UI**: `DataCollectionActivity` / `CollectScreen`
    *   **入口**: 模式选择页 (标准模式 vs 验证模式)。
    *   **弹窗**: 拍照后弹出。支持选择分辨率 (Original/1080p/720p)、路径 (Test/Data/Custom)。
    *   **验证模式**: 强制要求用户选择 **YES/NO (真值)** 后才能上传。
*   **逻辑**: `CollectViewModel`
    *   管理模式状态、图片压缩。
    *   构建复杂的 Multipart 请求（处理 Nullable 参数）。

#### 模块二：实时掩膜分割 (Mask Segmentation)
*   **UI**: `MaskSegmentationActivity`
    *   **布局**: `Box` 布局。底层是相机预览 (`PreviewView`)，顶层是覆盖层 (`Image`)。
    *   **显示**: 接收服务器返回的透明 PNG，使用 `ContentScale.FillBounds` 铺满屏幕与相机对齐。
*   **逻辑**: `SegmentationViewModel`
    *   **图片分析**: 使用 CameraX `ImageAnalysis`。
    *   **预处理**: 获取 `RotationDegrees`，**旋转图片**至竖屏，**压缩**至 320x240 或 640x480。
    *   **背压控制 (Backpressure)**: 使用 `AtomicBoolean isProcessing` 锁，确保上一帧处理完之前不发送下一帧，防止卡顿。

#### 模块三：停车规范检测 (Parking Check)
*   **UI**: `ParkingCheckActivity`
    *   简单拍照界面。
*   **逻辑**:
    *   上传图片，解析服务器返回的 **JSON**。
    *   根据 `is_valid` 字段显示绿色成功或红色警告弹窗。

### 3.3 关键类与文件
*   **`ApiService.kt`**: 定义所有 Retrofit 接口。
*   **`RetrofitInstance.kt`**: 单例网络客户端，配置 BaseURL。
*   **`CollectViewModel.kt`**: 核心逻辑，处理压缩、参数构建、异步上传。
*   **`libs.versions.toml`**: 统一管理依赖版本。

---

## 4. 性能与体验优化总结 (Highlights)

1.  **解决“图片横向”问题**:
    *   Android 端在 `ImageAnalysis` 中读取旋转角度，使用 `Matrix` 旋转 Bitmap 后再上传。

2.  **解决“实时回传卡顿”问题**:
    *   **策略 A**: 限制频率（AtomicBoolean 锁）。
    *   **策略 B**: 降低分辨率（上传 320x240 或 640x480）。
    *   **策略 C**: 减小回包大小（服务器只回传透明掩膜 PNG，不回传原图）。

3.  **解决“参数丢失”问题**:
    *   在 ViewModel 中使用“变量捕获”方式，确保在协程异步执行时，参数值不会被 UI 关闭操作提前清空。

---

## 5. 下一步建议 (Next Steps)

1.  **服务器**: 将 `Flask` 开发服务器部署为 `Gunicorn` 或 `uWSGI` 生产级服务。
2.  **协议**: 考虑将“实时分割”模块升级为 **WebSocket** 通信，进一步降低握手延迟。
3.  **端侧部署**: 若对实时性要求极高（>20 FPS），建议将 PyTorch 模型导出为 TFLite/ONNX，直接在 Android 手机 NPU 上运行，完全脱离网络依赖。