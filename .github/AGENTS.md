# XiaoanNew - 共享单车停放检测系统

## 项目概述

本项目是一个基于深度学习的共享单车智能停放检测系统，主要功能包括：

1. 实时掩膜分割：检测电动车、停车线、马路牙子、盲道等目标
2. 停车合规性判断：结合 CV 检测结果与云端 VLM 进行综合判定
3. 数据采集与管理：支持通过 API 上传图片并记录标注

## 项目结构

```
XiaoanNew/
├── app.py                    # Flask 后端 API 服务入口
├── mask_inference.py         # MaskRCNN 推理模块（备用）
├── scripts/                  # 脚本工具目录
│   ├── yolov8_seg_inference.py    # YOLOv8-Seg 推理模块（主用）
│   ├── yolov8_seg_batch.py        # 批量处理脚本
│   ├── contrast_VLM_CV_test.py    # VLM+CV 联合测试脚本
│   └── contrast_VLM_CV_test_v2.py # 联合测试脚本（轮廓版）
├── yolo/                     # YOLO 训练相关
│   ├── train_yolov8_seg.py        # YOLOv8 实例分割训练脚本
│   └── data/                      # 数据集配置
├── MMLab/                    # OpenMMLab 生态环境（MMDetection/MMYOLO）
├── App_collected_dataset/    # 采集的数据集
├── experiment_outputs/       # 实验结果输出（CSV 格式）
└── yolov8seg_visuals*/       # 分割可视化结果
```

## 类别定义

| ID | 名称 | 描述 |
|----|------|------|
| 0 | Electric bike | 电动车 |
| 1 | Curb | 马路牙子 |
| 2 | parking lane | 停车线 |
| 3 | Tactile paving | 盲道 |

## 代码规范

1. Python 代码遵循 PEP8 规范
2. 函数和类需包含中文文档字符串
3. 重要的配置参数应集中在脚本顶部的「配置区域」中定义
4. 模型权重路径使用绝对路径，便于跨目录调用

## 开发约定

1. 推理模块应提供统一接口：`predict()` 返回结构化字典，`predict_memory()` 返回 PNG 字节流
2. 新增功能时优先扩展 `YOLOv8SegInference` 类，保持接口向后兼容
3. 实验脚本命名格式：`{功能}_{模型}_{版本}.py`
4. 实验输出 CSV 命名格式：`results_{model}_{size}_q{quality}_p{prompt_id}_detailed.csv`

## 运行环境

1. Python 3.8+
2. PyTorch (CUDA 支持)
3. Ultralytics YOLO: `pip install ultralytics`
4. Flask: `pip install flask`
5. OpenCV: `pip install opencv-python`
6. OpenAI SDK: `pip install openai`

## 常用命令

```bash
# 启动后端服务
python app.py

# 运行 VLM+CV 联合测试
cd scripts && python contrast_VLM_CV_test_v2.py

# 批量处理图片
cd scripts && python yolov8_seg_batch.py <input_folder> --output <output_folder>

# 训练 YOLOv8-Seg 模型
cd yolo && python train_yolov8_seg.py
```

## API 端点

| 路径 | 方法 | 功能 |
|------|------|------|
| `/api/collect/upload` | POST | 数据采集上传 |
| `/api/segmentation/detect` | POST | 实时掩膜分割（返回 PNG） |
| `/api/segmentation/detect_static` | POST | 静态图片分析（返回 JSON） |
| `/api/test/check_parking` | POST | 停车检测（OCR + CV） |
| `/api/health` | GET | 健康检查 |

## 注意事项

1. 模型权重文件较大，不纳入 Git 版本控制
2. API Key 等敏感信息应通过环境变量或独立配置文件管理
3. 实验输出 CSV 保留作为性能对比参考，修改实验配置前先备份
4. 云端 VLM 调用有频率限制，批量测试时注意控制并发数
