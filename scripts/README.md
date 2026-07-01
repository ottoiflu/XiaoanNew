# scripts/ 脚本目录说明

> 入口脚本与工具集。不含库模块（库模块已迁至 modules/）。

## 入口脚本

| 脚本 | 用途 | 入口 | 依赖方向 |
|------|------|------|----------|
| `contrast_VLM_CV_test_v2.py` | **主实验入口**。YOLOv8-Seg + VLM 联合判定，线框轮廓输入 + IoU 预计算 | `uv run python scripts/contrast_VLM_CV_test_v2.py --config <yaml>` | modules/cv, modules/vlm, modules/experiment, modules/prompt |
| `contrast_VLM_test.py` | 纯 VLM 对照实验，无 CV 预处理 | `uv run python scripts/contrast_VLM_test.py` | modules/vlm, modules/experiment |
| `run_contrast_batch_v2.py` | 批量实验运行器，3 层缓存（YOLO 预计算 + VLM 去重 + JSON 持久化），覆盖 30 组实验矩阵 | `uv run python scripts/run_contrast_batch_v2.py` | modules/cv, modules/vlm, modules/experiment |
| `scoring_grid_search.py` | 加权评分网格搜索，覆盖权重/阈值/分数映射 | `uv run python scripts/scoring_grid_search.py` | modules/experiment/scoring.py |
| `yolov8_seg_batch.py` | YOLOv8-Seg 批量处理 | `uv run python scripts/yolov8_seg_batch.py` | modules/cv/yolov8_inference.py |
| `depth_pointcloud_demo.py` | YOLO 分割 + Depth Anything V2 深度估计 + Open3D 点云可视化 | `uv run python scripts/depth_pointcloud_demo.py` | modules/cv, transformers, open3d |
| `visualize_pointcloud_gui.py` | 点云 GUI 交互式查看 | `uv run python scripts/visualize_pointcloud_gui.py --ply <file>` | open3d |

## 工具脚本 (`scripts/tool/`)

| 脚本 | 用途 |
|------|------|
| `deploy_new_weights.py` | 训练权重自动部署到 `assets/weights/best.pt`，支持 `--watch` 模式 |
| `generate_charts.py` | 出版级实验图表生成（F1 柱状图、混淆矩阵、延迟对比等 9 张） |
| `apply_scene_review.py` | 场景标注审核 |
| `build_label_set.py` | 标注集构建 |
| `rescore_experiments.py` | 实验结果重新评分 |
| `labelme2yolo_seg.py` | LabelMe JSON 标注转 YOLO-Seg 格式 |
| `merge_union_masks.py` | 多类别掩膜并集合并 |
| `scene_classify.py` | 场景自动分类 |
| `sample_view.py` | 样本可视化预览 |
| `copy_sample_view.py` | 复制样本用于预览 |
| `debug_viewer.py` | 调试图片查看器 |
| `view_result_nolabel.py` | 无标签结果查看 |
| `batch_rotate_images.py` | 图片批量旋转 |
| `split_yes_dataset.py` | 合规数据集拆分 |

## _archived/（已弃用）

`contrast_VLM_CV_test.py`（v1 联合测试，被 v2 替代）、`run_contrast_batch.py`（v1 批量运行器，被 v2 替代）

## 命名规则

- 入口脚本：`<功能描述>_v<版本>.py`，v2 为当前活跃版本
- 工具脚本：`<动词>_<名词>.py`
- 入口脚本使用 `uv run` 启动，不可直接 `python`

## 注意事项

1. 所有脚本的路径解析基于 `settings.py` 中的 `PROJECT_ROOT`，不要依赖 `sys.path` 临时插入
2. 实验类脚本支持 `--config` 参数指定 YAML 配置
3. 批量运行器会缓存中间结果，清空缓存请删除 `outputs/contrast_experiments/` 下对应文件
