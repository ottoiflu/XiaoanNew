"""
电动车掩膜深度估计与点云可视化展示脚本

流程：
1. 调用现有 YOLOv8-Seg 模型检测电动车并提取掩膜
2. 使用 Depth Anything V2 对电动车区域进行单目深度估计
3. 将 RGB + Depth 转换为 3D 点云
4. 使用 Open3D 进行点云可视化，同时保存为 PLY 文件

使用方式：
    python scripts/depth_pointcloud_demo.py --image <图片路径>
    python scripts/depth_pointcloud_demo.py --image <图片路径> --no-gui
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.cv.yolov8_inference import YOLOv8SegInference

# ============================================================
# 配置区域
# ============================================================
YOLO_WEIGHTS = str(PROJECT_ROOT / "assets" / "weights" / "best.pt")
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
CONF_THRESHOLD = 0.5
ELECTRIC_BIKE_CLASS_ID = 0
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "depth_demo")
FOCAL_LENGTH_FACTOR = 0.8


def load_depth_model(model_name, device):
    """加载 Depth Anything V2 深度估计模型"""
    from transformers import pipeline

    print(f"[DepthAnything] 正在加载模型: {model_name}")
    print(f"[DepthAnything] 使用设备: {device}")

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = pipeline(
        "depth-estimation",
        model=model_name,
        device=device,
        torch_dtype=dtype,
    )
    print("[DepthAnything] 模型加载完成")
    return pipe


def estimate_depth(depth_pipe, image):
    """对输入 PIL 图像执行深度估计，返回归一化 [0, 1] 深度图"""
    result = depth_pipe(image)
    depth_map = np.array(result["depth"], dtype=np.float32)

    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min > 1e-6:
        depth_map = (depth_map - d_min) / (d_max - d_min)
    else:
        depth_map = np.zeros_like(depth_map)

    return depth_map


def create_pointcloud_from_rgbd(rgb, depth, mask, focal_length, depth_scale=5.0):
    """
    将 RGB 图像和深度图在掩膜区域内转换为 3D 点云

    参数:
        rgb: (H, W, 3) uint8 RGB图像
        depth: (H, W) float32 归一化深度图 [0, 1]
        mask: (H, W) bool 掩膜区域
        focal_length: 虚拟焦距
        depth_scale: 深度缩放因子
    """
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    fx = fy = focal_length

    ys, xs = np.where(mask)
    if len(xs) == 0:
        print("[Warning] 掩膜区域为空，无法生成点云")
        return o3d.geometry.PointCloud()

    # 下采样防止卡顿
    max_points = 200000
    if len(xs) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(xs), max_points, replace=False)
        xs = xs[indices]
        ys = ys[indices]

    z = depth[ys, xs] * depth_scale
    z = depth_scale - z + 0.1

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)
    colors = rgb[ys, xs].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def save_visualization_images(output_dir, image_name, raw_img, mask, masked_rgb, depth_map, depth_colored):
    """保存中间结果可视化图片"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem

    # 原图 + 掩膜叠加
    overlay = raw_img.copy()
    mask_vis = np.zeros_like(raw_img)
    mask_vis[mask] = [0, 255, 0]
    overlay = cv2.addWeighted(overlay, 0.7, mask_vis, 0.3, 0)
    cv2.imwrite(str(out / f"{stem}_01_mask_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 掩膜裁剪后的 RGB
    cv2.imwrite(str(out / f"{stem}_02_masked_rgb.jpg"), cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))

    # 深度图热力图
    cv2.imwrite(str(out / f"{stem}_03_depth_heatmap.jpg"), depth_colored)

    # 合并对比图
    h, w = raw_img.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    canvas[:, w : w * 2] = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    depth_resized = cv2.resize(depth_colored, (w, h))
    canvas[:, w * 2 :] = depth_resized
    cv2.imwrite(str(out / f"{stem}_04_comparison.jpg"), canvas)

    print(f"[IO] 可视化图片已保存到: {out}")


def run_demo(image_path, no_gui=False):
    """执行完整的展示流程"""
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[Error] 图片不存在: {image_path}")
        sys.exit(1)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Step 1: YOLO 分割检测 ----
    print("\n" + "=" * 60)
    print("Step 1: YOLOv8-Seg 分割检测")
    print("=" * 60)
    t0 = time.time()
    yolo = YOLOv8SegInference(YOLO_WEIGHTS, device=device, conf_threshold=CONF_THRESHOLD)
    result = yolo.predict(str(img_path))
    print(f"  推理耗时: {time.time() - t0:.2f}s")
    print(f"  检测到 {len(result['objects'])} 个目标:")
    for obj in result["objects"]:
        print(f"    - {obj['label']} (conf={obj['confidence']:.3f}, area={obj['area_ratio']:.4f})")

    ebikes = [o for o in result["objects"] if o["category_id"] == ELECTRIC_BIKE_CLASS_ID]
    if not ebikes:
        print("[Error] 未检测到电动车，无法继续")
        sys.exit(1)

    best_bike = max(ebikes, key=lambda o: o["confidence"])
    print(f"\n  选定目标: {best_bike['label']} (conf={best_bike['confidence']:.3f})")

    bike_mask = best_bike["mask"]
    raw_img = result["image_raw"]
    H, W = raw_img.shape[:2]

    masked_rgb = raw_img.copy()
    masked_rgb[~bike_mask] = 0

    # ---- Step 2: Depth Anything 深度估计 ----
    print("\n" + "=" * 60)
    print("Step 2: Depth Anything V2 深度估计")
    print("=" * 60)
    t0 = time.time()
    depth_pipe = load_depth_model(DEPTH_MODEL_NAME, device)

    pil_img = Image.fromarray(raw_img)
    depth_map = estimate_depth(depth_pipe, pil_img)

    if depth_map.shape != (H, W):
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

    print(f"  深度估计耗时: {time.time() - t0:.2f}s")
    print(f"  深度图尺寸: {depth_map.shape}")
    print(f"  掩膜区域深度范围: [{depth_map[bike_mask].min():.3f}, {depth_map[bike_mask].max():.3f}]")

    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    depth_colored_masked = depth_colored.copy()
    depth_colored_masked[~bike_mask] = (depth_colored_masked[~bike_mask] * 0.2).astype(np.uint8)

    # ---- Step 3: 生成点云 ----
    print("\n" + "=" * 60)
    print("Step 3: 生成 3D 点云")
    print("=" * 60)
    t0 = time.time()
    focal_length = max(H, W) * FOCAL_LENGTH_FACTOR
    pcd = create_pointcloud_from_rgbd(raw_img, depth_map, bike_mask, focal_length)
    print(f"  点云生成耗时: {time.time() - t0:.2f}s")
    print(f"  点云点数: {len(pcd.points)}")

    if len(pcd.points) > 100:
        pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        removed = len(pcd.points) - len(pcd_clean.points)
        print(f"  统计滤波移除离群点: {removed}")
        pcd = pcd_clean

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    # ---- Step 4: 保存结果 ----
    print("\n" + "=" * 60)
    print("Step 4: 保存结果")
    print("=" * 60)

    stem = img_path.stem
    ply_path = str(out_dir / f"{stem}_pointcloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"  点云文件: {ply_path}")

    save_visualization_images(
        str(out_dir), img_path.name, raw_img, bike_mask, masked_rgb, depth_map, depth_colored_masked
    )

    # ---- Step 5: 可视化 ----
    if not no_gui:
        print("\n" + "=" * 60)
        print("Step 5: Open3D 点云可视化")
        print("=" * 60)
        print("  即将打开 3D 可视化窗口...")
        print("  操作: 左键旋转 | 滚轮缩放 | 中键平移 | Q退出")

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Electric Bike Point Cloud - {img_path.name}",
            width=1280,
            height=720,
        )
        vis.add_geometry(pcd)

        render_opt = vis.get_render_option()
        render_opt.point_size = 2.0
        render_opt.background_color = np.array([0.05, 0.05, 0.05])

        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.6)

        vis.run()

        screenshot_path = str(out_dir / f"{stem}_pointcloud_screenshot.png")
        vis.capture_screen_image(screenshot_path, do_render=True)
        print(f"  截图已保存: {screenshot_path}")
        vis.destroy_window()
    else:
        print("\n[Info] 无 GUI 模式，跳过交互式可视化")
        print("[Info] 尝试离屏渲染截图...")
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1280, height=720)
            vis.add_geometry(pcd)

            render_opt = vis.get_render_option()
            render_opt.point_size = 2.0
            render_opt.background_color = np.array([0.05, 0.05, 0.05])

            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(0.6)

            vis.poll_events()
            vis.update_renderer()

            screenshot_path = str(out_dir / f"{stem}_pointcloud_screenshot.png")
            vis.capture_screen_image(screenshot_path, do_render=True)
            print(f"  离屏截图已保存: {screenshot_path}")
            vis.destroy_window()
        except Exception as e:
            print(f"  离屏渲染失败 ({e})，跳过截图")

    print("\n" + "=" * 60)
    print("展示完成")
    print(f"所有输出保存在: {out_dir}")
    print("=" * 60)


def main():
    global YOLO_WEIGHTS, DEPTH_MODEL_NAME, CONF_THRESHOLD, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="电动车掩膜深度估计与点云可视化展示")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--no-gui", action="store_true", help="无 GUI 模式")
    parser.add_argument("--weights", type=str, default=YOLO_WEIGHTS, help="YOLO 权重路径")
    parser.add_argument("--depth-model", type=str, default=DEPTH_MODEL_NAME, help="Depth Anything 模型名")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="置信度阈值")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="输出目录")

    args = parser.parse_args()

    YOLO_WEIGHTS = args.weights
    DEPTH_MODEL_NAME = args.depth_model
    CONF_THRESHOLD = args.conf
    OUTPUT_DIR = args.output

    run_demo(args.image, no_gui=args.no_gui)


if __name__ == "__main__":
    main()
