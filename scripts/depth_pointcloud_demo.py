"""
电动车掩膜深度估计与点云可视化展示脚本

对同一张图片生成 5 个输出：
1. 原图
2. 分割图（YOLO 实例分割可视化）
3. 深度图（Depth Anything V2 热力图）
4. 单车点云（仅电动车掩膜区域的 3D 点云）
5. 全场景点云（整张图片的 3D 点云）

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
MAX_POINTS_BIKE = 200000
MAX_POINTS_SCENE = 500000


def load_depth_model(model_name, device):
    """加载 Depth Anything V2 深度估计模型"""
    from transformers import pipeline

    print(f"[DepthAnything] 正在加载模型: {model_name}")
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


def create_pointcloud(rgb, depth, mask, focal_length, depth_scale=5.0, max_points=200000):
    """
    将 RGB + Depth 在 mask 区域内转换为 3D 点云

    参数:
        rgb: (H, W, 3) uint8 RGB
        depth: (H, W) float32 归一化深度 [0, 1]
        mask: (H, W) bool 有效区域
        focal_length: 虚拟焦距
        depth_scale: 深度缩放因子
        max_points: 最大点数限制
    """
    H, W = depth.shape
    cx, cy = W / 2.0, H / 2.0
    fx = fy = focal_length

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return o3d.geometry.PointCloud()

    if len(xs) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(xs), max_points, replace=False)
        xs, ys = xs[indices], ys[indices]

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


def clean_pointcloud(pcd, nb_neighbors=20, std_ratio=2.0):
    """统计滤波去除离群点，并估算法线"""
    if len(pcd.points) > 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    return pcd


def visualize_pointcloud(pcd, title, out_path, no_gui):
    """可视化或离屏渲染保存点云截图"""
    if not no_gui:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.05, 0.05, 0.05])
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        ctr.set_zoom(0.6)
        vis.run()
        vis.capture_screen_image(out_path, do_render=True)
        vis.destroy_window()
        print(f"  截图: {out_path}")
    else:
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1280, height=720)
            vis.add_geometry(pcd)
            opt = vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.array([0.05, 0.05, 0.05])
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(0.6)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(out_path, do_render=True)
            vis.destroy_window()
            print(f"  离屏截图: {out_path}")
        except Exception as e:
            print(f"  离屏渲染失败 ({e})，跳过截图")


def run_demo(image_path, no_gui=False):
    """执行完整的展示流程，生成 5 个输出"""
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[Error] 图片不存在: {image_path}")
        sys.exit(1)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ================================================================
    # Step 1: YOLO 分割检测
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 1: YOLOv8-Seg 分割检测")
    print("=" * 60)
    t0 = time.time()
    yolo = YOLOv8SegInference(YOLO_WEIGHTS, device=device, conf_threshold=CONF_THRESHOLD)
    result = yolo.predict(str(img_path))
    print(f"  推理耗时: {time.time() - t0:.2f}s")
    print(f"  检测到 {len(result['objects'])} 个目标:")
    for obj in result["objects"]:
        print(f"    - {obj['label']} (conf={obj['confidence']:.3f})")

    ebikes = [o for o in result["objects"] if o["category_id"] == ELECTRIC_BIKE_CLASS_ID]
    if not ebikes:
        print("[Error] 未检测到电动车，无法继续")
        sys.exit(1)

    best_bike = max(ebikes, key=lambda o: o["confidence"])
    bike_mask = best_bike["mask"]  # (H, W) bool
    raw_img = result["image_raw"]  # (H, W, 3) RGB uint8
    seg_img = result["image_visual"]  # (H, W, 3) RGB 分割可视化
    H, W = raw_img.shape[:2]

    # ================================================================
    # Step 2: Depth Anything V2 深度估计
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 2: Depth Anything V2 深度估计")
    print("=" * 60)
    t0 = time.time()
    depth_pipe = load_depth_model(DEPTH_MODEL_NAME, device)
    depth_map = estimate_depth(depth_pipe, Image.fromarray(raw_img))
    if depth_map.shape != (H, W):
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
    print(f"  耗时: {time.time() - t0:.2f}s")
    print(f"  深度图尺寸: {depth_map.shape}")

    # ================================================================
    # Step 3: 生成两份点云
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 3: 生成点云")
    print("=" * 60)
    focal = max(H, W) * FOCAL_LENGTH_FACTOR

    # 3a: 单车点云（仅电动车区域）
    t0 = time.time()
    pcd_bike = create_pointcloud(raw_img, depth_map, bike_mask, focal, max_points=MAX_POINTS_BIKE)
    pcd_bike = clean_pointcloud(pcd_bike)
    print(f"  单车点云: {len(pcd_bike.points)} 点 ({time.time() - t0:.2f}s)")

    # 3b: 全场景点云（整张图片）
    t0 = time.time()
    full_mask = np.ones((H, W), dtype=bool)
    pcd_scene = create_pointcloud(raw_img, depth_map, full_mask, focal, max_points=MAX_POINTS_SCENE)
    pcd_scene = clean_pointcloud(pcd_scene)
    print(f"  全场景点云: {len(pcd_scene.points)} 点 ({time.time() - t0:.2f}s)")

    # ================================================================
    # Step 4: 保存 5 个输出
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 4: 保存 5 个输出")
    print("=" * 60)

    # [1] 原图
    p1 = str(out_dir / f"{stem}_1_original.jpg")
    cv2.imwrite(p1, cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
    print(f"  [1/5] 原图: {p1}")

    # [2] 分割图
    p2 = str(out_dir / f"{stem}_2_segmentation.jpg")
    cv2.imwrite(p2, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    print(f"  [2/5] 分割图: {p2}")

    # [3] 深度图（INFERNO 热力图）
    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    p3 = str(out_dir / f"{stem}_3_depth.jpg")
    cv2.imwrite(p3, depth_colored)
    print(f"  [3/5] 深度图: {p3}")

    # [4] 单车点云 PLY
    p4 = str(out_dir / f"{stem}_4_bike_pointcloud.ply")
    o3d.io.write_point_cloud(p4, pcd_bike)
    print(f"  [4/5] 单车点云: {p4}")

    # [5] 全场景点云 PLY
    p5 = str(out_dir / f"{stem}_5_scene_pointcloud.ply")
    o3d.io.write_point_cloud(p5, pcd_scene)
    print(f"  [5/5] 全场景点云: {p5}")

    # ================================================================
    # Step 5: 可视化（可选）
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 5: 点云可视化")
    print("=" * 60)
    if no_gui:
        print("  无 GUI 模式，尝试离屏渲染...")

    bike_shot = str(out_dir / f"{stem}_4_bike_pointcloud_screenshot.png")
    visualize_pointcloud(pcd_bike, f"Bike Point Cloud - {stem}", bike_shot, no_gui)

    scene_shot = str(out_dir / f"{stem}_5_scene_pointcloud_screenshot.png")
    visualize_pointcloud(pcd_scene, f"Scene Point Cloud - {stem}", scene_shot, no_gui)

    print("\n" + "=" * 60)
    print("展示完成，输出文件清单：")
    print(f"  [1] 原图:       {p1}")
    print(f"  [2] 分割图:     {p2}")
    print(f"  [3] 深度图:     {p3}")
    print(f"  [4] 单车点云:   {p4}")
    print(f"  [5] 全场景点云: {p5}")
    print(f"  输出目录: {out_dir}")
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
