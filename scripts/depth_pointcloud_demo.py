"""
多类别掩膜深度估计与点云可视化展示脚本

对同一张图片，分别提取电动车、马路牙子、车道线的掩膜和点云，并生成全场景点云。
每张图片的输出统一存放在以图片名命名的子目录中。

输出结构：
    {output_dir}/{image_stem}/
        01_original.jpg              -- 原图
        02_segmentation.jpg          -- YOLO 实例分割可视化
        03_depth.jpg                 -- Depth Anything V2 热力图
        04_electric_bike_mask.jpg    -- 电动车掩膜
        04_electric_bike.ply         -- 电动车点云
        05_curb_mask.jpg             -- 马路牙子掩膜
        05_curb.ply                  -- 马路牙子点云
        06_parking_lane_mask.jpg     -- 车道线掩膜
        06_parking_lane.ply          -- 车道线点云
        07_scene.ply                 -- 全场景点云

使用方式：
    uv run python scripts/depth_pointcloud_demo.py --image <图片路径> --no-gui
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
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.cv.yolov8_inference import YOLOv8SegInference

# ============================================================
# 配置区域
# ============================================================
YOLO_WEIGHTS = str(PROJECT_ROOT / "assets" / "weights" / "best.pt")
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"
CONF_THRESHOLD = 0.3
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "depth_demo")
FOCAL_LENGTH_FACTOR = 0.8
MAX_POINTS_OBJECT = 200000
MAX_POINTS_SCENE = 500000

# 需要分别输出掩膜和点云的类别
TARGET_CLASSES = {
    0: {"name": "electric_bike", "label": "电动车", "index": 4, "color": (0, 255, 0)},
    1: {"name": "curb", "label": "马路牙子", "index": 5, "color": (255, 0, 255)},
    2: {"name": "parking_lane", "label": "车道线", "index": 6, "color": (255, 255, 0)},
}


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
    if len(pcd.points) > 0:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    return pcd


def compute_pca(pcd):
    """对点云执行 PCA，返回质心、特征值和特征向量（按特征值降序排列）"""
    points = np.asarray(pcd.points)
    if len(points) < 10:
        return None, None, None

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return centroid, eigenvalues, eigenvectors


def create_pca_arrows(centroid, eigenvectors, eigenvalues, scale_factor=1.0):
    """
    根据 PCA 结果创建 3 个方向箭头的 LineSet

    第一主成分（红色）代表主方向，第二（绿色）和第三（蓝色）为辅助轴。
    箭头长度按特征值的标准差比例缩放。
    """
    colors_map = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.4, 1.0],
    ]

    std_devs = np.sqrt(np.maximum(eigenvalues, 0))
    all_points = []
    all_lines = []
    all_colors = []

    for i in range(3):
        direction = eigenvectors[:, i]
        length = std_devs[i] * scale_factor

        tip_pos = centroid + direction * length
        tip_neg = centroid - direction * length

        base_idx = len(all_points)
        all_points.extend([centroid.tolist(), tip_pos.tolist(), tip_neg.tolist()])
        all_lines.append([base_idx, base_idx + 1])
        all_lines.append([base_idx, base_idx + 2])
        all_colors.append(colors_map[i])
        all_colors.append(colors_map[i])

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(all_points)
    lineset.lines = o3d.utility.Vector2iVector(all_lines)
    lineset.colors = o3d.utility.Vector3dVector(all_colors)
    return lineset


def render_mask_image(raw_img, mask, color, label):
    """将单类别掩膜叠加到原图上，生成可视化图像"""
    overlay = raw_img.copy()
    mask_region = mask.astype(bool)
    overlay[mask_region] = (
        overlay[mask_region].astype(np.float32) * 0.5 + np.array(color, dtype=np.float32) * 0.5
    ).astype(np.uint8)

    # 绘制掩膜轮廓
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr_color = (color[2], color[1], color[0])
    cv2.drawContours(overlay, contours, -1, bgr_color, 2)

    # 标注类别名
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.putText(
            overlay,
            label,
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            bgr_color,
            2,
        )
    return overlay


def merge_class_masks(objects, class_id):
    """将同一类别的多个实例掩膜合并为一个"""
    instances = [o for o in objects if o["category_id"] == class_id]
    if not instances:
        return None, []
    merged = instances[0]["mask"].copy()
    for inst in instances[1:]:
        merged = merged | inst["mask"]
    return merged.astype(bool), instances


def resolve_mask_priority(objects, priority_class_id=0, close_kernel_size=120, min_component_size=1000):
    """按优先级解决掩膜冲突，指定类别获得像素归属最高优先权

    处理逻辑:
    1. 对优先类别的掩膜做形态学闭运算，桥接因原型分辨率断裂的区域
    2. 闭运算区域中被其他类别占据的像素，强制归还给优先类别
    3. 从其他类别掩膜中移除被归还的像素
    """
    priority_instances = [o for o in objects if o["category_id"] == priority_class_id]
    if not priority_instances:
        return

    # 合并优先类别的所有实例
    priority_mask = priority_instances[0]["mask"].copy().astype(bool)
    for inst in priority_instances[1:]:
        priority_mask |= inst["mask"]

    # 检查是否有断裂
    u8 = priority_mask.astype(np.uint8) * 255
    n_components, _ = cv2.connectedComponents(u8)
    if n_components - 1 <= 1:
        return

    # 形态学闭运算桥接断裂
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)

    # 去除微小连通分量
    n_labels, labels = cv2.connectedComponents(closed)
    for i in range(1, n_labels):
        if (labels == i).sum() < min_component_size:
            closed[labels == i] = 0

    repaired = ndimage.binary_fill_holes(closed > 0)

    # 新增区域 = 修复后的掩膜 - 原始掩膜
    reclaimed = repaired & ~priority_mask
    n_reclaimed = reclaimed.sum()

    if n_reclaimed == 0:
        return

    # 更新优先类别实例的掩膜（将回收像素加到最高置信度实例上）
    best = max(priority_instances, key=lambda o: o["confidence"])
    best["mask"] = best["mask"] | reclaimed

    # 从其他类别掩膜中移除被回收的像素
    for o in objects:
        if o["category_id"] != priority_class_id:
            stolen = (o["mask"] & reclaimed).sum()
            if stolen > 0:
                o["mask"] = o["mask"] & ~reclaimed
                print(f"    优先级解决: 从 {o['label']} 回收 {stolen} px 归还 电动车")

    print(f"    掩膜修复: 回收 {n_reclaimed} px, 修复为 1 个连通区域")


def visualize_pointcloud(pcd, title, out_path, no_gui, extra_geometries=None):
    """可视化或离屏渲染保存点云截图"""
    if not no_gui:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(pcd)
        if extra_geometries:
            for geom in extra_geometries:
                vis.add_geometry(geom)
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
        print(f"    截图: {out_path}")
    else:
        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1280, height=720)
            vis.add_geometry(pcd)
            if extra_geometries:
                for geom in extra_geometries:
                    vis.add_geometry(geom)
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
            print(f"    离屏截图: {out_path}")
        except (RuntimeError, AttributeError, OSError) as e:
            print(f"    离屏渲染失败 ({e})，跳过截图")


def process_single_class(cls_id, cls_info, objects, raw_img, depth_map, focal, out_dir, stem, no_gui):
    """处理单个类别：合并掩膜、渲染掩膜图、生成点云、PCA 分析"""
    name = cls_info["name"]
    label = cls_info["label"]
    idx = cls_info["index"]
    color = cls_info["color"]

    print(f"\n  [{idx:02d}] {label} ({name})")

    merged_mask, instances = merge_class_masks(objects, cls_id)
    if merged_mask is None:
        print(f"    未检测到 {label}，跳过")
        return

    print(f"    检测到 {len(instances)} 个实例，掩膜像素: {merged_mask.sum()}")

    # 保存掩膜可视化
    mask_img = render_mask_image(raw_img, merged_mask, color, label)
    mask_path = str(out_dir / f"{idx:02d}_{name}_mask.jpg")
    cv2.imwrite(mask_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
    print(f"    掩膜图: {mask_path}")

    # 生成点云
    t0 = time.time()
    pcd = create_pointcloud(raw_img, depth_map, merged_mask, focal, max_points=MAX_POINTS_OBJECT)
    pcd = clean_pointcloud(pcd)
    n_points = len(pcd.points)
    print(f"    点云: {n_points} 点 ({time.time() - t0:.2f}s)")

    if n_points == 0:
        print("    点云为空，跳过保存")
        return

    # 保存 PLY
    ply_path = str(out_dir / f"{idx:02d}_{name}.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"    PLY: {ply_path}")

    # PCA 方向分析
    centroid, eigenvalues, eigenvectors = compute_pca(pcd)
    pca_geoms = []
    if centroid is not None:
        explained = eigenvalues / eigenvalues.sum() * 100
        pc1 = eigenvectors[:, 0]
        print(f"    PCA: PC1={explained[0]:.1f}%, 主方向=({pc1[0]:.4f}, {pc1[1]:.4f}, {pc1[2]:.4f})")

        pca_lineset = create_pca_arrows(centroid, eigenvectors, eigenvalues, scale_factor=3.0)
        pca_geoms.append(pca_lineset)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
        sphere.translate(centroid)
        sphere.paint_uniform_color([1.0, 1.0, 0.0])
        pca_geoms.append(sphere)

    # 点云可视化截图
    shot_path = str(out_dir / f"{idx:02d}_{name}_screenshot.png")
    visualize_pointcloud(pcd, f"{label} - {stem}", shot_path, no_gui, extra_geometries=pca_geoms)


def run_demo(image_path, no_gui=False):
    """执行完整的展示流程，分类别生成掩膜和点云"""
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[Error] 图片不存在: {image_path}")
        sys.exit(1)

    stem = img_path.stem
    out_dir = Path(OUTPUT_DIR) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ================================================================
    # Step 1: YOLO 分割检测
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 1: YOLOv8-Seg 分割检测")
    print("=" * 60)
    t0 = time.time()
    yolo = YOLOv8SegInference(YOLO_WEIGHTS, device=device, conf_threshold=CONF_THRESHOLD)
    result = yolo.predict(str(img_path), retina_masks=True)
    print(f"  推理耗时: {time.time() - t0:.2f}s")
    print(f"  检测到 {len(result['objects'])} 个目标:")
    for obj in result["objects"]:
        print(f"    - {obj['label']} (conf={obj['confidence']:.3f})")

    raw_img = result["image_raw"]
    seg_img = result["image_visual"]
    objects = result["objects"]
    H, W = raw_img.shape[:2]

    # ================================================================
    # Step 2: Depth Anything V2 深度估计
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 2: Depth Anything V2 深度估计 (Large)")
    print("=" * 60)
    t0 = time.time()
    depth_pipe = load_depth_model(DEPTH_MODEL_NAME, device)
    depth_map = estimate_depth(depth_pipe, Image.fromarray(raw_img))
    if depth_map.shape != (H, W):
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
    print(f"  耗时: {time.time() - t0:.2f}s")
    print(f"  深度图尺寸: {depth_map.shape}")

    focal = max(H, W) * FOCAL_LENGTH_FACTOR

    # ================================================================
    # Step 3: 保存公共输出（原图、分割图、深度图）
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 3: 保存公共输出")
    print("=" * 60)

    p1 = str(out_dir / "01_original.jpg")
    cv2.imwrite(p1, cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
    print(f"  [01] 原图: {p1}")

    p2 = str(out_dir / "02_segmentation.jpg")
    cv2.imwrite(p2, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    print(f"  [02] 分割图: {p2}")

    depth_vis = (depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    p3 = str(out_dir / "03_depth.jpg")
    cv2.imwrite(p3, depth_colored)
    print(f"  [03] 深度图: {p3}")

    # ================================================================
    # Step 4: 逐类别生成掩膜、点云、PCA
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 4: 逐类别处理（掩膜 + 点云 + PCA）")
    print("=" * 60)

    # 电动车掩膜优先级最高，先解决跨类别的像素冲突
    resolve_mask_priority(objects, priority_class_id=0)

    for cls_id, cls_info in TARGET_CLASSES.items():
        process_single_class(cls_id, cls_info, objects, raw_img, depth_map, focal, out_dir, stem, no_gui)

    # ================================================================
    # Step 5: 全场景点云
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 5: 全场景点云")
    print("=" * 60)
    t0 = time.time()
    full_mask = np.ones((H, W), dtype=bool)
    pcd_scene = create_pointcloud(raw_img, depth_map, full_mask, focal, max_points=MAX_POINTS_SCENE)
    pcd_scene = clean_pointcloud(pcd_scene)
    print(f"  全场景点云: {len(pcd_scene.points)} 点 ({time.time() - t0:.2f}s)")

    p_scene = str(out_dir / "07_scene.ply")
    o3d.io.write_point_cloud(p_scene, pcd_scene)
    print(f"  PLY: {p_scene}")

    scene_shot = str(out_dir / "07_scene_screenshot.png")
    visualize_pointcloud(pcd_scene, f"Scene - {stem}", scene_shot, no_gui)

    # ================================================================
    # 输出汇总
    # ================================================================
    print("\n" + "=" * 60)
    print(f"全部完成，输出目录: {out_dir}")
    print("=" * 60)
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} {size_kb:8.1f} KB")
    print("=" * 60)


def main():
    """命令行入口"""
    global YOLO_WEIGHTS, DEPTH_MODEL_NAME, CONF_THRESHOLD, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="多类别掩膜深度估计与点云可视化")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--no-gui", action="store_true", help="无 GUI 模式（离屏渲染）")
    parser.add_argument("--weights", type=str, default=YOLO_WEIGHTS, help="YOLO 权重路径")
    parser.add_argument("--depth-model", type=str, default=DEPTH_MODEL_NAME, help="Depth Anything 模型名")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="置信度阈值")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="输出根目录")

    args = parser.parse_args()

    YOLO_WEIGHTS = args.weights
    DEPTH_MODEL_NAME = args.depth_model
    CONF_THRESHOLD = args.conf
    OUTPUT_DIR = args.output

    run_demo(args.image, no_gui=args.no_gui)


if __name__ == "__main__":
    main()
