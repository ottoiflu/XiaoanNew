"""
点云 GUI 可视化脚本

在有显示器/X11 的环境中交互式查看 PLY 点云文件，
并在单车点云上叠加 PCA 方向轴（红/绿/蓝三轴 + 黄色质心）。

使用方式：
    python scripts/visualize_pointcloud_gui.py --ply <PLY文件路径>
    python scripts/visualize_pointcloud_gui.py --ply <PLY文件路径> --no-pca
    python scripts/visualize_pointcloud_gui.py --dir <depth_demo输出目录>

快捷键（Open3D 可视化窗口内）：
    鼠标左键拖拽    旋转视角
    鼠标滚轮        缩放
    鼠标右键拖拽    平移
    R               重置视角
    Q / Esc         退出
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


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


def create_pca_geometries(centroid, eigenvectors, eigenvalues, scale_factor=3.0):
    """
    根据 PCA 结果创建方向轴 LineSet 和质心球

    返回几何体列表，可直接添加到 Open3D 可视化窗口。
    PC1（红）= 主方向/朝向，PC2（绿），PC3（蓝）。
    """
    colors_map = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.4, 1.0],
    ]
    labels = ["PC1（主方向）", "PC2", "PC3"]

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

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    sphere.translate(centroid)
    sphere.paint_uniform_color([1.0, 1.0, 0.0])

    explained = eigenvalues / eigenvalues.sum() * 100
    print("  PCA 分析结果：")
    print(f"    质心: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
    for i in range(3):
        v = eigenvectors[:, i]
        print(f"    {labels[i]}: ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})  方差贡献={explained[i]:.1f}%")

    return [lineset, sphere]


def visualize_single_ply(ply_path, show_pca=True):
    """加载并可视化单个 PLY 文件"""
    pcd = o3d.io.read_point_cloud(ply_path)
    n_points = len(pcd.points)
    if n_points == 0:
        print(f"[Error] 点云为空: {ply_path}")
        sys.exit(1)

    print(f"  已加载: {ply_path}")
    print(f"  点数: {n_points}")

    geometries = [pcd]

    if show_pca:
        centroid, eigenvalues, eigenvectors = compute_pca(pcd)
        if centroid is not None:
            pca_geoms = create_pca_geometries(centroid, eigenvectors, eigenvalues)
            geometries.extend(pca_geoms)
        else:
            print("  点云过少，跳过 PCA")

    title = Path(ply_path).stem
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=720)
    for geom in geometries:
        vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.show_coordinate_frame = True

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.6)

    print(f"\n  窗口已打开: {title}")
    print("  操作: 左键旋转 | 滚轮缩放 | 右键平移 | Q 退出")
    vis.run()
    vis.destroy_window()


def visualize_directory(dir_path, show_pca=True):
    """可视化 depth_demo 输出目录中的所有 PLY 文件"""
    ply_files = sorted(Path(dir_path).glob("*.ply"))
    if not ply_files:
        print(f"[Error] 目录中未找到 PLY 文件: {dir_path}")
        sys.exit(1)

    print(f"  找到 {len(ply_files)} 个 PLY 文件:")
    for p in ply_files:
        print(f"    - {p.name}")

    for ply in ply_files:
        is_bike = "bike" in ply.name.lower()
        print(f"\n{'=' * 50}")
        print(f"  正在可视化: {ply.name}")
        print(f"{'=' * 50}")
        visualize_single_ply(str(ply), show_pca=show_pca and is_bike)


def main():
    parser = argparse.ArgumentParser(
        description="点云 GUI 可视化（支持 PCA 方向轴叠加）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  查看单个 PLY:
    python scripts/visualize_pointcloud_gui.py --ply outputs/depth_demo/xxx_4_bike_pointcloud.ply

  查看整个输出目录:
    python scripts/visualize_pointcloud_gui.py --dir outputs/depth_demo

  不显示 PCA 轴:
    python scripts/visualize_pointcloud_gui.py --ply xxx.ply --no-pca
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ply", type=str, help="单个 PLY 文件路径")
    group.add_argument("--dir", type=str, help="depth_demo 输出目录路径")
    parser.add_argument("--no-pca", action="store_true", help="不显示 PCA 方向轴")
    parser.add_argument("--point-size", type=float, default=2.0, help="点大小 (默认 2.0)")

    args = parser.parse_args()

    print("=" * 50)
    print("  点云 GUI 可视化工具")
    print("=" * 50)

    if args.ply:
        if not Path(args.ply).exists():
            print(f"[Error] 文件不存在: {args.ply}")
            sys.exit(1)
        visualize_single_ply(args.ply, show_pca=not args.no_pca)
    else:
        if not Path(args.dir).exists():
            print(f"[Error] 目录不存在: {args.dir}")
            sys.exit(1)
        visualize_directory(args.dir, show_pca=not args.no_pca)


if __name__ == "__main__":
    main()
