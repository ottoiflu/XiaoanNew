"""
三类掩膜点云并集合并工具

将 depth_demo 输出目录中电动车、curb、车道线三个 PLY 点云
合并为一个并集点云，用于论文可视化展示。

支持两种着色模式：
  --color original  保留原图 RGB 颜色（默认）
  --color category  每个类别赋予独立固定颜色

使用方式：
    uv run python scripts/tool/merge_union_masks.py \
        --dir outputs/depth_demo/01e93aa7-5ec3-4ac3-9f52-7869f8b4462d

    uv run python scripts/tool/merge_union_masks.py \
        --dir outputs/depth_demo/01e93aa7-5ec3-4ac3-9f52-7869f8b4462d \
        --color category
"""

import argparse
import sys
from pathlib import Path

import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MASK_CLASSES = [
    {
        "file_prefix": "04_electric_bike.ply",
        "label": "electric_bike",
        "category_color": [0.0, 1.0, 0.0],
    },
    {
        "file_prefix": "05_curb.ply",
        "label": "curb",
        "category_color": [1.0, 0.0, 1.0],
    },
    {
        "file_prefix": "06_parking_lane.ply",
        "label": "parking_lane",
        "category_color": [1.0, 1.0, 0.0],
    },
]

OUTPUT_FILENAME = "08_union_masks.ply"


def load_and_optionally_recolor(ply_path, category_color):
    """加载 PLY 点云，可选按类别颜色重新着色"""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if len(pcd.points) == 0:
        return pcd
    if category_color is not None:
        pcd.paint_uniform_color(category_color)
    return pcd


def merge_pointclouds(pcds):
    """将多个点云合并为一个，并执行统计离群点过滤"""
    merged = o3d.geometry.PointCloud()
    for pcd in pcds:
        if len(pcd.points) > 0:
            merged += pcd

    if len(merged.points) > 100:
        merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(merged.points) > 0:
        merged.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    return merged


def run(output_dir, color_mode):
    """主流程：加载三个掩膜点云并合并导出"""
    dir_path = Path(output_dir).resolve()
    if not dir_path.is_dir():
        print(f"[Error] 目录不存在: {dir_path}")
        sys.exit(1)

    use_category_color = color_mode == "category"

    print("=" * 60)
    print("  三类掩膜点云并集合并")
    print(f"  输入目录: {dir_path}")
    print(f"  着色模式: {'类别固定色' if use_category_color else '保留原图RGB'}")
    print("=" * 60)

    pcds = []
    total_before = 0

    for cls in MASK_CLASSES:
        ply_path = dir_path / cls["file_prefix"]
        if not ply_path.exists():
            print(f"  [跳过] 文件不存在: {ply_path.name}")
            continue

        cat_color = cls["category_color"] if use_category_color else None
        pcd = load_and_optionally_recolor(ply_path, cat_color)
        n = len(pcd.points)
        total_before += n
        print(f"  [{cls['label']:20s}]  {ply_path.name}  ->  {n:,} 点")
        pcds.append(pcd)

    if not pcds:
        print("[Error] 未找到任何有效点云文件，退出")
        sys.exit(1)

    print(f"\n  合并前总点数: {total_before:,}")
    merged = merge_pointclouds(pcds)
    n_merged = len(merged.points)
    print(f"  离群点过滤后:  {n_merged:,} 点")

    out_path = dir_path / OUTPUT_FILENAME
    o3d.io.write_point_cloud(str(out_path), merged)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n  已保存: {out_path}  ({size_kb:.1f} KB)")
    print("=" * 60)
    return out_path


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="将电动车/curb/车道线三个掩膜点云合并为并集点云（论文展示用）",
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="depth_demo 输出目录（含各类别 PLY 文件）",
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["original", "category"],
        default="original",
        help="着色模式：original=保留原图RGB（默认），category=类别固定色",
    )
    args = parser.parse_args()
    run(args.dir, args.color)


if __name__ == "__main__":
    main()
