"""CV 几何 angle 计算 + 可视化:用深度点云 PCA 算车主轴与参照线夹角。
用法: python cv_geometry_angle.py <image_path> [--out OUT]
"""
import sys, os, argparse, numpy as np
sys.path.insert(0, "/root/otto/XiaoanNew")
from modules.config.settings import settings
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.cv.depth_assist import DepthAssist
from modules.cv.image_utils import combine_masks
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mask_to_points(mask, depth, fx, fy, cx, cy):
    """mask + 深度 → 3D 点云(近似内参,相对尺度)"""
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return None
    ds = depth[ys, xs]
    X = (xs - cx) * ds / fx
    Y = (ys - cy) * ds / fy
    Z = ds
    return np.column_stack([X, Y, Z])

def pca_main_axis(points):
    """PCA 主轴方向(最大特征值对应的特征向量)"""
    if points is None or len(points) < 5:
        return None
    center = points.mean(axis=0)
    centered = points - center
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return center, Vt[0]  # 主轴方向

def project_to_ground(points, ground_normal=None):
    """投影到地面平面(XZ 平面,去掉 Y/高度)"""
    if ground_normal is None:
        # 简化:地面=XZ平面,去掉Y
        return points[:, [0, 2]]  # (X, Z)
    # TODO: 拟合地面平面投影
    return points[:, [0, 2]]

def angle_between(v1, v2):
    """两个 2D 向量的夹角(度)"""
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_a = np.clip(cos_a, -1, 1)
    return np.degrees(np.arccos(cos_a))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="/tmp/angle_vis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 加载模型
    seg = load_yolov8_seg(settings.YOLO_WEIGHTS, device=settings.INFERENCE_DEVICE)
    da = DepthAssist(device=settings.INFERENCE_DEVICE)

    img_bytes = open(args.image, "rb").read()
    pil = Image.open(args.image).convert("RGB")
    W, H = pil.size

    # YOLO
    result = seg.predict(img_bytes, visual=False, retina_masks=True, max_input_size=1280)
    objects = result["objects"]
    raw = result["image_raw"]

    # 深度
    depth = da.estimate(pil.resize((W, H)))

    # 近似内参(相对尺度)
    fx = fy = float(W)
    cx, cy = W / 2.0, H / 2.0

    # 提取各 mask
    bike_mask = None
    for obj in objects:
        if obj["label"] == "Electric bike":
            # 取最大/最居中的车
            if bike_mask is None or obj.get("mask") is not None and obj["mask"].sum() > bike_mask.sum():
                bike_mask = obj.get("mask")
    parking_mask = combine_masks(objects, "parking lane")
    curb_mask = combine_masks(objects, "Curb")
    tactile_mask = combine_masks(objects, "Tactile paving")

    print(f"车mask: {bike_mask.sum() if bike_mask is not None else 0}")
    print(f"停车线mask: {parking_mask.sum() if parking_mask is not None else 0}")
    print(f"路缘mask: {curb_mask.sum() if curb_mask is not None else 0}")

    if bike_mask is None:
        print("无车检测,退出")
        return

    # 车 点云
    bike_pts = mask_to_points(bike_mask, depth, fx, fy, cx, cy)
    bike_center, bike_axis = pca_main_axis(bike_pts) if bike_pts is not None else (None, None)

    # 参照线 点云(优先停车线,无则路缘)
    ref_mask = parking_mask if parking_mask is not None and parking_mask.sum() > 50 else curb_mask
    ref_name = "parking lane" if parking_mask is not None and parking_mask.sum() > 50 else ("curb" if curb_mask is not None else None)
    ref_pts = mask_to_points(ref_mask, depth, fx, fy, cx, cy) if ref_mask is not None else None
    ref_center, ref_axis = pca_main_axis(ref_pts) if ref_pts is not None else (None, None)

    # 投影到地面(XZ平面)算夹角
    if bike_axis is not None and ref_axis is not None:
        bike_ground = np.array([bike_axis[0], bike_axis[2]])  # X, Z
        ref_ground = np.array([ref_axis[0], ref_axis[2]])
        ang = angle_between(bike_ground, ref_ground)
        # 角度可能>90,取补角(车与线平行或垂直都算合规)
        if ang > 90: ang = 180 - ang  # 锐角
            ang = 180 - ang
        print(f"\n=== angle 计算 ===")
        print(f"车主轴方向: {bike_axis}")
        print(f"参照线({ref_name})主轴: {ref_axis}")
        print(f"地面投影夹角: {ang:.1f}°")
        print(f"判定: {'[合规]' if ang <= 30 else '[不合规-斜停]'}")

    # 可视化
    fig = plt.figure(figsize=(15, 5))
    # 原图+mask
    ax1 = fig.add_subplot(131)
    vis = np.array(pil.resize((W, H)))
    if bike_mask is not None:
        vis[bike_mask > 0] = vis[bike_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    if parking_mask is not None:
        vis[parking_mask > 0] = vis[parking_mask > 0] * 0.5 + np.array([255, 255, 0]) * 0.5
    if curb_mask is not None:
        vis[curb_mask > 0] = vis[curb_mask > 0] * 0.5 + np.array([0, 165, 255]) * 0.5
    if tactile_mask is not None:
        vis[tactile_mask > 0] = vis[tactile_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    ax1.imshow(vis.astype(np.uint8))
    ax1.set_title("Original + Masks (车绿/线黄/路缘橙/盲道红)")
    ax1.axis("off")

    # 3D 点云
    ax2 = fig.add_subplot(132, projection="3d")
    if bike_pts is not None:
        ax2.scatter(bike_pts[::5, 0], bike_pts[::5, 1], bike_pts[::5, 2], c="green", s=1, label="bike")
    if ref_pts is not None:
        ax2.scatter(ref_pts[::5, 0], ref_pts[::5, 1], ref_pts[::5, 2], c="orange", s=1, label=ref_name or "ref")
    if bike_center is not None and bike_axis is not None:
        ax2.quiver(*bike_center, *[a*0.5 for a in bike_axis], color="lime", linewidth=2)
    if ref_center is not None and ref_axis is not None:
        ax2.quiver(*ref_center, *[a*0.5 for a in ref_axis], color="red", linewidth=2)
    ax2.set_title("3D Point Cloud + PCA Axes")
    ax2.legend()

    # 地面俯视图(XZ)
    ax3 = fig.add_subplot(133)
    if bike_pts is not None:
        bg = project_to_ground(bike_pts)
        ax3.scatter(bg[::5, 0], bg[::5, 1], c="green", s=1, label="bike")
    if ref_pts is not None:
        rg = project_to_ground(ref_pts)
        ax3.scatter(rg[::5, 0], rg[rg[::5].shape[0]//2 if len(rg)>10 else 0, 1] if False else rg[::5, 1], c="orange", s=1, label=ref_name or "ref")
    if bike_axis is not None:
        bc = project_to_ground(np.array([bike_center, bike_center + bike_axis*0.5]))
        ax3.plot(bc[:, 0], bc[:, 1], "lime", linewidth=2)
    if ref_axis is not None:
        rc = project_to_ground(np.array([ref_center, ref_center + ref_axis*0.5]))
        ax3.plot(rc[:, 0], rc[:, 1], "red", linewidth=2)
    ax3.set_title(f"Ground View (XZ) angle={ang:.1f}°" if bike_axis is not None and ref_axis is not None else "Ground View")
    ax3.legend()
    ax3.set_aspect("equal")

    plt.tight_layout()
    out_path = os.path.join(args.out, "angle_vis.png")
    plt.savefig(out_path, dpi=100)
    print(f"\n可视化已保存: {out_path}")

if __name__ == "__main__":
    main()

# 追加:保存点云数据供本地可视化
def save_pointcloud_data():
    import numpy as np
    data = {"bike_pts": bike_pts, "ref_pts": ref_pts, "bike_axis": bike_axis, "ref_axis": ref_axis, "bike_center": bike_center, "ref_center": ref_center, "angle": ang if 'ang' in dir() else None, "ref_name": ref_name}
    np.save("/tmp/angle_pc_data.npy", data, allow_pickle=True)
    print("点云数据已保存 /tmp/angle_pc_data.npy")
