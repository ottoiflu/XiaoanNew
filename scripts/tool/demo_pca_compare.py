#!/usr/bin/env python3
"""单图 YOLO + 2D PCA + 3D PCA 对比可视化，PPT 配图用。
说明正对视角下 2D 掩模 PCA 够用，无需 3D 点云 PCA。
"""
import os, sys
sys.path.insert(0, "/root/XiaoanNew")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
# 中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from modules.cv.yolov8_inference import load_yolov8_seg
from modules.cv.angle_inference import pca_2d_image, angle_between, get_centermost_bike
from modules.cv.depth_assist import DepthAssist

IMG = "/root/sample_pca_demo.png"
OUT = "/root/pca_demo_output"
os.makedirs(OUT, exist_ok=True)

COLORS = {
    "Electric bike": (0, 255, 0),
    "Curb": (255, 0, 255),
    "parking lane": (255, 255, 0),
    "Tactile paving": (255, 165, 0),
    "Green belt": (0, 180, 255),
}

def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except Exception:
        return ImageFont.load_default()

print("[1/5] 加载 YOLO + DepthAssist")
seg = load_yolov8_seg("/root/XiaoanNew/assets/weights/new_best.pt", device="cuda:0")
da = DepthAssist(device="cuda:0")

print("[2/5] YOLO 分割")
img_pil = Image.open(IMG).convert("RGB")
img_arr = np.array(img_pil)
H, W = img_arr.shape[:2]
res = seg.predict(IMG, conf=0.35, iou=0.7, imgsz=1024, visual=False)
objects = res["objects"]
print(f"  检出 {len(objects)} 个目标:")
for o in objects:
    print(f"    {o['label']} conf={o['confidence']:.2f}")

# 主车 + 参照线
main_bike = get_centermost_bike(objects, W, H)
if main_bike is None:
    print("ERROR: 未检出电动车"); sys.exit(1)
bike_mask = main_bike["mask"]
ref_label = None
ref_mask = None
for lbl in ["parking lane", "Curb", "Tactile paving"]:
    for o in objects:
        if o["label"] == lbl and o.get("mask") is not None:
            ref_label = lbl; ref_mask = o["mask"]; break
    if ref_mask is not None: break
if ref_mask is None:
    print("ERROR: 无标线/路缘参照"); sys.exit(1)
print(f"  参照: {ref_label}")

# 图1: 原图
img_pil.save(f"{OUT}/1_original.jpg", quality=95)
print("[3/5] 图1 原图 OK")

# 图2: YOLO 掩模叠加
overlay = img_pil.copy()
overlay_arr = np.array(overlay).astype(np.uint8).copy()
for o in objects:
    if o.get("mask") is None: continue
    m = o["mask"]
    c = COLORS.get(o["label"], (128,128,128))
    color_layer = np.zeros_like(overlay_arr)
    color_layer[m] = c
    overlay_arr = (overlay_arr * 0.6 + color_layer * 0.4).astype(np.uint8)
Image.fromarray(overlay_arr).save(f"{OUT}/2_yolo_mask.jpg", quality=95)
print("[4/5] 图2 YOLO掩模 OK")

# 图3: 2D PCA
bike_center, bike_dir = pca_2d_image(bike_mask)
ref_center, ref_dir = pca_2d_image(ref_mask)
angle_2d = angle_between(bike_dir, ref_dir) if (bike_dir is not None and ref_dir is not None) else None
print(f"  2D PCA 夹角: {angle_2d:.2f}°" if angle_2d else "  2D PCA 失败")

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
ax.imshow(img_arr)
# mask 半透明
def mask_rgba(mask, color):
    r = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    r[mask] = [*color, 100]
    return r
ax.imshow(mask_rgba(bike_mask, (0,255,0)), alpha=0.4)
ax.imshow(mask_rgba(ref_mask, (255,255,0)), alpha=0.4)
# 主轴线
def draw_axis(ax, center, direction, color, length=400, label=""):
    cx, cy = center
    dx, dy = direction
    x1, y1 = cx - dx*length, cy - dy*length
    x2, y2 = cx + dx*length, cy + dy*length
    ax.plot([x1,x2],[y1,y2], color=color, linewidth=3.5, solid_capstyle='round')
    ax.scatter([cx],[cy], color=color, s=80, zorder=5, edgecolors='white', linewidths=1.5)
    if label:
        ax.text(x2, y2, label, color=color, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
if bike_center is not None and bike_dir is not None:
    draw_axis(ax, bike_center, bike_dir, '#00FF00', 350, 'bike 2D PCA')
if ref_center is not None and ref_dir is not None:
    draw_axis(ax, ref_center, ref_dir, '#FFFF00', 350, f'{ref_label} 2D PCA')
ax.set_title(f"2D 掩模 PCA  |  车辆 vs {ref_label}  |  夹角 = {angle_2d:.1f}°" if angle_2d else "2D PCA",
             fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/3_pca_2d.jpg", dpi=150, bbox_inches='tight')
plt.close()
print("[5/5] 图3 2D PCA OK")

# 图4: 3D 点云 + 3D PCA
print("[6/5] 3D 点云化 + 3D PCA")
depth = da.estimate(img_pil)

def cloud_3d(mask, depth):
    ys, xs = np.where(mask)
    zs = depth[ys, xs]
    return np.column_stack([xs, ys, zs]).astype(float)

def pca_3d(points):
    center = points.mean(axis=0)
    cov = np.cov((points - center).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 最大特征值对应的主轴
    main_axis = eigvecs[:, -1]
    return center, main_axis, eigvals

bike_pts = cloud_3d(bike_mask, depth)
ref_pts = cloud_3d(ref_mask, depth)
bike_c3, bike_axis3, _ = pca_3d(bike_pts)
ref_c3, ref_axis3, _ = pca_3d(ref_pts)
# 3D 夹角（取绝对值，主轴方向无方向性）
cos3 = abs(np.dot(bike_axis3, ref_axis3) / (np.linalg.norm(bike_axis3)*np.linalg.norm(ref_axis3)+1e-8))
angle_3d = np.degrees(np.arccos(np.clip(cos3, -1, 1)))
print(f"  3D PCA 夹角: {angle_3d:.2f}°")

fig = plt.figure(figsize=(10, 8), dpi=150)
ax3 = fig.add_subplot(111, projection='3d')
# 下采样点云（太多点慢）
step = max(1, len(bike_pts)//2000)
ax3.scatter(bike_pts[::step,0], bike_pts[::step,1], bike_pts[::step,2],
            c='green', s=2, alpha=0.4, label='bike 点云')
step2 = max(1, len(ref_pts)//2000)
ax3.scatter(ref_pts[::step2,0], ref_pts[::step2,1], ref_pts[::step2,2],
            c='gold', s=2, alpha=0.4, label=f'{ref_label} 点云')
# 主轴箭头
def draw_3d_axis(ax, center, direction, color, length=300, name=""):
    cx,cy,cz = center
    dx,dy,dz = direction/ (np.linalg.norm(direction)+1e-8) * length
    ax.quiver(cx,cy,cz, dx,dy,dz, color=color, linewidth=3, arrow_length_ratio=0.15)
draw_3d_axis(ax3, bike_c3, bike_axis3, 'lime', 300, 'bike 3D PCA')
draw_3d_axis(ax3, ref_c3, ref_axis3, 'yellow', 300, f'{ref_label} 3D PCA')
ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Depth')
ax3.set_title(f"3D 点云 PCA  |  夹角 = {angle_3d:.1f}°", fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
# 调视角
ax3.view_init(elev=25, azim=-60)
plt.tight_layout()
plt.savefig(f"{OUT}/4_pca_3d.jpg", dpi=150, bbox_inches='tight')
plt.close()
print("  图4 3D PCA OK")

# 图5: 对比拼接
fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=150)
img2d = Image.open(f"{OUT}/3_pca_2d.jpg")
img3d = Image.open(f"{OUT}/4_pca_3d.jpg")
axes[0].imshow(img2d); axes[0].axis('off')
axes[1].imshow(img3d); axes[1].axis('off')
delta = abs(angle_2d - angle_3d) if angle_2d and angle_3d else None
suptitle = f"正对视角下 2D 掩模 PCA ≈ 3D 点云 PCA    |    2D: {angle_2d:.1f}°   3D: {angle_3d:.1f}°   Δ={delta:.1f}°"
fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
fig.text(0.5, 0.04, "正对拍摄 → 投影共线 → 2D 掩模 PCA 主轴夹角直接近似真实夹角，无需点云化 / 深度估计 / 相机内参",
         ha='center', fontsize=11, style='italic', color='#444')
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig(f"{OUT}/5_compare.jpg", dpi=150, bbox_inches='tight')
plt.close()
print("  图5 对比图 OK")

print(f"\n=== 完成 ===")
print(f"2D夹角: {angle_2d:.2f}°  3D夹角: {angle_3d:.2f}°  差值: {delta:.2f}°")
print(f"输出: {OUT}/")
for f in sorted(os.listdir(OUT)):
    sz = os.path.getsize(f"{OUT}/{f}")
    print(f"  {f}  {sz//1024}KB")
