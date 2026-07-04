"""提取带原图颜色的点云数据。2D图像PCA(不受深度拉伸影响)。
用法: python extract_pc_colored.py <image_path> [--out OUT]
"""
import sys, os, base64, numpy as np
sys.path.insert(0, "/root/otto/XiaoanNew")
from modules.config.settings import settings
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.cv.depth_assist import DepthAssist
from modules.cv.image_utils import combine_masks
from PIL import Image

def mask_to_pts(mask, depth, fx, fy, cx, cy):
    """mask → 点云(X,Y,Z,u,v)"""
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return None
    ds = depth[ys, xs]
    X = (xs - cx) * ds / fx
    Y = (ys - cy) * ds / fy
    Z = ds
    return np.column_stack([X, Y, Z, xs, ys])

def pca_2d_image(pts):
    """在2D图像平面(u,v)做PCA——不受深度拉伸影响"""
    if pts is None or len(pts) < 5:
        return None, None
    uv = pts[:, [3, 4]].astype(float)  # u=列, v=行
    c = uv.mean(axis=0)
    U, S, Vt = np.linalg.svd(uv - c, full_matrices=False)
    return c, Vt[0]  # 第一主轴=图像里的长边方向

def angle_2d(v1, v2):
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def get_centermost_bike(objects, W, H):
    img_cx, img_cy = W / 2.0, H / 2.0
    best, best_dist = None, float("inf")
    for obj in objects:
        if obj["label"] != "Electric bike":
            continue
        bx1, by1, bx2, by2 = obj["bbox"]
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        dist = (bcx - img_cx) ** 2 + (bcy - img_cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best = obj
    return best

def main():
    img_path = sys.argv[1]
    out_dir = "/tmp/pc_colored"
    if "--out" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(out_dir, exist_ok=True)

    seg = load_yolov8_seg(settings.YOLO_WEIGHTS, device=settings.INFERENCE_DEVICE)
    da = DepthAssist(device=settings.INFERENCE_DEVICE)

    img_bytes = open(img_path, "rb").read()
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    result = seg.predict(img_bytes, visual=False, retina_masks=True, max_input_size=None)
    objects = result["objects"]
    depth = da.estimate(pil)
    fx = fy = float(W); cx, cy = W / 2.0, H / 2.0

    main_bike = get_centermost_bike(objects, W, H)
    points = {}
    if main_bike and main_bike.get("mask") is not None:
        pts = mask_to_pts(main_bike["mask"], depth, fx, fy, cx, cy)
        if pts is not None:
            points["Electric bike"] = pts
            print(f"Electric bike(最居中): {len(pts)}点")

    for label in ["parking lane", "Curb", "Tactile paving"]:
        mask = combine_masks(objects, label)
        if mask is not None and mask.sum() > 50:
            pts = mask_to_pts(mask, depth, fx, fy, cx, cy)
            if pts is not None:
                points[label] = pts
                print(f"{label}: {len(pts)}点")

    # 2D图像PCA(关键:用u,v不用X,Z)
    axes = {}
    for label, pts in points.items():
        c, a = pca_2d_image(pts)
        if c is not None:
            axes[label] = {"center": c, "axis": a}
            print(f"  {label} 2D图像PCA主轴: {a}")

    # 角度
    ref_name = None
    for name in ["parking lane", "Curb"]:
        if name in points:
            ref_name = name
            break

    angle = None
    if "Electric bike" in axes and ref_name in axes:
        ba = axes["Electric bike"]["axis"]
        ra = axes[ref_name]["axis"]
        angle = angle_2d(ba, ra)
        if angle > 90:
            angle = 180 - angle
        print(f"\n锐角={angle:.1f}° 参照={ref_name}")

    buf = os.path.join(out_dir, "_tmp.jpg")
    pil.save(buf, "JPEG", quality=80)
    img_b64 = base64.b64encode(open(buf, "rb").read()).decode()
    os.remove(buf)

    name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{name}_pc.npz")
    np.savez_compressed(out_path, points=points, axes=axes, angle=angle,
        ref_name=ref_name or "", image_b64=img_b64, image_size=[W, H])
    print(f"\n已存: {out_path}")

if __name__ == "__main__":
    main()
