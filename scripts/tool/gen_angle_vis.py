#!/usr/bin/env python3
"""按 v3 angle_pipeline 流程生成 YOLO 掩膜 + PCA 可视化图。
复用 v3 的 PCA 函数（pca_2d_image/get_centermost_bike），对齐测试流程：
- YOLO 分割掩膜半透明叠加（车绿/标线黄/路缘橙/盲道红）
- PCA 主轴双向线段（白=车 / 黄=标线 / 橙=路缘）
- 图上标注 CSV 中间值（cv_angle/judgment/disambiguation/vlm_angle/final_angle）
"""
import sys, os, json, csv
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/root/otto/XiaoanNew")
from modules.config.settings import settings
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.cv.image_utils import combine_masks

GT = "/root/otto/XiaoanNew/data/benchmark/benchmark_v4/fourdim_gt_v4.json"
DATA_ROOT = "/root/otto/XiaoanNew/data"
CSV_PATH = "/root/otto/XiaoanNew/outputs/benchmark_output/v4/angle_pipeline_results.csv"
OUT_DIR = "/root/otto/XiaoanNew/outputs/benchmark_output/v4/angle_vis_imgs"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    "Electric bike": (0, 255, 0),
    "Curb": (255, 128, 0),
    "parking lane": (255, 255, 0),
    "Tactile paving": (255, 0, 0),
}

# ---- 复用 angle_pipeline 的 PCA 函数（保持与 v3 流程一致）----
def pca_2d_image(mask):
    """对二值 mask 做 PCA，返回 (中心点, 主轴方向单位向量)"""
    if mask is None: return None, None
    ys, xs = np.where(mask > 0)
    if len(xs) < 5: return None, None
    uv = np.column_stack([xs, ys]).astype(float)
    c = uv.mean(axis=0)
    _, _, Vt = np.linalg.svd(uv - c, full_matrices=False)
    return c, Vt[0]

def get_centermost_bike(objects, W, H):
    """取距画面中心最近的 Electric bike"""
    cx, cy = W / 2.0, H / 2.0
    best, bd = None, float("inf")
    for obj in objects:
        if obj["label"] != "Electric bike": continue
        bx1, by1, bx2, by2 = obj["bbox"]
        d = ((bx1 + bx2) / 2 - cx) ** 2 + ((by1 + by2) / 2 - cy) ** 2
        if d < bd: bd, best = d, obj
    return best

def find_img(src):
    p = os.path.join(DATA_ROOT, src)
    if os.path.exists(p): return p
    for ext in (".JPEG", ".jpeg", ".png", ".jpg"):
        alt = os.path.splitext(p)[0] + ext
        if os.path.exists(alt): return alt
    return None

def draw_axis(draw, center, direction, color, length=140, width=4):
    """画 PCA 主轴双向线段"""
    c = np.array(center, dtype=float)
    d = np.array(direction, dtype=float)
    n = np.linalg.norm(d)
    if n < 1e-8: return
    d = d / n
    end = c + d * length
    start = c - d * length
    draw.line([start[0], start[1], end[0], end[1]], fill=color, width=width)

def process_one(entry, csv_row, seg, idx):
    img_path = find_img(entry.get("src", ""))
    if not img_path: return False
    img_bytes = open(img_path, "rb").read()
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    res = seg.predict(img_bytes, visual=False, retina_masks=True)
    objects = res["objects"]

    bike_obj = get_centermost_bike(objects, W, H)
    bike_mask = bike_obj.get("mask") if bike_obj else None
    line_objects = [o for o in objects if o["label"] == "parking lane"]
    curb_mask = combine_masks(objects, "Curb")
    if curb_mask is not None and curb_mask.sum() < 50:
        curb_mask = None

    # PCA（对齐 v3 流程）
    bike_c, bike_d = pca_2d_image(bike_mask) if bike_mask is not None else (None, None)
    line_pcas = []
    for lo in line_objects:
        lm = lo.get("mask")
        if lm is not None:
            c, d = pca_2d_image(lm)
            if c is not None: line_pcas.append((c, d))
    curb_c, curb_d = (None, None)
    if curb_mask is not None:
        curb_c, curb_d = pca_2d_image(curb_mask)

    # 画掩膜半透明
    arr = np.array(pil).astype(np.float32)
    for o in objects:
        label = o["label"]; color = COLORS.get(label, (128, 128, 128))
        mask = o.get("mask")
        if isinstance(mask, np.ndarray) and mask.sum() > 0:
            m = mask.astype(bool)
            if m.shape == arr.shape[:2]:
                for c in range(3):
                    arr[:, :, c][m] = arr[:, :, c][m] * 0.5 + color[c] * 0.5
    overlay = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(overlay)

    # 画 PCA 主轴
    if bike_c is not None: draw_axis(draw, bike_c, bike_d, (255, 255, 255), 160, 4)  # 白=车
    for c, d in line_pcas: draw_axis(draw, c, d, (255, 255, 0), 130, 3)  # 黄=标线
    if curb_c is not None: draw_axis(draw, curb_c, curb_d, (255, 128, 0), 130, 3)  # 橙=路缘

    # bbox + label
    for o in objects:
        label = o["label"]; color = COLORS.get(label, (128, 128, 128))
        bx1, by1, bx2, by2 = o["bbox"]
        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
        draw.text((bx1, max(0, by1 - 12)), label, fill=color)

    # 标注 CSV 中间值（左上角）
    lines = [
        f"id={entry['id'][:32]} gt={csv_row.get('gt', '')}",
        f"lines={csv_row.get('cv_n_line_types', '')} cv_ang1={csv_row.get('cv_angle_1', '')} j1={csv_row.get('cv_judgment_1', '')}",
        f"cv_ang2={csv_row.get('cv_angle_2', '')} j2={csv_row.get('cv_judgment_2', '')}",
        f"disamb={csv_row.get('cv_disambiguation_summary', '')}",
        f"vlm_ang={csv_row.get('vlm_angle', '')} fin_ang={csv_row.get('final_angle', '')}",
    ]
    y = 8
    for ln in lines:
        draw.text((8, y), ln, fill=(255, 255, 255))
        y += 14

    out = os.path.join(OUT_DIR, f"{idx:04d}.jpg")
    overlay.save(out, quality=80)
    return True

def main():
    gt = json.load(open(GT))
    csv_rows = {r["id"]: r for r in csv.DictReader(open(CSV_PATH, encoding="utf-8"))}
    print(f"[Load] YOLOv8-Seg ...")
    seg = load_yolov8_seg(settings.YOLO_WEIGHTS, device=settings.INFERENCE_DEVICE)
    print(f"[Run] {len(gt)} imgs ...")
    mapping = {}
    ok = 0
    for i, entry in enumerate(gt):
        eid = entry["id"]
        csv_row = csv_rows.get(eid, {})
        if process_one(entry, csv_row, seg, i):
            mapping[eid] = f"{i:04d}.jpg"
            ok += 1
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(gt)}", flush=True)
    json.dump(mapping, open(os.path.join(OUT_DIR, "id2img.json"), "w"))
    print(f"[Done] {ok}/{len(gt)} imgs -> {OUT_DIR}")

if __name__ == "__main__":
    main()
