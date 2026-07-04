"""可视化所有检出车+参照线的样本: YOLO mask + 2D PCA箭头 + 角度值, 生成HTML相册。"""
import sys, os, csv, json, numpy as np, base64, io
sys.path.insert(0, "/root/otto/XiaoanNew")
from modules.config.settings import settings
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.cv.image_utils import combine_masks, draw_wireframe_visual
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

seg = load_yolov8_seg(settings.YOLO_WEIGHTS, device=settings.INFERENCE_DEVICE)
gt = {x["id"]: x for x in json.load(open("/root/otto/XiaoanNew/data/benchmark/benchmark_v4/fourdim_gt_v4.json"))}

rows = list(csv.DictReader(open("/root/otto/XiaoanNew/outputs/benchmark_output/v4/baseline_qwen36/results_cv_2d_angle.csv", encoding="utf-8-sig", errors="replace")))
orig = list(csv.DictReader(open("/root/otto/XiaoanNew/outputs/benchmark_output/v4/baseline_qwen36/results.csv", encoding="utf-8-sig", errors="replace")))
orig_angle = {r.get("id",""): r.get("angle_status","") for r in orig}

def pca_2d_image(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 5: return None, None
    uv = np.column_stack([xs, ys]).astype(float)
    c = uv.mean(axis=0)
    U, S, Vt = np.linalg.svd(uv - c, full_matrices=False)
    return c, Vt[0]

def angle_2d(v1, v2):
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def get_centermost_bike(objects, W, H):
    cx, cy = W/2.0, H/2.0
    best, bd = None, float("inf")
    for obj in objects:
        if obj["label"] != "Electric bike": continue
        bx1,by1,bx2,by2 = obj["bbox"]
        d = ((bx1+bx2)/2-cx)**2 + ((by1+by2)/2-cy)**2
        if d < bd: bd, best = d, obj
    return best

def draw_arrow(img, start, end, color, width=3):
    draw = ImageDraw.Draw(img)
    draw.line([tuple(start), tuple(end)], fill=color, width=width)
    # 箭头
    import math
    angle = math.atan2(end[1]-start[1], end[0]-start[0])
    for da in [2.5, -2.5]:
        ax = end[0] - 15 * math.cos(angle - da)
        ay = end[1] - 15 * math.sin(angle - da)
        draw.line([tuple(end), (ax, ay)], fill=color, width=width)

out_dir = "/tmp/pca_viz"
os.makedirs(out_dir, exist_ok=True)

gallery = []
n_ok = 0; n_fail = 0

for i, r in enumerate(rows):
    rid = r.get("id","")
    g = gt.get(rid, {})
    src = g.get("src","")
    if not src:
        n_fail += 1; continue
    img_path = os.path.join("/root/otto/XiaoanNew/data", src)
    if not os.path.exists(img_path):
        for ext in (".JPEG",".jpeg",".png"):
            alt = os.path.splitext(img_path)[0]+ext
            if os.path.exists(alt): img_path = alt; break
    if not os.path.exists(img_path):
        n_fail += 1; continue

    try:
        img_bytes = open(img_path,"rb").read()
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        result = seg.predict(img_bytes, visual=False, retina_masks=True, max_input_size=None)
        objects = result["objects"]

        bike_obj = get_centermost_bike(objects, W, H)
        if bike_obj is None or bike_obj.get("mask") is None:
            n_fail += 1; continue
        bike_mask = bike_obj["mask"]

        parking_mask = combine_masks(objects,"parking lane")
        curb_mask = combine_masks(objects,"Curb")
        ref_mask = parking_mask if parking_mask is not None and parking_mask.sum()>50 else curb_mask
        ref_name = "parking" if parking_mask is not None and parking_mask.sum()>50 else "curb"
        if ref_mask is None or ref_mask.sum()<50:
            n_fail += 1; continue

        # PCA
        bc, ba = pca_2d_image(bike_mask)
        rc, ra = pca_2d_image(ref_mask)
        if ba is None or ra is None:
            n_fail += 1; continue
        ang = angle_2d(ba, ra)
        if ang > 90: ang = 180 - ang
        cv_angle = "[合规]" if 60<=ang<=90 else "[不合规-斜停]"

        # 缩放图(太大画不动)
        scale = min(1.0, 800.0 / max(W, H))
        small = pil.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
        vis = small.copy()

        # 画mask半透明
        overlay = Image.new("RGBA", small.size, (0,0,0,0))
        draw_ov = ImageDraw.Draw(overlay)
        # 车mask(绿)
        ys, xs = np.where(bike_mask > 0)
        for y, x in zip(ys[::5], xs[::5]):
            draw_ov.point((int(x*scale), int(y*scale)), fill=(0,255,0,100))
        # 参照线mask(黄/橙)
        ys2, xs2 = np.where(ref_mask > 0)
        color_ref = (255,255,0,100) if "parking" in ref_name else (255,165,0,100)
        for y, x in zip(ys2[::5], xs2[::5]):
            draw_ov.point((int(x*scale), int(y*scale)), fill=color_ref)
        vis = Image.alpha_composite(vis.convert("RGBA"), overlay).convert("RGB")

        # 画PCA箭头
        bc_s = bc * scale
        ba_end = (bc_s[0] + ba[0]*150, bc_s[1] + ba[1]*150)
        draw_arrow(vis, bc_s, ba_end, (0,255,0), 3)  # 车轴(绿)

        rc_s = rc * scale
        ra_end = (rc_s[0] + ra[0]*150, rc_s[1] + ra[1]*150)
        draw_arrow(vis, rc_s, ra_end, (255,0,0), 3)  # 线轴(红)

        # 文字
        draw = ImageDraw.Draw(vis)
        gt_angle = g.get("angle","?")
        vlm_angle = orig_angle.get(rid,"?")
        txt = f"GT:{gt_angle} VLM:{vlm_angle} CV:{cv_angle} ang={ang:.1f}° ref={ref_name}"
        draw.rectangle([0, 0, small.size[0], 25], fill=(0,0,0))
        draw.text((5, 5), txt, fill=(255,255,255))

        # 保存
        out_path = os.path.join(out_dir, f"{n_ok:04d}.jpg")
        vis.save(out_path, "JPEG", quality=85)

        gallery.append({
            "img": f"{n_ok:04d}.jpg",
            "id": rid[:30],
            "gt": gt_angle,
            "vlm": vlm_angle,
            "cv": cv_angle,
            "angle": f"{ang:.1f}",
            "ref": ref_name,
            "correct": cv_angle == gt_angle if gt_angle != "?" else None,
        })
        n_ok += 1
    except Exception as e:
        n_fail += 1

    if (i+1) % 200 == 0:
        print(f"  {i+1}/{len(rows)} 可视化{n_ok} 失败{n_fail}")

# HTML相册
html = "<html><head><meta charset='utf-8'><style>body{background:#111;color:#ddd;font-family:sans-serif}img{max-width:400px;display:block;margin:5px}div{display:inline-block;margin:10px;vertical-align:top}.ok{color:#7ee29a}.bad{color:#ff8fa3}</style></head><body><h2>2D PCA Gallery (%d张)</h2>\n" % n_ok
for g in gallery:
    cls = "ok" if g["correct"] else "bad" if g["correct"] is not None else ""
    html += f"<div><img src='{g['img']}'><span class='{cls}'>GT:{g['gt']} VLM:{g['vlm']} CV:{g['cv']} ang={g['angle']}° ref={g['ref']}</span><br><small>{g['id']}</small></div>\n"
html += "</body></html>"

with open(os.path.join(out_dir, "gallery.html"), "w") as f:
    f.write(html)

print(f"\n可视化: {n_ok}, 失败: {n_fail}")
correct = sum(1 for g in gallery if g["correct"])
wrong = sum(1 for g in gallery if g["correct"] is False)
print(f"覆盖样本中 CV angle正确: {correct}/{n_ok} = {correct/n_ok:.3f}")
print(f"HTML: {os.path.join(out_dir, 'gallery.html')}")
