#!/usr/bin/env python3
"""IoU 实验：主车 mask 与盲道 mask 的 2D IoU 分布 + 各阈值 precision"""
import json, os, sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/otto/XiaoanNew")
from modules.config.settings import settings
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.cv.image_utils import combine_masks, calculate_iou_and_overlap

GT_PATH = "/root/otto/XiaoanNew/data/benchmark/benchmark_v4/fourdim_gt_v4.json"
DATA_ROOT = "/root/otto/XiaoanNew/data"
OUT_PATH = "/tmp/iou_result.txt"

gt = json.load(open(GT_PATH))
N = len(gt)
print(f"[Load] YOLOv8-Seg (N={N}) ...")
seg = load_yolov8_seg(settings.YOLO_WEIGHTS, device=settings.INFERENCE_DEVICE)


def find_img(src: str) -> str | None:
    p = os.path.join(DATA_ROOT, src)
    if os.path.exists(p):
        return p
    base, _ = os.path.splitext(p)
    for ext in (".jpg", ".jpeg", ".png", ".JPEG"):
        alt = base + ext
        if os.path.exists(alt):
            return alt
    return None


def get_centermost_bike(objects: list[dict], W: float, H: float) -> dict | None:
    cx, cy = W / 2.0, H / 2.0
    best, bd = None, float("inf")
    for obj in objects:
        if obj["label"] != "Electric bike":
            continue
        bx1, by1, bx2, by2 = obj["bbox"]
        d = ((bx1 + bx2) / 2 - cx) ** 2 + ((by1 + by2) / 2 - cy) ** 2
        if d < bd:
            bd, best = d, obj
    return best


# ── 遍历 ──
records = []
stats = Counter()

for i, entry in enumerate(gt):
    img_path = find_img(entry.get("src", ""))
    if not img_path:
        stats["no_img"] += 1
        records.append({"id": entry["id"], "gt": entry["gt"], "has_both": False})
        continue

    img_bytes = open(img_path, "rb").read()
    pil = Image.open(img_path)
    W, H = pil.size
    res = seg.predict(img_bytes, visual=False, retina_masks=True)
    objs = res["objects"]

    # 主车
    bike_obj = get_centermost_bike(objs, W, H)
    if bike_obj is None:
        stats["no_bike"] += 1
        records.append({"id": entry["id"], "gt": entry["gt"], "has_both": False})
        continue
    bike_mask = bike_obj.get("mask")
    if bike_mask is None or (isinstance(bike_mask, np.ndarray) and bike_mask.sum() < 1):
        stats["no_bike_mask"] += 1
        records.append({"id": entry["id"], "gt": entry["gt"], "has_both": False})
        continue

    # 盲道
    blind_mask = combine_masks(objs, "Tactile paving")
    if blind_mask is None:
        stats["no_blind"] += 1
        records.append({"id": entry["id"], "gt": entry["gt"], "has_both": False})
        continue

    stats["has_both"] += 1
    iou, overlap = calculate_iou_and_overlap(bike_mask, blind_mask)
    records.append({"id": entry["id"], "gt": entry["gt"], "has_both": True, "iou": round(iou, 6), "overlap": round(overlap, 6)})

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{N}", flush=True)

# ── 统计 ──
has_both_recs = [r for r in records if r["has_both"]]
iou_vals = [r["iou"] for r in has_both_recs]

# IoU 分布 — 互斥区间
bins = [
    ("=0", lambda v: v == 0.0),
    ("(0, 0.01)", lambda v: 0.0 < v < 0.01),
    ("[0.01, 0.05)", lambda v: 0.01 <= v < 0.05),
    ("[0.05, 0.1)", lambda v: 0.05 <= v < 0.10),
    ("[0.1, 0.2)", lambda v: 0.10 <= v < 0.20),
    ("[0.2, 0.5)", lambda v: 0.20 <= v < 0.50),
    ("[0.5, 1.0]", lambda v: 0.50 <= v <= 1.0),
]
bin_counts = {lb: sum(1 for v in iou_vals if pred(v)) for lb, pred in bins}
assert sum(bin_counts.values()) == len(iou_vals), f"{sum(bin_counts.values())} != {len(iou_vals)}"

# 各阈值
thresholds = [0.01, 0.05, 0.1, 0.2]
th_results = []
total_violations = sum(1 for r in records if r["gt"] == "no")
total_compliant = sum(1 for r in records if r["gt"] == "yes")

for th in thresholds:
    flagged = [r for r in has_both_recs if r["iou"] >= th]
    flagged_n = len(flagged)
    tp = sum(1 for r in flagged if r["gt"] == "no")
    fp = sum(1 for r in flagged if r["gt"] == "yes")
    prec = tp / flagged_n if flagged_n > 0 else 0.0
    recall_viol = tp / total_violations if total_violations > 0 else 0.0
    fp_rate = fp / total_compliant if total_compliant > 0 else 0.0
    th_results.append({
        "th": th, "flagged": flagged_n, "tp": tp, "fp": fp,
        "precision": prec, "recall_viol": recall_viol, "fp_rate": fp_rate
    })

# ── 写入结果 ──
lines = []
lines.append("=" * 65)
lines.append(" IoU 实验报告: 主车 mask vs 盲道 mask 2D IoU")
lines.append("=" * 65)
lines.append(f" 总样本数: {N}")
lines.append(f"   gt=yes(合规): {total_compliant}")
lines.append(f"   gt=no(违规):  {total_violations}")
lines.append("")

lines.append("--- 各状态计数 ---")
lines.append(f"  检出盲道(has_both): {stats['has_both']}")
for k in ("no_img", "no_bike", "no_bike_mask", "no_blind"):
    lines.append(f"  {k}: {stats.get(k, 0)}")
lines.append("")

lines.append(f"--- IoU 分布 (检出盲道的 {len(has_both_recs)} 张) ---")
for lb, pred in bins:
    c = bin_counts[lb]
    pct = c / len(has_both_recs) * 100 if has_both_recs else 0
    lines.append(f"  IoU {lb:12s}: {c:4d} 张 ({pct:.1f}%)")
lines.append(f"  合计             : {len(has_both_recs):4d} 张")
lines.append("")

lines.append("--- 各 IoU 阈值判违规 (has_both 子集) ---")
lines.append(f"{'阈值':>6s} | {'判违规':>7s} | {'TP':>5s} | {'FP':>5s} | {'Precision':>10s} | {'违规召回':>9s} | {'合规误报率':>10s}")
lines.append("-" * 66)
for r in th_results:
    lines.append(
        f"{r['th']:>6.2f} | {r['flagged']:>7d} | {r['tp']:>5d} | {r['fp']:>5d} | "
        f"{r['precision']:>9.1%} | {r['recall_viol']:>8.1%} | {r['fp_rate']:>9.1%}"
    )
lines.append("")

lines.append("--- 参考: VLM 盲道违规判定 ---")
lines.append("  VLM precision: 77.9% (116/149)")
lines.append("")

lines.append("--- 覆盖提升潜力分析 ---")
viol_with_blind = sum(1 for r in records if r["gt"] == "no" and r["has_both"])
viol_no_blind = total_violations - viol_with_blind
compliant_with_blind = sum(1 for r in records if r["gt"] == "yes" and r["has_both"])
lines.append(f"  违规样本(gt=no)共 {total_violations} 张:")
lines.append(f"    盲道已检出: {viol_with_blind} 张 ({viol_with_blind/total_violations:.1%})")
lines.append(f"    盲道未检出: {viol_no_blind} 张 ({viol_no_blind/total_violations:.1%})  <- YOLO 漏检, CV 无法覆盖")
lines.append(f"  合规样本(gt=yes)共 {total_compliant} 张:")
lines.append(f"    盲道已检出: {compliant_with_blind} 张 (可能误判盲道违规)")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"\n结果已写入 {OUT_PATH}")
print("\n".join(lines))
