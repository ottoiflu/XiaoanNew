"""Benchmark v2 CV 结果归因分析 + Grid Search

从 results.csv 加载 600 张预测，从 fourdim_gt_v2.json 加载新四维 GT，
做四维错误归因、grid search、硬否决模拟。
"""
import csv
import itertools
import json
import os
import sys
from collections import Counter, defaultdict

_PROJECT_ROOT = "/root/otto/XiaoanNew"
sys.path.insert(0, _PROJECT_ROOT)

from modules.experiment.scoring import ScoringConfig
from modules.vlm.parser import normalize_label

# ──────────────────────────── 加载数据 ────────────────────────────

CV_DIR = os.path.join(_PROJECT_ROOT, "outputs/benchmark_output/v2/base_prompt_cv")
CSV_PATH = os.path.join(CV_DIR, "results.csv")
GT_PATH = os.path.join(_PROJECT_ROOT, "data/benchmark/benchmark_v2/fourdim_gt_v2.json")

rows = []
with open(CSV_PATH, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        rows.append(r)

with open(GT_PATH, encoding="utf-8") as f:
    gt_items = {d["id"]: d for d in json.load(f)}

print(f"=== CV 轮结果概览 ===")
print(f"总样本: {len(rows)}")

# 分类
correct = [r for r in rows if normalize_label(r["gt"]) == normalize_label(r["pred"])]
wrong   = [r for r in rows if normalize_label(r["gt"]) != normalize_label(r["pred"])]
fn_list = [r for r in wrong if r["gt"] == "yes" and r["pred"] == "no"]
fp_list = [r for r in wrong if r["gt"] == "no" and r["pred"] == "yes"]
print(f"正确: {len(correct)} 错误: {len(wrong)}  (FP={len(fp_list)} FN={len(fn_list)})")

# ═══════════════════════════════════════════════════════════════
# 1. 错误归因：四维 status 分布
# ═══════════════════════════════════════════════════════════════

DIMS = ["composition_status", "angle_status", "distance_status", "context_status"]
DIM_NAMES = ["composition", "angle", "distance", "context"]

print("\n" + "=" * 60)
print("1. 错误归因: 四维 status 分布")
print("=" * 60)

def dim_status_summary(target_rows, label):
    print(f"\n--- {label} (n={len(target_rows)}) ---")
    for i, dim in enumerate(DIMS):
        cnt = Counter(r[dim] for r in target_rows)
        print(f"  {DIM_NAMES[i]}: {dict(cnt.most_common())}")

dim_status_summary(fp_list, "FP - 实违规判合规")
dim_status_summary(fn_list, "FN - 实合规判违规")
dim_status_summary(wrong, "所有错误")

# ═══════════════════════════════════════════════════════════════
# 2. 四维 GT 映射对比 (新四维 position/medium/angle/state)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("2. 新四维 GT vs 旧四维预测 映射对比")
print("=" * 60)

MAPPING = {
    "position": ["[无参照]", "[有参照]"],
    "medium": ["[合规]", "[禁停区]", "[非机动车道]"],
    "angle": ["[N/A]", "[合规]", "[不合规]"],
    "state": ["[正立]", "[倒放]"],
}

for new_dim, possible in MAPPING.items():
    print(f"\n--- GT[{new_dim}] vs 四维对照 ---")
    for val in possible:
        subset = [r for r in rows if gt_items.get(r["id"], {}).get(new_dim) == val]
        if not subset:
            continue
        # 旧四维中各维度的分布
        dist_summary = {}
        for i, dim in enumerate(DIMS):
            cnt = Counter(r[dim] for r in subset)
            dist_summary[DIM_NAMES[i]] = dict(cnt.most_common())
        # 准确率
        acc = sum(1 for r in subset if normalize_label(r["gt"]) == normalize_label(r["pred"]))
        print(f"  {new_dim}={val} (n={len(subset)}, acc={acc/len(subset):.3f})")
        for dn, dist in dist_summary.items():
            print(f"    {dn}: {dist}")

# 错误样本中 GT position 分布
print("\n--- 错误样本的 position/medium 分布 ---")
for label, target in [("FP", fp_list), ("FN", fn_list), ("ALL wrong", wrong)]:
    print(f"\n  {label}:")
    pos_cnt = Counter(gt_items.get(r["id"], {}).get("position", "?") for r in target)
    med_cnt = Counter(gt_items.get(r["id"], {}).get("medium", "?") for r in target)
    print(f"    position: {dict(pos_cnt.most_common())}")
    print(f"    medium:   {dict(med_cnt.most_common())}")

# ═══════════════════════════════════════════════════════════════
# 3. Grid Search: 扫权重+阈值
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("3. Grid Search: 权重+阈值扫描")
print("=" * 60)

# 基础 score_map（沿用 p4 映射）
BASE_SCORE_MAP = {
    "composition": {
        "[合规]": 1.0, "[基本合规]": 0.7,
        "[不合规-构图]": 0.0, "[不合规-无参照]": 0.0,
    },
    "angle": {
        "[合规]": 1.0, "[不合规-角度]": 0.0,
    },
    "distance": {
        "[完全合规]": 1.0, "[基本合规-压线]": 0.5,
        "[不合规-超界]": 0.0,
    },
    "context": {
        "[合规]": 1.0, "[不合规-环境]": 0.0,
    },
}

# 权重候选
WEIGHT_CANDIDATES = [
    # 基准
    {"composition": 0.05, "angle": 0.25, "distance": 0.40, "context": 0.30},
    # 降低 distance 权重
    {"composition": 0.05, "angle": 0.30, "distance": 0.30, "context": 0.35},
    {"composition": 0.05, "angle": 0.35, "distance": 0.20, "context": 0.40},
    {"composition": 0.05, "angle": 0.30, "distance": 0.25, "context": 0.40},
    # 提高 angle/context 权重
    {"composition": 0.05, "angle": 0.30, "distance": 0.35, "context": 0.30},
    {"composition": 0.10, "angle": 0.25, "distance": 0.35, "context": 0.30},
    # 极端：几乎无视 distance
    {"composition": 0.10, "angle": 0.40, "distance": 0.10, "context": 0.40},
    # 均衡
    {"composition": 0.10, "angle": 0.30, "distance": 0.30, "context": 0.30},
    {"composition": 0.15, "angle": 0.25, "distance": 0.30, "context": 0.30},
]

THRESHOLD_CANDIDATES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

def score_sample(r, weights, threshold):
    """对单样本打分"""
    statuses = [r["composition_status"], r["angle_status"],
                r["distance_status"], r["context_status"]]
    dim_names = ["composition", "angle", "distance", "context"]

    total = 0.0
    dim_scores = {}
    for dim, st, nm in zip(DIMS, statuses, dim_names):
        s = BASE_SCORE_MAP[nm].get(st, 0.0)
        dim_scores[nm] = s
        total += weights[nm] * s

    return total >= threshold, total

results = []
for weights in WEIGHT_CANDIDATES:
    for thresh in THRESHOLD_CANDIDATES:
        g_tp = g_tn = g_fp = g_fn = 0
        for r in rows:
            compliant, _ = score_sample(r, weights, thresh)
            pred = "yes" if compliant else "no"
            gt = normalize_label(r["gt"])
            if gt == "yes":
                g_tp += pred == "yes"
                g_fn += pred != "yes"
            else:
                g_tn += pred == "no"
                g_fp += pred != "no"

        total = g_tp + g_tn + g_fp + g_fn
        acc = (g_tp + g_tn) / total if total else 0
        pre = g_tp / (g_tp + g_fp) if (g_tp + g_fp) else 0
        rec = g_tp / (g_tp + g_fn) if (g_tp + g_fn) else 0
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
        viol_rec = g_tn / (g_tn + g_fp) if (g_tn + g_fp) else 0
        results.append((weights, thresh, acc, pre, rec, f1, viol_rec, g_tp, g_tn, g_fp, g_fn))

# 按 F1 排序
results.sort(key=lambda x: (-x[5], -x[3]))

print(f"\n{'权重':<55} {'阈值':<6} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'ViolRec':<8}")
print("-" * 105)
for w, t, acc, pre, rec, f1, vr, tp, tn, fp, fn in results:
    w_str = f"comp={w['composition']:.2f},angle={w['angle']:.2f},dist={w['distance']:.2f},ctx={w['context']:.2f}"
    print(f"{w_str:<55} {t:<6.2f} {acc:<8.4f} {pre:<8.4f} {rec:<8.4f} {f1:<8.4f} {vr:<8.4f}")

# Top 3 详细
print("\n--- Top 3 配置 ---")
for i, (w, t, acc, pre, rec, f1, vr, tp, tn, fp, fn) in enumerate(results[:3]):
    print(f"\n#{i+1}: weights={w}, threshold={t}")
    print(f"  Acc={acc:.4f} Prec={pre:.4f} Rec={rec:.4f} F1={f1:.4f} ViolRec={vr:.4f}")
    print(f"  TP={tp} FN={fn} FP={fp} TN={tn}")

# ═══════════════════════════════════════════════════════════════
# 4. distance=超界 硬否决模拟
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("4. distance=超界 硬否决模拟")
print("=" * 60)

# 当前 p4 配置 (权重+阈值)
P4_WEIGHTS = {"composition": 0.05, "angle": 0.20, "distance": 0.45, "context": 0.30}
P4_THRESH = 0.35
P4_SCORE_MAP = {
    "composition": {"[合规]": 1.0, "[基本合规]": 0.7, "[不合规-无参照]": 0.0, "[不合规-构图]": 0.0},
    "angle": {"[合规]": 1.0, "[基本合规-角度]": 0.5, "[不合规-角度]": 0.0},
    "distance": {"[完全合规]": 1.0, "[基本合规-压线]": 0.5, "[不合规-超界]": 0.0},
    "context": {"[合规]": 1.0, "[基本合规-环境]": 0.5, "[不合规-环境]": 0.0},
}

# a) 合规样本中 distance=超界 比例
compliant_gt = [r for r in rows if r["gt"] == "yes"]
dist_oob_among_compliant = [r for r in compliant_gt if r["distance_status"] == "[不合规-超界]"]
print(f"合规样本中 VLM 判 distance=[不合规-超界]: {len(dist_oob_among_compliant)}/{len(compliant_gt)} = {len(dist_oob_among_compliant)/len(compliant_gt)*100:.1f}%")

# b) 硬否决：distance=超界 or context=不合规环境 → 直接 no
def hard_veto(r):
    return r["distance_status"] == "[不合规-超界]" or r["context_status"] == "[不合规-环境]"

vtp = vtn = vfp = vfn = 0
for r in rows:
    pred = "no" if hard_veto(r) else "yes"
    gt = normalize_label(r["gt"])
    if gt == "yes":
        vtp += pred == "yes"
        vfn += pred != "yes"
    else:
        vtn += pred == "no"
        vfp += pred != "no"

v_total = vtp + vtn + vfp + vfn
v_acc = (vtp + vtn) / v_total
v_pre = vtp / (vtp + vfp) if (vtp + vfp) else 0
v_rec = vtp / (vtp + vfn) if (vtp + vfn) else 0
v_f1 = 2 * v_pre * v_rec / (v_pre + v_rec) if (v_pre + v_rec) else 0
v_vr = vtn / (vtn + vfp) if (vtn + vfp) else 0
print(f"\n硬否决 (distance=超界 OR context=不合规环境):")
print(f"  Acc={v_acc:.4f} Prec={v_pre:.4f} Rec={v_rec:.4f} F1={v_f1:.4f} ViolRec={v_vr:.4f}")
print(f"  TP={vtp} FN={vfn} FP={vfp} TN={vtn}")

# c) distance-only 硬否决
def dist_veto(r):
    return r["distance_status"] == "[不合规-超界]"

d_tp = d_tn = d_fp = d_fn = 0
for r in rows:
    pred = "no" if dist_veto(r) else "yes"
    gt = normalize_label(r["gt"])
    if gt == "yes":
        d_tp += pred == "yes"
        d_fn += pred != "yes"
    else:
        d_tn += pred == "no"
        d_fp += pred != "no"

d_total = d_tp + d_tn + d_fp + d_fn
d_acc = (d_tp + d_tn) / d_total
d_pre = d_tp / (d_tp + d_fp) if (d_tp + d_fp) else 0
d_rec = d_tp / (d_tp + d_fn) if (d_tp + d_fn) else 0
d_f1 = 2 * d_pre * d_rec / (d_pre + d_rec) if (d_pre + d_rec) else 0
d_vr = d_tn / (d_tn + d_fp) if (d_tn + d_fp) else 0
print(f"\ndistance-only 硬否决:")
print(f"  Acc={d_acc:.4f} Prec={d_pre:.4f} Rec={d_rec:.4f} F1={d_f1:.4f} ViolRec={d_vr:.4f}")
print(f"  TP={d_tp} FN={d_fn} FP={d_fp} TN={d_tn}")

# ═══════════════════════════════════════════════════════════════
# 5. FP/FN 细节：每个错误样本四维展开
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("5. 错误样本展开 (前 20)")
print("=" * 60)

for label, target in [("FN - 实合规判违规", fn_list), ("FP - 实违规判合规", fp_list)]:
    print(f"\n--- {label} ---")
    for r in target[:10]:
        g = gt_items.get(r["id"], {})
        pos = g.get("position", "?")
        med = g.get("medium", "?")
        print(f"  id={r['id'][:12]} score={r['final_score']}")
        print(f"    GT: pos={pos} med={med}  PRED: comp={r['composition_status']} angle={r['angle_status']} dist={r['distance_status']} ctx={r['context_status']}")

print("\n=== 分析完成 ===")