"""新四维 Grid Search V2 — 扫权重+阈值+[无参照]分值

同时在 train(300) 上搜索：
- 12 组权重组合
- 8 个阈值
- 5 个 [无参照] 分值候选 (0.0/0.1/0.2/0.3/0.5)

val(150) 验证，输出最优配置。
"""
import csv
import json
import os
import sys

_PROJECT_ROOT = "/root/otto/XiaoanNew"
sys.path.insert(0, _PROJECT_ROOT)

from modules.vlm.parser import normalize_label

SPLIT_PATH = os.path.join(_PROJECT_ROOT, "data/benchmark/benchmark_v2/split.json")
CV_RESULTS = os.path.join(_PROJECT_ROOT, "outputs/benchmark_output/v2/new4dim_baseline/results.csv")

with open(SPLIT_PATH, encoding="utf-8") as f:
    SPLIT_MAP = json.load(f)

print("=== 加载数据 ===")
rows = []
with open(CV_RESULTS, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        r["split"] = SPLIT_MAP.get(r["id"], "test")
        rows.append(r)
print(f"Total: {len(rows)}")

CSV_TO_DIM = {
    "position": "position_status",
    "medium": "medium_status",
    "angle": "angle_status",
    "state": "state_status",
}
DIMS = list(CSV_TO_DIM.keys())

# 固定 score_map（[无参照] 分值用占位符，在 score_one 中替换）
BASE_SCORE_MAP = {
    "position": {
        "[合规]": 1.0, "[基本合规-压线]": 0.6,
        "[不合规-超界]": 0.0, "[无参照]": None,
    },
    "medium": {
        "[合规]": 1.0, "[不合规-盲道]": 0.0,
        "[不合规-绿化]": 0.0, "[不合规-禁停区]": 0.0,
    },
    "angle": {
        "[合规]": 1.0, "[不合规-斜停]": 0.0, "[N/A]": None,
    },
    "state": {
        "[正立]": 1.0, "[倒伏]": 0.0,
    },
}

NO_REF_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.5]

WEIGHTS_CANDIDATES = [
    {"position": 0.15, "medium": 0.45, "angle": 0.30, "state": 0.10},
    {"position": 0.15, "medium": 0.50, "angle": 0.25, "state": 0.10},
    {"position": 0.10, "medium": 0.50, "angle": 0.30, "state": 0.10},
    {"position": 0.10, "medium": 0.45, "angle": 0.35, "state": 0.10},
    {"position": 0.20, "medium": 0.40, "angle": 0.25, "state": 0.15},
    {"position": 0.20, "medium": 0.35, "angle": 0.30, "state": 0.15},
    {"position": 0.15, "medium": 0.55, "angle": 0.20, "state": 0.10},
    {"position": 0.10, "medium": 0.55, "angle": 0.25, "state": 0.10},
    {"position": 0.25, "medium": 0.35, "angle": 0.25, "state": 0.15},
    {"position": 0.25, "medium": 0.30, "angle": 0.30, "state": 0.15},
    {"position": 0.30, "medium": 0.30, "angle": 0.25, "state": 0.15},
    {"position": 0.30, "medium": 0.35, "angle": 0.20, "state": 0.15},
]

THRESH_CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def score_one(r, weights, thresh, no_ref_score):
    scores = {}
    active_w = {}
    for dim in DIMS:
        val = r[CSV_TO_DIM[dim]]
        if dim == "position" and val == "[无参照]":
            s = no_ref_score
        else:
            s = BASE_SCORE_MAP[dim].get(val, 0.0)
        if s is None:
            continue
        scores[dim] = s
        active_w[dim] = weights[dim]

    w_sum = sum(active_w.values())
    if w_sum == 0:
        return False, 0.0

    total = sum(scores[d] * active_w[d] / w_sum for d in scores)
    return total >= thresh, round(total, 4)


def evaluate(subset, weights, thresh, no_ref):
    tp = tn = fp = fn = 0
    for r in subset:
        compliant, _ = score_one(r, weights, thresh, no_ref)
        pred = "yes" if compliant else "no"
        gt = normalize_label(r["gt"])
        if gt == "yes":
            tp += pred == "yes"
            fn += pred != "yes"
        else:
            tn += pred == "no"
            fp += pred != "no"

    total = tp + tn + fp + fn
    if total == 0:
        return None
    acc = (tp + tn) / total
    pre = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
    vr = tn / (tn + fp) if (tn + fp) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    ba = (rec + spec) / 2
    return {"acc": round(acc, 4), "ba": round(ba, 4), "f1": round(f1, 4),
            "vr": round(vr, 4), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


train = [r for r in rows if r["split"] == "train"]
val = [r for r in rows if r["split"] == "val"]
test = [r for r in rows if r["split"] == "test"]
print(f"train={len(train)} val={len(val)} test={len(test)}")

print(f"\n{'[无参照]分值':<12} {'权重':<52} {'Th':<5} {'Acc':<8} {'BA':<8} {'F1':<8} {'VR':<8} {'TP':<5} {'TN':<5} {'FP':<5} {'FN':<5}")
print("-" * 125)

results = []
for no_ref in NO_REF_CANDIDATES:
    for w in WEIGHTS_CANDIDATES:
        for th in THRESH_CANDIDATES:
            m = evaluate(train, w, th, no_ref)
            if m is None:
                continue
            w_str = ",".join(f"{k}={v:.2f}" for k, v in w.items())
            print(f"{no_ref:<12} {w_str:<52} {th:<5.2f} {m['acc']:<8.4f} {m['ba']:<8.4f} {m['f1']:<8.4f} {m['vr']:<8.4f} {m['tp']:<5} {m['tn']:<5} {m['fp']:<5} {m['fn']:<5}")
            results.append((no_ref, w, th, m))

if not results:
    print("无结果")
    sys.exit(0)

# Top by balanced_accuracy with VR>=0.6 constraint
qualified = [r for r in results if r[3]["vr"] >= 0.6]
qualified.sort(key=lambda x: -x[3]["ba"])

print(f"\n{'='*60}")
print(f"VR>=0.6 配置数: {len(qualified)}")
if qualified:
    print(f"\n=== Top 5 (VR>=0.6, sorted by BA) ===")
    for i, (no_ref, w, th, m) in enumerate(qualified[:5]):
        m_v = evaluate(val, w, th, no_ref)
        print(f"\n#{i+1}: no_ref_score={no_ref}, weights={w}, thresh={th}")
        print(f"  train: Acc={m['acc']} BA={m['ba']} F1={m['f1']} VR={m['vr']}")
        if m_v:
            print(f"  val:   Acc={m_v['acc']} BA={m_v['ba']} F1={m_v['f1']} VR={m_v['vr']}")
else:
    # 找 VR 最高的
    results.sort(key=lambda x: (-x[3]["vr"], -x[3]["ba"]))
    print(f"\n=== Top 5 (all, sorted by VR desc) ===")
    for i, (no_ref, w, th, m) in enumerate(results[:5]):
        m_v = evaluate(val, w, th, no_ref)
        print(f"\n#{i+1}: no_ref_score={no_ref}, weights={w}, thresh={th}")
        print(f"  train: Acc={m['acc']} BA={m['ba']} F1={m['f1']} VR={m['vr']}")
        if m_v:
            print(f"  val:   Acc={m_v['acc']} BA={m_v['ba']} F1={m_v['f1']} VR={m_v['vr']}")

# Best overall by BA (no VR constraint)
results.sort(key=lambda x: -x[3]["ba"])
print(f"\n{'='*60}")
print(f"=== Best overall by BA ===")
no_ref, w, th, m = results[0]
m_v = evaluate(val, w, th, no_ref)
m_t = evaluate(test, w, th, no_ref)
print(f"no_ref_score={no_ref}, weights={w}, thresh={th}")
print(f"train: Acc={m['acc']} BA={m['ba']} F1={m['f1']} VR={m['vr']}")
if m_v:
    print(f"val:   Acc={m_v['acc']} BA={m_v['ba']} F1={m_v['f1']} VR={m_v['vr']}")
if m_t:
    print(f"test:  Acc={m_t['acc']} BA={m_t['ba']} F1={m_t['f1']} VR={m_t['vr']}")

# Save best config
config = {
    "score_map": {
        "position": {"[合规]": 1.0, "[基本合规-压线]": 0.6,
                      "[不合规-超界]": 0.0, "[无参照]": no_ref},
        "medium": {"[合规]": 1.0, "[不合规-盲道]": 0.0,
                    "[不合规-绿化]": 0.0, "[不合规-禁停区]": 0.0},
        "angle": {"[合规]": 1.0, "[不合规-斜停]": 0.0, "[N/A]": None},
        "state": {"[正立]": 1.0, "[倒伏]": 0.0},
    },
    "weights": w,
    "threshold": th,
}
out_path = os.path.join(_PROJECT_ROOT, "assets/configs/scoring_new4d_gs_best.yaml")
import yaml
with open(out_path, "w", encoding="utf-8") as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
print(f"\nConfig saved: {out_path}")