"""加权评分策略全面网格搜索

对所有实验 CSV 数据执行三层搜索：
1. 阈值扫描 (0.10-0.95, step=0.05)
2. 权重网格搜索 (四维度组合)
3. 分数映射变体 (基本合规/压线的中间得分)

输出最优方案对比表和可视化图表。
"""

from __future__ import annotations

import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modules.experiment.scoring import ScoringConfig, ScoringEngine

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = "outputs/contrast_experiments/scoring_search"
os.makedirs(OUT_DIR, exist_ok=True)


# ────────────────── 数据加载 ──────────────────


def load_csv(path: str) -> list[dict]:
    """加载CSV结果文件"""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_gt(row: dict) -> str:
    """提取真实标签"""
    gt = row.get("gt", row.get("ground_truth", "")).strip().lower()
    return "yes" if gt in ("yes", "合规") else "no"


# ────────────────── 评估函数 ──────────────────


def evaluate_with_config(
    rows: list[dict],
    weights: dict[str, float],
    threshold: float,
    score_map: dict | None = None,
    composition_gate: bool = True,
) -> dict:
    """使用指定配置评估CSV数据，返回指标"""
    cfg = ScoringConfig.default()
    cfg.weights = weights
    cfg.threshold = threshold
    cfg.composition_gate = composition_gate
    if score_map:
        cfg.score_map = score_map
    engine = ScoringEngine(cfg)

    tp = tn = fp = fn = 0
    for row in rows:
        gt = get_gt(row)
        result = engine.score(
            row.get("composition", ""),
            row.get("angle", ""),
            row.get("distance", ""),
            row.get("context", ""),
        )
        pred = "yes" if result.is_compliant else "no"
        if gt == "yes":
            if pred == "yes":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "no":
                tn += 1
            else:
                fp += 1

    total = tp + tn + fp + fn
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    return {"acc": acc, "pre": pre, "rec": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ────────────────── 搜索策略 ──────────────────


def sweep_thresholds(rows, weights, thresholds, score_map=None, gate=True):
    """扫描阈值列表，返回每个阈值的指标"""
    results = []
    for t in thresholds:
        m = evaluate_with_config(rows, weights, t, score_map, gate)
        m["threshold"] = t
        results.append(m)
    return results


def grid_search_weights_and_threshold(rows, score_map=None, gate=True):
    """网格搜索权重+阈值组合"""
    weight_options = {
        "composition": [0.00, 0.05, 0.10],
        "angle": [0.15, 0.20, 0.25, 0.30, 0.35],
        "distance": [0.25, 0.30, 0.35, 0.40, 0.45],
        "context": [0.15, 0.20, 0.25, 0.30, 0.35],
    }
    thresholds = [round(0.10 + i * 0.05, 2) for i in range(18)]

    dims = ["composition", "angle", "distance", "context"]
    combos = list(itertools.product(*[weight_options[d] for d in dims]))
    valid = [(dict(zip(dims, c))) for c in combos if abs(sum(c) - 1.0) < 0.015]

    print(f"  有效权重组合数: {len(valid)}, 阈值数: {len(thresholds)}, 总搜索空间: {len(valid) * len(thresholds)}")

    best_f1 = {"f1": -1}
    best_acc = {"acc": -1}
    best_balanced = {"score": -1}
    all_results = []

    for w in valid:
        for t in thresholds:
            m = evaluate_with_config(rows, w, t, score_map, gate)
            m["weights"] = dict(w)
            m["threshold"] = t
            balanced = 0.6 * m["f1"] + 0.4 * m["acc"]
            m["balanced"] = balanced

            if m["f1"] > best_f1["f1"]:
                best_f1 = dict(m)
            if m["acc"] > best_acc["acc"]:
                best_acc = dict(m)
            if balanced > best_balanced["score"]:
                best_balanced = dict(m)
                best_balanced["score"] = balanced

            all_results.append(m)

    return best_f1, best_acc, best_balanced, all_results


def search_score_maps(rows):
    """搜索不同的分数映射方案"""
    base_map = ScoringConfig.default().score_map

    variants = {
        "default": base_map,
        "soft_distance": {
            **base_map,
            "distance": {
                "[完全合规]": 1.0,
                "[基本合规-压线]": 0.5,
                "[不合规-超界]": 0.0,
            },
        },
        "softer_distance": {
            **base_map,
            "distance": {
                "[完全合规]": 1.0,
                "[基本合规-压线]": 0.7,
                "[不合规-超界]": 0.0,
            },
        },
        "soft_composition": {
            **{k: dict(v) for k, v in base_map.items()},
        },
        "soft_all": {
            "composition": {
                "[合规]": 1.0,
                "[基本合规]": 0.8,
                "[不合规-构图]": 0.0,
                "[不合规-无参照]": 0.0,
            },
            "angle": {
                "[合规]": 1.0,
                "[不合规-角度]": 0.0,
            },
            "distance": {
                "[完全合规]": 1.0,
                "[基本合规-压线]": 0.5,
                "[不合规-超界]": 0.0,
            },
            "context": {
                "[合规]": 1.0,
                "[不合规-环境]": 0.0,
            },
        },
        "no_gate_soft": {
            "composition": {
                "[合规]": 1.0,
                "[基本合规]": 0.8,
                "[不合规-构图]": 0.3,
                "[不合规-无参照]": 0.2,
            },
            "angle": {
                "[合规]": 1.0,
                "[不合规-角度]": 0.0,
            },
            "distance": {
                "[完全合规]": 1.0,
                "[基本合规-压线]": 0.5,
                "[不合规-超界]": 0.0,
            },
            "context": {
                "[合规]": 1.0,
                "[不合规-环境]": 0.0,
            },
        },
    }
    # soft_composition 要单独设置
    variants["soft_composition"]["composition"] = {
        "[合规]": 1.0,
        "[基本合规]": 0.8,
        "[不合规-构图]": 0.0,
        "[不合规-无参照]": 0.0,
    }

    return variants


# ────────────────── 主流程 ──────────────────


def main():
    """执行全面网格搜索"""
    csv_files = {
        "cv_p4": "outputs/contrast_experiments/20260328_184334_cv_p4_veto/cv_p4_veto.csv",
        "vlm_p4": "outputs/contrast_experiments/20260328_184905_vlm_p4_veto/vlm_p4_veto.csv",
        "cv_p6": "outputs/contrast_experiments/20260328_183512_cv_p6_veto/cv_p6_veto.csv",
        "vlm_p6": "outputs/contrast_experiments/20260328_183450_vlm_p6_veto/vlm_p6_veto.csv",
        "cv_p5": "outputs/contrast_experiments/20260328_184055_cv_p5_veto/cv_p5_veto.csv",
        "vlm_p5": "outputs/contrast_experiments/20260328_184030_vlm_p5_veto/vlm_p5_veto.csv",
    }

    # 验证文件都存在
    for name, path in csv_files.items():
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping {name}")

    all_best = {}
    score_map_variants = None

    # ========== Phase 1: 对每个实验CSV做权重+阈值网格搜索 ==========
    print("=" * 70)
    print("Phase 1: 权重 + 阈值网格搜索")
    print("=" * 70)

    for name, path in csv_files.items():
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        print(f"\n--- {name} ({len(rows)} samples) ---")
        best_f1, best_acc, best_bal, _ = grid_search_weights_and_threshold(rows)

        all_best[name] = {
            "best_f1": best_f1,
            "best_acc": best_acc,
            "best_balanced": best_bal,
        }
        print(
            f"  Best F1:  {best_f1['f1']:.4f} (Acc={best_f1['acc']:.4f}) "
            f"thresh={best_f1['threshold']:.2f} w={best_f1['weights']}"
        )
        print(
            f"  Best Acc: {best_acc['acc']:.4f} (F1={best_acc['f1']:.4f}) "
            f"thresh={best_acc['threshold']:.2f} w={best_acc['weights']}"
        )
        print(
            f"  Best Balanced(0.6*F1+0.4*Acc): {best_bal['balanced']:.4f} "
            f"(F1={best_bal['f1']:.4f} Acc={best_bal['acc']:.4f}) "
            f"thresh={best_bal['threshold']:.2f} w={best_bal['weights']}"
        )

    # ========== Phase 2: 分数映射变体搜索 (仅对top实验) ==========
    print("\n" + "=" * 70)
    print("Phase 2: 分数映射变体搜索 (cv_p4 & vlm_p4)")
    print("=" * 70)

    score_map_variants = search_score_maps(None)
    map_results = {}

    for exp_name in ["cv_p4", "vlm_p4"]:
        path = csv_files[exp_name]
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        map_results[exp_name] = {}
        print(f"\n--- {exp_name} ---")

        for variant_name, smap in score_map_variants.items():
            gate = variant_name != "no_gate_soft"
            best_f1, best_acc, best_bal, _ = grid_search_weights_and_threshold(rows, score_map=smap, gate=gate)
            map_results[exp_name][variant_name] = {
                "best_f1": best_f1,
                "best_acc": best_acc,
                "best_balanced": best_bal,
            }
            print(
                f"  [{variant_name}] F1={best_f1['f1']:.4f} Acc={best_f1['acc']:.4f} thresh={best_f1['threshold']:.2f}"
            )

    # ========== Phase 3: 阈值敏感性曲线 (按最优权重) ==========
    print("\n" + "=" * 70)
    print("Phase 3: 阈值敏感性曲线")
    print("=" * 70)

    thresholds = [round(0.05 + i * 0.05, 2) for i in range(19)]
    threshold_curves = {}

    for exp_name in ["cv_p4", "vlm_p4", "cv_p6", "vlm_p6"]:
        path = csv_files[exp_name]
        if not os.path.exists(path):
            continue
        rows = load_csv(path)
        best_w = all_best[exp_name]["best_f1"]["weights"]
        curve = sweep_thresholds(rows, best_w, thresholds)
        threshold_curves[exp_name] = curve
        peaks = max(curve, key=lambda x: x["f1"])
        print(f"  {exp_name}: peak F1={peaks['f1']:.4f} @ thresh={peaks['threshold']:.2f}")

    # ========== 输出汇总 ==========
    print("\n" + "=" * 70)
    print("汇总: 各实验最优加权方案 vs 一票否决基线")
    print("=" * 70)

    baselines = {
        "cv_p4": {"acc": 0.74, "pre": 0.7069, "rec": 0.82, "f1": 0.7593},
        "vlm_p4": {"acc": 0.75, "pre": 0.8378, "rec": 0.62, "f1": 0.7126},
        "cv_p6": {"acc": 0.62, "pre": 0.70, "rec": 0.42, "f1": 0.5250},
        "vlm_p6": {"acc": 0.6263, "pre": 0.5769, "rec": 0.9184, "f1": 0.7087},
        "cv_p5": {"acc": 0.6263, "pre": 0.6364, "rec": 0.5714, "f1": 0.6022},
        "vlm_p5": {"acc": 0.63, "pre": 0.6585, "rec": 0.54, "f1": 0.5934},
    }

    print(f"\n{'实验':<12} {'模式':<8} {'一票否决F1':>10} {'最优加权F1':>10} {'改进':>8} {'最优阈值':>8} {'权重组合'}")
    print("-" * 90)
    for name in ["cv_p4", "vlm_p4", "cv_p5", "vlm_p5", "cv_p6", "vlm_p6"]:
        if name not in all_best:
            continue
        b = baselines[name]
        opt = all_best[name]["best_f1"]
        delta_f1 = opt["f1"] - b["f1"]
        w_str = f"A={opt['weights']['angle']:.2f} D={opt['weights']['distance']:.2f} C={opt['weights']['context']:.2f}"
        print(
            f"{name:<12} {'加权':>6}  {b['f1']:>9.4f} {opt['f1']:>10.4f} "
            f"{delta_f1:>+7.4f}  {opt['threshold']:>7.2f}  {w_str}"
        )

    # ========== 可视化 ==========
    # 图1: 阈值敏感性曲线
    if threshold_curves:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        colors = {"cv_p4": "#E5634D", "vlm_p4": "#4A90D9", "cv_p6": "#F5A07A", "vlm_p6": "#7EB8DA"}

        for exp_name, curve in threshold_curves.items():
            ts = [c["threshold"] for c in curve]
            f1s = [c["f1"] * 100 for c in curve]
            accs = [c["acc"] * 100 for c in curve]
            axes[0].plot(ts, f1s, "o-", label=exp_name, color=colors.get(exp_name, "gray"), markersize=4)
            axes[1].plot(ts, accs, "s-", label=exp_name, color=colors.get(exp_name, "gray"), markersize=4)

        axes[0].set_xlabel("阈值")
        axes[0].set_ylabel("F1 (%)")
        axes[0].set_title("F1 vs 阈值", fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("阈值")
        axes[1].set_ylabel("准确率 (%)")
        axes[1].set_title("准确率 vs 阈值", fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/fig5_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved: {OUT_DIR}/fig5_threshold_sensitivity.png")

    # 图2: 分数映射变体对比
    if map_results:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for idx, exp_name in enumerate(["cv_p4", "vlm_p4"]):
            if exp_name not in map_results:
                continue
            variants = map_results[exp_name]
            names = list(variants.keys())
            f1s = [variants[n]["best_f1"]["f1"] * 100 for n in names]
            accs = [variants[n]["best_f1"]["acc"] * 100 for n in names]

            x = np.arange(len(names))
            w = 0.35
            axes[idx].bar(x - w / 2, f1s, w, label="F1", color="#4A90D9")
            axes[idx].bar(x + w / 2, accs, w, label="Acc", color="#E5634D")
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
            axes[idx].set_ylabel("%")
            axes[idx].set_title(f"{exp_name} 分数映射变体", fontweight="bold")
            axes[idx].legend()
            axes[idx].grid(axis="y", alpha=0.3)
            axes[idx].set_ylim(50, 90)
            for i, (f, a) in enumerate(zip(f1s, accs)):
                axes[idx].text(i - w / 2, f + 0.5, f"{f:.1f}", ha="center", fontsize=7)
                axes[idx].text(i + w / 2, a + 0.5, f"{a:.1f}", ha="center", fontsize=7)

        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/fig6_score_map_variants.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUT_DIR}/fig6_score_map_variants.png")

    # 图3: 最优方案 vs 一票否决 对比
    fig, ax = plt.subplots(figsize=(12, 6))
    exps = ["cv_p4", "vlm_p4", "cv_p5", "vlm_p5", "cv_p6", "vlm_p6"]
    exps = [e for e in exps if e in all_best]
    veto_f1 = [baselines[e]["f1"] * 100 for e in exps]
    opt_f1 = [all_best[e]["best_f1"]["f1"] * 100 for e in exps]

    x = np.arange(len(exps))
    w = 0.35
    bars1 = ax.bar(x - w / 2, veto_f1, w, label="一票否决", color="#7EB8DA", edgecolor="white")
    bars2 = ax.bar(x + w / 2, opt_f1, w, label="最优加权", color="#E5634D", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(exps, fontsize=10)
    ax.set_ylabel("F1 (%)", fontsize=12)
    ax.set_title("一票否决 vs 最优加权评分 F1 对比", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(40, 90)
    for b in [bars1, bars2]:
        for bar in b:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig7_veto_vs_optimal_weighted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR}/fig7_veto_vs_optimal_weighted.png")

    # ========== 写出最优配置YAML ==========
    if "cv_p4" in all_best:
        best = all_best["cv_p4"]["best_f1"]
        cfg = ScoringConfig.default()
        cfg.weights = best["weights"]
        cfg.threshold = best["threshold"]
        yaml_path = "assets/configs/scoring_optimized_cv_p4.yaml"
        cfg.to_yaml(yaml_path)
        print(f"\nSaved optimal config: {yaml_path}")

    if "vlm_p4" in all_best:
        best = all_best["vlm_p4"]["best_f1"]
        cfg = ScoringConfig.default()
        cfg.weights = best["weights"]
        cfg.threshold = best["threshold"]
        yaml_path = "assets/configs/scoring_optimized_vlm_p4.yaml"
        cfg.to_yaml(yaml_path)
        print(f"Saved optimal config: {yaml_path}")

    # 如果分数映射搜索有更好的结果，也保存
    if map_results and "cv_p4" in map_results:
        best_variant = max(
            map_results["cv_p4"].items(),
            key=lambda x: x[1]["best_f1"]["f1"],
        )
        vname, vdata = best_variant
        if vdata["best_f1"]["f1"] > all_best.get("cv_p4", {}).get("best_f1", {}).get("f1", 0):
            print(f"\n分数映射变体 [{vname}] 优于默认映射!")
            best = vdata["best_f1"]
            cfg = ScoringConfig.default()
            cfg.weights = best["weights"]
            cfg.threshold = best["threshold"]
            if vname in score_map_variants:
                cfg.score_map = score_map_variants[vname]
            if vname == "no_gate_soft":
                cfg.composition_gate = False
            yaml_path = f"assets/configs/scoring_optimized_cv_p4_{vname}.yaml"
            cfg.to_yaml(yaml_path)
            print(f"Saved: {yaml_path}")

    print("\n全部搜索完成。")


if __name__ == "__main__":
    main()
