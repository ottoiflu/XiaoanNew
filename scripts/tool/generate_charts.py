"""实验结果可视化图表生成器

生成出版级对比图表，覆盖：
  1. F1 分组柱状图（按 prompt 版本 × 工作流模式）
  2. 精确率-召回率散点图
  3. 热力图：prompt × mode → F1
  4. p4 系列消融分析
  5. 误差分布（FP/FN 堆叠图）
  6. 评分策略对比（veto vs weighted）
  7. 延迟对比
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# 中文字体
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_CSV = os.path.join(PROJECT_ROOT, "outputs/contrast_experiments/all_results.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs/contrast_experiments/figures")
os.makedirs(OUT_DIR, exist_ok=True)

# 专业配色 (colorblind-friendly)
C_VLM = "#2274A5"  # 深蓝
C_CV = "#E84855"  # 红
C_CVMIN = "#F9A03F"  # 橙
C_VETO = "#1B998B"  # 青
C_WEIGHTED = "#8B5CF6"  # 紫
C_OPT = "#F472B6"  # 粉

PALETTE = {
    "pure_vlm": C_VLM,
    "vlm_cv": C_CV,
    "vlm_cv_minimal": C_CVMIN,
}


def load_data():
    """加载实验数据"""
    rows = []
    with open(DATA_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["f1"] = float(row["f1"])
            row["acc"] = float(row["acc"])
            row["pre"] = float(row["pre"])
            row["rec"] = float(row["rec"])
            row["tp"] = int(row["tp"])
            row["tn"] = int(row["tn"])
            row["fp"] = int(row["fp"])
            row["fn"] = int(row["fn"])
            row["avg_lat"] = float(row["avg_lat"])
            rows.append(row)
    return rows


def fig1_f1_grouped_bar(data):
    """F1 分组柱状图：按 prompt 版本分组，VLM vs VLM+CV"""
    # 筛选 veto 实验, 排除 minimal
    veto = [r for r in data if r["scoring"] == "veto" and "minimal" not in r["exp_name"]]

    prompt_order = ["p4", "p4.3", "p4.1", "p4.2", "p5", "p6", "p7"]
    prompt_map_vlm = {
        "standard_p4": "p4",
        "standard_p4_1": "p4.1",
        "standard_p4_2": "p4.2",
        "standard_p5": "p5",
        "standard_p6": "p6",
        "standard_p7": "p7",
    }
    prompt_map_cv = {
        "cv_enhanced_p4": "p4",
        "cv_enhanced_p4_1": "p4.1",
        "cv_enhanced_p4_2": "p4.2",
        "cv_enhanced_p4_3": "p4.3",
        "cv_enhanced_p5": "p5",
        "cv_enhanced_p6": "p6",
        "cv_enhanced_p7": "p7",
    }

    vlm_f1 = {}
    cv_f1 = {}
    for r in veto:
        pid = r["prompt_id"]
        if r["mode"] == "pure_vlm" and pid in prompt_map_vlm:
            vlm_f1[prompt_map_vlm[pid]] = r["f1"]
        elif r["mode"] == "vlm_cv" and pid in prompt_map_cv:
            cv_f1[prompt_map_cv[pid]] = r["f1"]

    prompts = [p for p in prompt_order if p in vlm_f1 or p in cv_f1]
    x = np.arange(len(prompts))
    w = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    vlm_vals = [vlm_f1.get(p, 0) for p in prompts]
    cv_vals = [cv_f1.get(p, 0) for p in prompts]

    bars1 = ax.bar(x - w / 2, vlm_vals, w, label="VLM-only", color=C_VLM, edgecolor="white", linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + w / 2, cv_vals, w, label="VLM + CV", color=C_CV, edgecolor="white", linewidth=0.8, zorder=3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.005,
                    f"{h:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    # 标注最优
    best_f1 = max(max(vlm_vals), max(cv_vals))
    for bars, vals in [(bars1, vlm_vals), (bars2, cv_vals)]:
        for bar, v in zip(bars, vals):
            if v == best_f1:
                ax.annotate(
                    "BEST",
                    xy=(bar.get_x() + bar.get_width() / 2, v + 0.02),
                    fontsize=10,
                    fontweight="bold",
                    color="#16a34a",
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#dcfce7", edgecolor="#16a34a"),
                )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Prompt {p}" for p in prompts], fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_title("Veto Scoring: F1 by Prompt Version", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=11, framealpha=0.9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_f1_grouped_bar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def fig2_precision_recall_scatter(data):
    """精确率-召回率散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in data:
        mode = r["mode"]
        if "minimal" in r["exp_name"]:
            color = C_CVMIN
            marker = "D"
        elif mode == "pure_vlm":
            color = C_VLM
            marker = "o"
        else:
            color = C_CV
            marker = "s"

        size = 120 if r["scoring"] == "veto" else 80
        alpha = 1.0 if r["scoring"] == "veto" else 0.6
        edgecolor = "black" if r["scoring"] == "veto" else "gray"

        ax.scatter(
            r["rec"],
            r["pre"],
            c=color,
            s=size,
            marker=marker,
            alpha=alpha,
            edgecolors=edgecolor,
            linewidths=0.8,
            zorder=3,
        )

        # 标注关键实验
        if r["exp_name"] in ("cv_p4_veto", "vlm_p4_veto", "cv_p4_3_veto"):
            offset = (8, 8) if r["exp_name"] != "vlm_p4_veto" else (8, -12)
            ax.annotate(
                r["exp_name"].replace("_veto", ""),
                (r["rec"], r["pre"]),
                textcoords="offset points",
                xytext=offset,
                fontsize=8,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )

    # F1 等高线
    p_range = np.linspace(0.01, 1, 200)
    r_range = np.linspace(0.01, 1, 200)
    R, P = np.meshgrid(r_range, p_range)
    F1 = 2 * P * R / (P + R)
    levels = [0.50, 0.60, 0.70, 0.75, 0.80]
    cs = ax.contour(R, P, F1, levels=levels, colors="gray", alpha=0.4, linestyles="--", linewidths=0.8)
    ax.clabel(cs, fmt="F1=%.2f", fontsize=8, inline=True)

    # 图例
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_VLM, markersize=10, label="VLM-only"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_CV, markersize=10, label="VLM + CV"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_CVMIN, markersize=8, label="VLM + CV (strip geo)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=1.2,
            label="Veto scoring",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, alpha=0.6, label="Weighted scoring"
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_title("Precision-Recall Landscape (28 Experiments)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0.35, 1.05)
    ax.set_ylim(0.45, 0.95)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_precision_recall.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def fig3_heatmap(data):
    """热力图：prompt × (mode, scoring) → F1"""
    # 排除 minimal
    filtered = [r for r in data if "minimal" not in r["exp_name"]]

    prompt_map = {
        "standard_p4": "p4",
        "standard_p4_1": "p4.1",
        "standard_p4_2": "p4.2",
        "standard_p5": "p5",
        "standard_p6": "p6",
        "cv_enhanced_p4": "p4",
        "cv_enhanced_p4_1": "p4.1",
        "cv_enhanced_p4_2": "p4.2",
        "cv_enhanced_p4_3": "p4.3",
        "cv_enhanced_p5": "p5",
        "cv_enhanced_p6": "p6",
        "cv_enhanced_p7": "p7",
    }

    prompts = ["p4", "p4.1", "p4.2", "p4.3", "p5", "p6", "p7"]
    columns = ["VLM\nveto", "CV\nveto", "VLM\nweighted", "CV\nweighted"]
    matrix = np.full((len(prompts), len(columns)), np.nan)

    for r in filtered:
        pid = prompt_map.get(r["prompt_id"])
        if pid is None or pid not in prompts:
            continue
        row_idx = prompts.index(pid)

        if r["mode"] == "pure_vlm" and r["scoring"] == "veto":
            col_idx = 0
        elif r["mode"] == "vlm_cv" and r["scoring"] == "veto":
            col_idx = 1
        elif r["mode"] == "pure_vlm" and r["scoring"] == "weighted":
            col_idx = 2
        elif r["mode"] == "vlm_cv" and r["scoring"] == "weighted":
            col_idx = 3
        else:
            continue

        # 取最高 F1（如有多个同配置实验）
        if np.isnan(matrix[row_idx, col_idx]) or r["f1"] > matrix[row_idx, col_idx]:
            matrix[row_idx, col_idx] = r["f1"]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.45, vmax=0.80)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(prompts)))
    ax.set_yticklabels([f"Prompt {p}" for p in prompts], fontsize=11)

    # 标注数值
    for i in range(len(prompts)):
        for j in range(len(columns)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.58 or val > 0.74 else "black"
                fontw = "bold" if val > 0.72 else "normal"
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=11, fontweight=fontw, color=color)
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray", fontstyle="italic")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F1 Score", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ax.set_title("F1 Score Heatmap: Prompt x Pipeline x Scoring", fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def fig4_p4_ablation(data):
    """p4 系列消融分析 (veto): 双轴柱状图 + 折线"""
    versions = ["p4", "p4.1", "p4.2", "p4.3"]
    cv_map = {
        "cv_enhanced_p4": "p4",
        "cv_enhanced_p4_1": "p4.1",
        "cv_enhanced_p4_2": "p4.2",
        "cv_enhanced_p4_3": "p4.3",
    }

    cv_veto = {}
    for r in data:
        if r["scoring"] == "veto" and r["prompt_id"] in cv_map and r["mode"] == "vlm_cv":
            v = cv_map[r["prompt_id"]]
            cv_veto[v] = r

    versions = [v for v in versions if v in cv_veto]
    f1s = [cv_veto[v]["f1"] for v in versions]
    pres = [cv_veto[v]["pre"] for v in versions]
    recs = [cv_veto[v]["rec"] for v in versions]
    fps = [cv_veto[v]["fp"] for v in versions]
    fns = [cv_veto[v]["fn"] for v in versions]

    x = np.arange(len(versions))
    w = 0.25

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    # 柱状图：FP/FN
    b1 = ax1.bar(
        x - w / 2,
        fps,
        w,
        label="False Positives",
        color="#ef4444",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    b2 = ax1.bar(
        x + w / 2,
        fns,
        w,
        label="False Negatives",
        color="#f59e0b",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                str(int(h)),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # 折线：F1 / Pre / Rec
    ax2.plot(x, f1s, "o-", color="#16a34a", linewidth=2.5, markersize=8, label="F1", zorder=4)
    ax2.plot(x, pres, "s--", color=C_VLM, linewidth=1.8, markersize=6, label="Precision", zorder=4)
    ax2.plot(x, recs, "^--", color=C_CV, linewidth=1.8, markersize=6, label="Recall", zorder=4)

    for i, f in enumerate(f1s):
        ax2.annotate(
            f"{f:.1%}",
            (x[i], f),
            textcoords="offset points",
            xytext=(0, 10),
            fontsize=9,
            fontweight="bold",
            color="#16a34a",
            ha="center",
        )

    ax1.set_xticks(x)
    labels = []
    for v in versions:
        if v == "p4":
            labels.append("p4\n(baseline)")
        elif v == "p4.1":
            labels.append("p4.1\n(angle+IoU fix)")
        elif v == "p4.2":
            labels.append("p4.2\n(angle+IoU soft)")
        elif v == "p4.3":
            labels.append("p4.3\n(angle fix only)")
        else:
            labels.append(v)
    ax1.set_xticklabels(labels, fontsize=10)

    ax1.set_ylabel("Error Count (FP / FN)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.4, 1.0)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        ncol=5,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.12),
    )

    ax1.set_title("p4 Series Ablation Study (VLM+CV, Veto Scoring)", fontsize=14, fontweight="bold", pad=15)
    ax1.grid(axis="y", alpha=0.2, linestyle="--")
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_p4_ablation.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def fig5_error_breakdown(data):
    """FP/FN 水平堆叠条形图 (veto only, sorted by F1)"""
    veto = [r for r in data if r["scoring"] == "veto" and "minimal" not in r["exp_name"]]
    veto.sort(key=lambda r: r["f1"], reverse=True)

    names = [r["exp_name"] for r in veto]
    fps = [r["fp"] for r in veto]
    fns = [r["fn"] for r in veto]
    f1s = [r["f1"] for r in veto]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.4)))
    y = np.arange(len(names))

    ax.barh(y, fps, height=0.6, label="FP (Type I)", color="#ef4444", alpha=0.85, zorder=3)
    ax.barh(y, fns, height=0.6, left=fps, label="FN (Type II)", color="#f59e0b", alpha=0.85, zorder=3)

    # F1 标注
    for i, (fp, fn, f1) in enumerate(zip(fps, fns, f1s)):
        total_err = fp + fn
        ax.text(total_err + 0.5, i, f"F1={f1:.1%}", va="center", fontsize=9, fontweight="bold", color="#16a34a")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Error Count", fontsize=12, fontweight="bold")
    ax.set_title(
        "Error Distribution by Experiment (Veto Scoring, sorted by F1)", fontsize=14, fontweight="bold", pad=15
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_error_breakdown.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def fig6_veto_vs_weighted(data):
    """veto vs weighted F1 对比（配对箭头图）"""
    # 找配对
    pairs = {}
    for r in data:
        name = r["exp_name"]
        base = name.replace("_veto", "").replace("_weighted", "").replace("_opt_", "_")
        if "minimal" in name:
            continue
        scoring = "veto" if "veto" in name else "weighted"
        if base not in pairs:
            pairs[base] = {}
        if scoring not in pairs[base] or r["f1"] > pairs[base][scoring]["f1"]:
            pairs[base][scoring] = r

    # 筛出完整配对
    valid = {k: v for k, v in pairs.items() if "veto" in v and "weighted" in v}
    valid = dict(sorted(valid.items(), key=lambda x: x[1]["veto"]["f1"], reverse=True))

    fig, ax = plt.subplots(figsize=(12, max(6, len(valid) * 0.5)))
    y = np.arange(len(valid))
    names = list(valid.keys())

    for i, name in enumerate(names):
        veto_f1 = valid[name]["veto"]["f1"]
        wt_f1 = valid[name]["weighted"]["f1"]

        ax.scatter(veto_f1, i, color=C_VETO, s=100, zorder=4, marker="o", edgecolors="black", linewidths=0.5)
        ax.scatter(wt_f1, i, color=C_WEIGHTED, s=100, zorder=4, marker="s", edgecolors="black", linewidths=0.5)

        color = "#16a34a" if wt_f1 > veto_f1 else "#ef4444"
        ax.annotate("", xy=(wt_f1, i), xytext=(veto_f1, i), arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

        diff = wt_f1 - veto_f1
        mid = (veto_f1 + wt_f1) / 2
        ax.text(mid, i + 0.25, f"{diff:+.1%}", ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("F1 Score", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Veto vs Best Weighted Scoring (paired comparison)", fontsize=14, fontweight="bold", pad=15)

    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_VETO, markersize=10, label="Veto"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_WEIGHTED, markersize=10, label="Best Weighted"),
    ]
    ax.legend(handles=legend, fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_veto_vs_weighted.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def fig7_cv_contribution(data):
    """CV 贡献分析：VLM-only vs VLM+CV 差异瀑布图"""
    veto = [r for r in data if r["scoring"] == "veto" and "minimal" not in r["exp_name"]]

    prompt_map = {
        "standard_p4": "p4",
        "standard_p5": "p5",
        "standard_p6": "p6",
        "cv_enhanced_p4": "p4",
        "cv_enhanced_p5": "p5",
        "cv_enhanced_p6": "p6",
        "cv_enhanced_p7": "p7",
    }

    vlm_data = {}
    cv_data = {}
    for r in veto:
        pid = prompt_map.get(r["prompt_id"])
        if pid is None:
            continue
        if r["mode"] == "pure_vlm":
            vlm_data[pid] = r
        else:
            cv_data[pid] = r

    common = sorted(set(vlm_data.keys()) & set(cv_data.keys()))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("f1", "F1 Score"), ("pre", "Precision"), ("rec", "Recall")]

    for ax, (metric, title) in zip(axes, metrics):
        vlm_vals = [vlm_data[p][metric] for p in common]
        cv_vals = [cv_data[p][metric] for p in common]
        diffs = [c - v for c, v in zip(cv_vals, vlm_vals)]

        x = np.arange(len(common))
        colors = ["#16a34a" if d > 0 else "#ef4444" for d in diffs]

        ax.bar(x, diffs, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8, zorder=3)

        for i, d in enumerate(diffs):
            ax.text(
                i,
                d + (0.005 if d >= 0 else -0.015),
                f"{d:+.1%}",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=colors[i],
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"p{p.split('p')[1]}" for p in common], fontsize=10)
        ax.set_ylabel(f"{title} Change", fontsize=11)
        ax.set_title(f"CV Contribution: {title}", fontsize=12, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Impact of CV Integration (VLM+CV minus VLM-only, Veto Scoring)", fontsize=14, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig7_cv_contribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def fig8_latency_comparison(data):
    """延迟对比柱状图"""
    veto = [r for r in data if r["scoring"] == "veto" and "minimal" not in r["exp_name"]]
    veto.sort(key=lambda r: r["avg_lat"])

    names = [r["exp_name"] for r in veto]
    lats = [r["avg_lat"] for r in veto]
    colors = [C_VLM if r["mode"] == "pure_vlm" else C_CV for r in veto]

    fig, ax = plt.subplots(figsize=(12, max(5, len(names) * 0.35)))
    y = np.arange(len(names))

    ax.barh(y, lats, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8, height=0.6, zorder=3)

    for i, lat in enumerate(lats):
        ax.text(lat + 0.05, i, f"{lat:.2f}s", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Average Latency (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Average Inference Latency per Image (Veto Experiments)", fontsize=14, fontweight="bold", pad=15)

    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], color=C_VLM, lw=8, label="VLM-only"),
        Line2D([0], [0], color=C_CV, lw=8, label="VLM + CV"),
    ]
    ax.legend(handles=legend, fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig8_latency.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def fig9_confusion_matrices(data):
    """最优两组实验的混淆矩阵"""
    targets = ["cv_p4_veto", "vlm_p4_veto"]
    target_data = {r["exp_name"]: r for r in data if r["exp_name"] in targets}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, name in zip(axes, targets):
        if name not in target_data:
            ax.set_visible(False)
            continue

        r = target_data[name]
        cm = np.array([[r["tp"], r["fp"]], [r["fn"], r["tn"]]])
        total = cm.sum()

        ax.imshow(cm, cmap="Blues", vmin=0, vmax=50)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Positive\n(Compliant)", "Negative\n(Non-compliant)"], fontsize=10)
        ax.set_yticklabels(["Positive\n(Compliant)", "Negative\n(Non-compliant)"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=11, fontweight="bold")

        labels_map = [["TP", "FP"], ["FN", "TN"]]
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100
                color = "white" if val > 30 else "black"
                ax.text(
                    j,
                    i,
                    f"{labels_map[i][j]}\n{val}\n({pct:.0f}%)",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color=color,
                )

        title = name.replace("_", " ").title()
        ax.set_title(f"{title}\nF1={r['f1']:.1%}  Acc={r['acc']:.1%}", fontsize=12, fontweight="bold")

    fig.suptitle("Confusion Matrices: Top 2 Experiments", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig9_confusion_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")


def main():
    """生成全部图表"""
    print(">>> Loading experiment data...")
    data = load_data()
    print(f">>> {len(data)} experiments loaded")
    print(">>> Generating charts...")

    fig1_f1_grouped_bar(data)
    fig2_precision_recall_scatter(data)
    fig3_heatmap(data)
    fig4_p4_ablation(data)
    fig5_error_breakdown(data)
    fig6_veto_vs_weighted(data)
    fig7_cv_contribution(data)
    fig8_latency_comparison(data)
    fig9_confusion_matrices(data)

    print(f"\n>>> All charts saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
