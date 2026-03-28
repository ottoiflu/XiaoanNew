"""离线重评分工具：基于已有实验CSV中的四维标签，重新计算加权评分。"""

import csv
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRAST_DIR = PROJECT_ROOT / "outputs" / "contrast_experiments"

# opt_weighted 统一评分配置（网格搜索最优）
OPT_SCORING = {
    "weights": {"composition": 0.05, "angle": 0.20, "distance": 0.45, "context": 0.30},
    "threshold": 0.35,
    "composition_gate": True,
    "score_map": {
        "composition": {"[合规]": 1.0, "[基本合规]": 0.7, "[不合规-构图]": 0.0, "[不合规-无参照]": 0.0},
        "angle": {"[合规]": 1.0, "[不合规-角度]": 0.0},
        "distance": {"[完全合规]": 1.0, "[基本合规-压线]": 0.0, "[不合规-超界]": 0.0},
        "context": {"[合规]": 1.0, "[不合规-环境]": 0.0},
    },
}


def compute_opt_weighted(row, config=None):
    """基于四维标签计算加权分数和预测结果。"""
    if config is None:
        config = OPT_SCORING

    dims = ["composition", "angle", "distance", "context"]
    scores = {}
    for dim in dims:
        label = row.get(dim, "").strip()
        scores[dim] = config["score_map"].get(dim, {}).get(label, 0.0)

    # composition gate
    if config.get("composition_gate") and scores["composition"] == 0.0:
        return 0.0, "no"

    weighted = sum(config["weights"][d] * scores[d] for d in dims)
    pred = "yes" if weighted >= config["threshold"] else "no"
    return weighted, pred


def rescore_experiment(exp_dir, output_suffix="opt_weighted"):
    """对单个实验目录重新评分，生成新的实验目录。"""
    exp_name = exp_dir.name
    # 找到CSV文件
    csvs = list(exp_dir.glob("*.csv"))
    if not csvs:
        csvs = list(exp_dir.glob("results/*.csv"))
    if not csvs:
        print(f"  跳过 {exp_name}: 无CSV文件")
        return None

    src_csv = csvs[0]
    with open(src_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"  跳过 {exp_name}: CSV为空")
        return None

    # 重新评分
    tp = tn = fp = fn = 0
    total_lat = 0
    invalid = 0
    rescored_rows = []

    for row in rows:
        new_score, new_pred = compute_opt_weighted(row)
        gt = row.get("gt", "").strip()
        if gt not in ("yes", "no"):
            invalid += 1
            continue

        new_row = dict(row)
        new_row["pred"] = new_pred
        new_row["weighted_score"] = f"{new_score:.4f}"
        rescored_rows.append(new_row)

        if new_pred == "yes" and gt == "yes":
            tp += 1
        elif new_pred == "no" and gt == "no":
            tn += 1
        elif new_pred == "yes" and gt == "no":
            fp += 1
        else:
            fn += 1

        try:
            total_lat += float(row.get("latency", 0))
        except (ValueError, TypeError):
            pass

    total = tp + tn + fp + fn
    if total == 0:
        print(f"  跳过 {exp_name}: 无有效样本")
        return None

    acc = (tp + tn) / total
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    avg_lat = total_lat / total if total > 0 else 0

    # 生成新实验名
    base_name = exp_name.split("_", 2)[-1] if exp_name.count("_") >= 2 else exp_name
    # 去掉旧的 scoring 后缀
    for suffix in ["_veto", "_weighted", "_opt_weighted"]:
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break

    new_exp_name = f"{base_name}_{output_suffix}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir_name = f"{timestamp}_{new_exp_name}"
    new_dir = CONTRAST_DIR / new_dir_name

    # 创建目录并写入CSV
    new_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rescored_rows[0].keys()) if rescored_rows else []
    out_csv = new_dir / f"{new_exp_name}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rescored_rows)

    # 复制可视化图片（如果有）
    vis_dir = exp_dir / "vis"
    if vis_dir.exists():
        shutil.copytree(vis_dir, new_dir / "vis", dirs_exist_ok=True)

    result = {
        "exp_name": new_exp_name,
        "mode": "vlm_cv" if "cv_p" in new_exp_name else "pure_vlm",
        "prompt_id": "",
        "scoring": "opt_weighted",
        "acc": acc,
        "pre": pre,
        "rec": rec,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "invalid": invalid,
        "avg_lat": round(avg_lat, 3),
        "timestamp": timestamp,
    }

    # 推断 prompt_id
    for pid in ["p4_3", "p4_2", "p4_1", "p4", "p5", "p6", "p7", "p8"]:
        if f"_{pid}_" in f"_{new_exp_name}_" or new_exp_name.endswith(f"_{pid}"):
            if "cv_" in new_exp_name:
                result["prompt_id"] = f"cv_enhanced_{pid}"
            else:
                result["prompt_id"] = f"standard_{pid}"
            break

    print(f"  {new_exp_name}: acc={acc:.3f} pre={pre:.3f} rec={rec:.3f} f1={f1:.3f}")
    return result


def main():
    """主入口：对指定实验进行离线重评分。"""
    # 找到需要重评分的实验（使用 standard weighted 的 p4/p5/p6/p7）
    targets = []
    for d in sorted(CONTRAST_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("20260328_"):
            continue
        name = d.name
        # 只对使用 standard weighted (非opt) 的实验重评分
        if "_weighted" in name and "_opt_weighted" not in name and "_minimal" not in name:
            targets.append(d)

    # 同时也对 veto 实验重评分以获取统一的 opt_weighted 结果
    for d in sorted(CONTRAST_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("20260328_"):
            continue
        name = d.name
        if "_veto" in name and "_minimal" not in name:
            targets.append(d)

    print(f"待重评分实验: {len(targets)} 组\n")

    results = []
    seen = set()
    for t in targets:
        r = rescore_experiment(t)
        if r and r["exp_name"] not in seen:
            results.append(r)
            seen.add(r["exp_name"])

    if results:
        # 写入 summary CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = CONTRAST_DIR / f"summary_{timestamp}_rescore.csv"
        fieldnames = list(results[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n汇总已写入: {summary_path}")

        # 追加到 all_results.csv
        all_csv = CONTRAST_DIR / "all_results.csv"
        with open(all_csv, encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))
        existing_names = {r["exp_name"] for r in existing}

        all_fields = list(existing[0].keys()) if existing else fieldnames
        added = 0
        for r in results:
            if r["exp_name"] not in existing_names:
                existing.append(r)
                existing_names.add(r["exp_name"])
                added += 1

        with open(all_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(existing)
        print(f"all_results.csv: 追加 {added} 组，总计 {len(existing)} 组")


if __name__ == "__main__":
    main()
