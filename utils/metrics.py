"""
评估指标计算模块

功能：
1. 计算分类指标 (Accuracy, Precision, Recall, F1)
2. 生成混淆矩阵
3. 格式化输出评估报告
4. 维护实验排行榜 (Top 20)

使用方式:
    from utils.metrics import calculate_metrics, print_metrics_report, update_leaderboard
    
    metrics = calculate_metrics(predictions, ground_truths)
    print_metrics_report(metrics)
    update_leaderboard("/path/to/test_outputs")
"""

import csv
import glob
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class BinaryMetrics:
    """二分类评估指标"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    tp: int  # True Positive
    tn: int  # True Negative
    fp: int  # False Positive
    fn: int  # False Negative
    total: int
    invalid: int
    avg_latency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "acc": self.accuracy,
            "pre": self.precision,
            "rec": self.recall,
            "f1": self.f1_score,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "total": self.total,
            "invalid": self.invalid,
            "avg_lat": self.avg_latency,
        }


def normalize_label(label: str) -> str:
    """
    标准化标签值
    
    Args:
        label: 原始标签字符串
        
    Returns:
        标准化后的 "yes" 或 "no"
    """
    if not label:
        return "unknown"
    
    label = str(label).strip().lower()
    
    # 正样本关键词
    positive_keywords = ["yes", "合规", "是", "true", "1", "positive", "正"]
    # 负样本关键词
    negative_keywords = ["no", "违规", "否", "false", "0", "negative", "负"]
    
    for kw in positive_keywords:
        if kw in label:
            return "yes"
    
    for kw in negative_keywords:
        if kw in label:
            return "no"
    
    return "unknown"


def calculate_metrics(
    predictions: List[str],
    ground_truths: List[str],
    latencies: List[float] = None
) -> BinaryMetrics:
    """
    计算二分类评估指标
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标签列表
        latencies: 延迟时间列表（可选）
        
    Returns:
        BinaryMetrics 对象
    """
    tp, tn, fp, fn, invalid = 0, 0, 0, 0, 0
    valid_latencies = []
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_norm = normalize_label(pred)
        gt_norm = normalize_label(gt)
        
        # 处理无效预测
        if pred_norm == "unknown" or pred == "error":
            invalid += 1
            continue
        
        # 计算混淆矩阵
        if gt_norm == "yes":
            if pred_norm == "yes":
                tp += 1
            else:
                fn += 1
        elif gt_norm == "no":
            if pred_norm == "no":
                tn += 1
            else:
                fp += 1
        
        # 收集延迟
        if latencies and i < len(latencies) and latencies[i] > 0:
            valid_latencies.append(latencies[i])
    
    # 计算指标
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_lat = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0
    
    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        total=total,
        invalid=invalid,
        avg_latency=round(avg_lat, 3),
    )


def print_metrics_report(
    metrics: BinaryMetrics,
    title: str = "评估报告",
    show_confusion_matrix: bool = True
) -> None:
    """
    打印格式化的评估报告
    
    Args:
        metrics: BinaryMetrics 对象
        title: 报告标题
        show_confusion_matrix: 是否显示混淆矩阵
    """
    print(f"\n{'='*20} {title} {'='*20}")
    print(f"总样本数 (Total): {metrics.total}")
    print(f"无效预测 (Invalid): {metrics.invalid}")
    print("-" * 60)
    print(f"准确率 (Accuracy) : {metrics.accuracy:.2%}")
    print(f"精确率 (Precision): {metrics.precision:.2%}")
    print(f"召回率 (Recall)   : {metrics.recall:.2%}")
    print(f"F1分数 (F1-Score) : {metrics.f1_score:.2f}")
    
    if show_confusion_matrix:
        print("-" * 60)
        print("混淆矩阵:")
        print(f"  [TP] 正确预测合规: {metrics.tp}")
        print(f"  [TN] 正确预测违规: {metrics.tn}")
        print(f"  [FP] 误判为合规: {metrics.fp}")
        print(f"  [FN] 误判为违规: {metrics.fn}")
    
    if metrics.avg_latency > 0:
        print(f"平均延迟: {metrics.avg_latency}s")
    
    print("=" * 60)



# ================= 排行榜管理 =================

LEADERBOARD_FILENAME = "leaderboard_top20.csv"
LEADERBOARD_FIELDS = [
    "rank", "f1", "acc", "pre", "rec",
    "tp", "tn", "fp", "fn", "total", "invalid",
    "avg_lat", "exp_name", "segmentor", "folders", "timestamp"
]


def _collect_all_summaries(test_outputs_dir: str) -> List[Dict[str, Any]]:
    """
    从 test_outputs 目录下收集所有实验汇总记录

    扫描范围:
    1. archived_experiments/all_experiments_summary.csv
    2. 每个 exp_* 目录下的 all_experiments_summary.csv
    """
    records = []
    seen = set()

    summary_paths = []
    archived = os.path.join(test_outputs_dir, "archived_experiments", "all_experiments_summary.csv")
    if os.path.exists(archived):
        summary_paths.append(archived)

    for pattern in ["exp_*/all_experiments_summary.csv"]:
        summary_paths.extend(glob.glob(os.path.join(test_outputs_dir, pattern)))

    for path in summary_paths:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    f1_val = float(row.get("f1", 0))
                    # 跳过无效实验（f1=0 且 total=0）
                    total = int(row.get("total", 0))
                    if f1_val == 0 and total == 0:
                        continue

                    # 去重键：实验名 + 时间戳
                    dedup_key = (row.get("exp_name", ""), row.get("timestamp", ""))
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    records.append({
                        "f1": f1_val,
                        "acc": float(row.get("acc", 0)),
                        "pre": float(row.get("pre", 0)),
                        "rec": float(row.get("rec", 0)),
                        "tp": int(row.get("tp", 0)),
                        "tn": int(row.get("tn", 0)),
                        "fp": int(row.get("fp", 0)),
                        "fn": int(row.get("fn", 0)),
                        "total": total,
                        "invalid": int(row.get("invalid", 0)),
                        "avg_lat": float(row.get("avg_lat", 0)),
                        "exp_name": row.get("exp_name", ""),
                        "segmentor": row.get("segmentor", ""),
                        "folders": row.get("folders", ""),
                        "timestamp": row.get("timestamp", ""),
                    })
        except Exception as e:
            print(f"  [警告] 读取 {path} 失败: {e}")

    return records


def update_leaderboard(test_outputs_dir: str, top_n: int = 20) -> str:
    """
    更新实验排行榜，保留 F1 最高的前 N 条记录

    排序规则: F1 降序 > Accuracy 降序

    Args:
        test_outputs_dir: test_outputs 目录的绝对路径
        top_n: 保留的记录数，默认 20

    Returns:
        排行榜文件路径
    """
    records = _collect_all_summaries(test_outputs_dir)
    if not records:
        print("  [排行榜] 未找到任何实验记录")
        return ""

    # 按 F1 降序，Accuracy 降序排序
    records.sort(key=lambda r: (r["f1"], r["acc"]), reverse=True)
    top_records = records[:top_n]

    # 添加排名
    for i, rec in enumerate(top_records, 1):
        rec["rank"] = i

    # 写入 CSV
    leaderboard_path = os.path.join(test_outputs_dir, LEADERBOARD_FILENAME)
    with open(leaderboard_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        writer.writerows(top_records)

    # 打印排行榜
    _print_leaderboard(top_records)

    return leaderboard_path


def _print_leaderboard(records: List[Dict[str, Any]]) -> None:
    """格式化打印排行榜"""
    print(f"\n{'='*80}")
    print(f"{'实验排行榜 (Top ' + str(len(records)) + ')':^80}")
    print(f"{'='*80}")
    header = f"{'#':>3} | {'F1':>6} | {'Acc':>6} | {'Pre':>6} | {'Rec':>6} | {'Total':>5} | {'Lat':>5} | {'实验名称'}"
    print(header)
    print("-" * 80)
    for r in records:
        print(
            f"{r['rank']:>3} | {r['f1']:>6.4f} | {r['acc']:>6.4f} | "
            f"{r['pre']:>6.4f} | {r['rec']:>6.4f} | {r['total']:>5} | "
            f"{r['avg_lat']:>5.1f} | {r['exp_name']}"
        )
    print(f"{'='*80}")


if __name__ == "__main__":
    # 测试
    preds = ["yes", "no", "yes", "no", "yes"]
    gts = ["yes", "no", "no", "no", "yes"]
    
    metrics = calculate_metrics(preds, gts)
    print_metrics_report(metrics, "测试报告")
