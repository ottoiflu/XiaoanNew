"""
加权评判引擎 - 替代一票否决制的得分评判系统

提供可配置的维度权重、分数映射和阈值，支持 YAML 配置文件加载、
单条评判、批量 CSV 重评估和阈值网格搜索。

典型用法：
    engine = ScoringEngine.from_yaml("configs/scoring_default.yaml")
    result = engine.score("[合规]", "[合规]", "[基本合规-压线]", "[合规]")
    print(result.is_compliant, result.final_score)
"""

from __future__ import annotations

import csv
import copy
import itertools
from dataclasses import dataclass
from utils.metrics import BinaryMetrics
from typing import Optional


# ──────────────────────────── 数据结构 ────────────────────────────

@dataclass
class ScoringResult:
    """单条评判结果"""
    is_compliant: bool
    final_score: float
    dimension_scores: dict[str, float]
    raw_statuses: dict[str, str]
    gated: bool = False  # 构图门控是否触发


@dataclass
class ScoringConfig:
    """评判配置"""
    score_map: dict[str, dict[str, float]]
    weights: dict[str, float]
    threshold: float
    composition_gate: bool = True

    @classmethod
    def default(cls) -> ScoringConfig:
        return cls(
            score_map={
                "composition": {
                    "[合规]": 1.0,
                    "[基本合规]": 0.7,
                    "[不合规-构图]": 0.0,
                    "[不合规-无参照]": 0.0,
                },
                "angle": {
                    "[合规]": 1.0,
                    "[不合规-角度]": 0.0,
                },
                "distance": {
                    "[完全合规]": 1.0,
                    "[基本合规-压线]": 0.0,
                    "[不合规-超界]": 0.0,
                },
                "context": {
                    "[合规]": 1.0,
                    "[不合规-环境]": 0.0,
                },
            },
            weights={
                "composition": 0.05,
                "angle": 0.25,
                "distance": 0.40,
                "context": 0.30,
            },
            threshold=0.60,
            composition_gate=True,
        )

    @classmethod
    def from_yaml(cls, path: str) -> ScoringConfig:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            score_map=data["score_map"],
            weights=data["weights"],
            threshold=data["threshold"],
            composition_gate=data.get("composition_gate", True),
        )

    def to_yaml(self, path: str) -> None:
        import yaml
        data = {
            "score_map": self.score_map,
            "weights": self.weights,
            "threshold": self.threshold,
            "composition_gate": self.composition_gate,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


# ──────────────────────────── 评判引擎 ────────────────────────────

class ScoringEngine:
    """加权评判引擎"""

    DIMENSIONS = ("composition", "angle", "distance", "context")

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig.default()
        self._validate_config()

    def _validate_config(self) -> None:
        w_sum = sum(self.config.weights.values())
        if abs(w_sum - 1.0) > 0.01:
            raise ValueError(f"权重总和应为 1.0，当前为 {w_sum}")
        for dim in self.DIMENSIONS:
            if dim not in self.config.weights:
                raise ValueError(f"缺少维度权重: {dim}")
            if dim not in self.config.score_map:
                raise ValueError(f"缺少维度分数映射: {dim}")

    # ── 核心评判 ──

    def score(
        self,
        composition: str,
        angle: str,
        distance: str,
        context: str,
    ) -> ScoringResult:
        """对单条 VLM 输出进行加权评判"""
        raw = {
            "composition": composition.strip(),
            "angle": angle.strip(),
            "distance": distance.strip(),
            "context": context.strip(),
        }
        dim_scores = {}
        for dim in self.DIMENSIONS:
            status = raw[dim]
            mapping = self.config.score_map[dim]
            dim_scores[dim] = mapping.get(status, self._fuzzy_match(status, mapping))

        # 构图门控：图像质量不合格则直接否决
        if self.config.composition_gate and dim_scores["composition"] == 0.0:
            return ScoringResult(
                is_compliant=False,
                final_score=0.0,
                dimension_scores=dim_scores,
                raw_statuses=raw,
                gated=True,
            )

        final = sum(
            self.config.weights[d] * dim_scores[d] for d in self.DIMENSIONS
        )
        return ScoringResult(
            is_compliant=final >= self.config.threshold,
            final_score=round(final, 4),
            dimension_scores=dim_scores,
            raw_statuses=raw,
        )

    def judge(
        self,
        composition: str,
        angle: str,
        distance: str,
        context: str,
    ) -> str:
        """返回 'yes' 或 'no'，供脚本直接替换一票否决逻辑"""
        return "yes" if self.score(composition, angle, distance, context).is_compliant else "no"

    # ── 一票否决（保留兼容） ──

    @staticmethod
    def veto_judge(
        composition: str,
        angle: str,
        distance: str,
        context: str,
    ) -> str:
        """原始一票否决逻辑"""
        if "不合规" in composition:
            return "no"
        if "不合规" in angle:
            return "no"
        if "不合规" in context:
            return "no"
        if "超界" in distance:
            return "no"
        return "yes"

    # ── 批量评估 ──

    def batch_evaluate(
        self,
        csv_path: str,
        gt_col: str = "ground_truth",
        comp_col: str = "composition",
        angle_col: str = "angle",
        dist_col: str = "distance",
        ctx_col: str = "context",
    ) -> dict:
        """从已有 CSV 结果文件批量重评估，返回指标字典"""
        rows = self._load_csv(csv_path)
        tp = tn = fp = fn = 0
        for row in rows:
            gt = row[gt_col].strip().lower()
            if gt in ("yes", "合规"):
                gt = "yes"
            else:
                gt = "no"
            pred = self.judge(row[comp_col], row[angle_col], row[dist_col], row[ctx_col])
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
        return self._calc_metrics(tp, tn, fp, fn)

    # ── 阈值扫描 ──

    def sweep_threshold(
        self,
        csv_path: str,
        start: float = 0.0,
        stop: float = 1.01,
        step: float = 0.05,
        gt_col: str = "ground_truth",
        comp_col: str = "composition",
        angle_col: str = "angle",
        dist_col: str = "distance",
        ctx_col: str = "context",
    ) -> list[dict]:
        """遍历阈值区间，返回每个阈值下的指标"""
        rows = self._load_csv(csv_path)

        # 预计算所有样本的 final_score 和 gt
        scored = []
        for row in rows:
            gt = row[gt_col].strip().lower()
            gt = "yes" if gt in ("yes", "合规") else "no"
            result = self.score(row[comp_col], row[angle_col], row[dist_col], row[ctx_col])
            scored.append((gt, result.final_score, result.gated))

        results = []
        threshold = start
        while threshold <= stop:
            tp = tn = fp = fn = 0
            for gt, fs, gated in scored:
                if gated:
                    pred = "no"
                else:
                    pred = "yes" if fs >= threshold else "no"
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
            metrics = self._calc_metrics(tp, tn, fp, fn)
            metrics["threshold"] = round(threshold, 4)
            results.append(metrics)
            threshold += step
        return results

    # ── 权重网格搜索 ──

    def grid_search(
        self,
        csv_path: str,
        weight_grid: Optional[dict] = None,
        threshold_range: tuple = (0.3, 0.95, 0.05),
        optimize: str = "f1",
        gt_col: str = "ground_truth",
        comp_col: str = "composition",
        angle_col: str = "angle",
        dist_col: str = "distance",
        ctx_col: str = "context",
    ) -> dict:
        """网格搜索最优权重和阈值组合

        Args:
            weight_grid: 各维度候选权重列表，如
                {"angle": [0.3, 0.35, 0.4], "distance": [0.3, 0.35, 0.4]}
                未指定的维度使用剩余权重均分
            threshold_range: (start, stop, step)
            optimize: 优化目标 ("f1", "acc", "pre", "rec")
        Returns:
            最优参数字典，含 weights / threshold / metrics
        """
        rows = self._load_csv(csv_path)

        if weight_grid is None:
            weight_grid = {
                "composition": [0.05, 0.10, 0.15],
                "angle": [0.25, 0.30, 0.35, 0.40],
                "distance": [0.25, 0.30, 0.35, 0.40],
                "context": [0.15, 0.20, 0.25],
            }

        # 生成所有权重组合（权重总和必须为 1.0）
        dims = list(weight_grid.keys())
        combos = list(itertools.product(*[weight_grid[d] for d in dims]))
        valid_combos = [(c, dict(zip(dims, c))) for c in combos if abs(sum(c) - 1.0) < 0.01]

        t_start, t_stop, t_step = threshold_range
        thresholds = []
        t = t_start
        while t <= t_stop:
            thresholds.append(round(t, 4))
            t += t_step

        best = {"metric": -1}
        for _, weights in valid_combos:
            test_config = copy.deepcopy(self.config)
            test_config.weights = weights
            test_engine = ScoringEngine(test_config)

            scored = []
            for row in rows:
                gt = row[gt_col].strip().lower()
                gt = "yes" if gt in ("yes", "合规") else "no"
                result = test_engine.score(
                    row[comp_col], row[angle_col], row[dist_col], row[ctx_col]
                )
                scored.append((gt, result.final_score, result.gated))

            for threshold in thresholds:
                tp = tn = fp = fn = 0
                for gt, fs, gated in scored:
                    pred = "no" if gated else ("yes" if fs >= threshold else "no")
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

                metrics = self._calc_metrics(tp, tn, fp, fn)
                val = metrics[optimize]
                if val > best["metric"]:
                    best = {
                        "metric": val,
                        "optimize": optimize,
                        "weights": dict(weights),
                        "threshold": threshold,
                        "metrics": metrics,
                    }

        return best

    # ── 工具方法 ──

    @staticmethod
    def _fuzzy_match(status: str, mapping: dict) -> float:
        """模糊匹配状态标签，容忍格式差异"""
        s = status.strip().replace("（", "(").replace("）", ")")
        for key, val in mapping.items():
            k = key.strip().replace("（", "(").replace("）", ")")
            if s == k or s in k or k in s:
                return val
        if "不合规" in s:
            return 0.0
        if "基本" in s:
            return 0.5
        if "合规" in s:
            return 1.0
        return 0.0

    @staticmethod
    def _load_csv(csv_path: str) -> list[dict]:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _calc_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
        return BinaryMetrics.from_confusion_matrix(tp, tn, fp, fn).to_dict()

    # ── 工厂方法 ──

    @classmethod
    def from_yaml(cls, path: str) -> ScoringEngine:
        return cls(ScoringConfig.from_yaml(path))


# ──────────────────────────── CLI 入口 ────────────────────────────

def main():
    """命令行批量重评估和阈值搜索"""
    import argparse

    parser = argparse.ArgumentParser(description="加权评判引擎 - 批量重评估工具")
    sub = parser.add_subparsers(dest="command")

    # 重评估子命令
    eval_p = sub.add_parser("evaluate", help="用加权评判重新评估已有 CSV 结果")
    eval_p.add_argument("csv", help="结果 CSV 文件路径")
    eval_p.add_argument("-c", "--config", help="评判配置 YAML 路径")
    eval_p.add_argument("-t", "--threshold", type=float, help="覆盖默认阈值")

    # 阈值扫描子命令
    sweep_p = sub.add_parser("sweep", help="扫描阈值区间寻找最优点")
    sweep_p.add_argument("csv", help="结果 CSV 文件路径")
    sweep_p.add_argument("-c", "--config", help="评判配置 YAML 路径")

    # 网格搜索子命令
    grid_p = sub.add_parser("grid", help="网格搜索最优权重和阈值")
    grid_p.add_argument("csv", help="结果 CSV 文件路径")
    grid_p.add_argument("-c", "--config", help="评判配置 YAML 路径")
    grid_p.add_argument("-o", "--optimize", default="f1", choices=["f1", "acc", "pre", "rec"])

    args = parser.parse_args()

    if args.command == "evaluate":
        engine = ScoringEngine.from_yaml(args.config) if args.config else ScoringEngine()
        if args.threshold is not None:
            engine.config.threshold = args.threshold
        metrics = engine.batch_evaluate(args.csv)
        _print_metrics(metrics, engine.config.threshold)

    elif args.command == "sweep":
        engine = ScoringEngine.from_yaml(args.config) if args.config else ScoringEngine()
        results = engine.sweep_threshold(args.csv)
        print(f"\n{'阈值':>6} | {'F1':>6} | {'Acc':>6} | {'Pre':>6} | {'Rec':>6} | {'FP':>4} | {'FN':>4}")
        print("-" * 52)
        for r in results:
            print(f"{r['threshold']:6.2f} | {r['f1']:6.4f} | {r['acc']:6.4f} | {r['pre']:6.4f} | {r['rec']:6.4f} | {r['fp']:4d} | {r['fn']:4d}")
        best = max(results, key=lambda x: x["f1"])
        print(f"\n最优阈值: {best['threshold']:.2f} -> F1={best['f1']:.4f}, Acc={best['acc']:.4f}")

    elif args.command == "grid":
        engine = ScoringEngine.from_yaml(args.config) if args.config else ScoringEngine()
        best = engine.grid_search(args.csv, optimize=args.optimize)
        print(f"\n网格搜索完成 (优化目标: {best['optimize']})")
        print(f"最优权重: {best['weights']}")
        print(f"最优阈值: {best['threshold']}")
        _print_metrics(best["metrics"], best["threshold"])
    else:
        parser.print_help()


def _print_metrics(metrics: dict, threshold: float) -> None:
    print(f"\n{'='*20} 加权评判结果 (阈值={threshold}) {'='*20}")
    print(f"准确率: {metrics['acc']:.2%}  精确率: {metrics['pre']:.2%}")
    print(f"召回率: {metrics['rec']:.2%}  F1: {metrics['f1']:.4f}")
    print(f"TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
