"""
加权评判引擎 - 新四维版 (position/medium/angle/state)

变更：
- 维度从 composition/angle/distance/context 换为 position/medium/angle/state
- 删除构图门控 (composition_gate)
- angle=[N/A] 时不参与加权，剩余维度权重归一化重算
"""

from __future__ import annotations

import copy
import csv
import itertools
from dataclasses import dataclass
from typing import Optional

from modules.experiment.metrics import BinaryMetrics

# ──────────────────────────── 数据结构 ────────────────────────────


@dataclass
class ScoringResult:
    """单条评判结果"""

    is_compliant: bool
    final_score: float
    dimension_scores: dict[str, float]
    raw_statuses: dict[str, str]


@dataclass
class ScoringConfig:
    """评判配置"""

    score_map: dict[str, dict[str, float]]
    weights: dict[str, float]
    threshold: float

    @classmethod
    def default(cls) -> ScoringConfig:
        return cls(
            score_map={
                "position": {
                    "[合规]": 1.0,
                    "[基本合规-压线]": 0.6,
                    "[不合规-超界]": 0.0,
                    "[无参照]": 0.5,
                },
                "medium": {
                    "[合规]": 1.0,
                    "[不合规-盲道]": 0.0,
                    "[不合规-绿化]": 0.0,
                    "[不合规-禁停区]": 0.0,
                },
                "angle": {
                    "[合规]": 1.0,
                    "[不合规-斜停]": 0.0,
                    "[N/A]": None,
                },
                "state": {
                    "[正立]": 1.0,
                    "[倒伏]": 0.0,
                },
            },
            weights={
                "position": 0.30,
                "medium": 0.35,
                "angle": 0.20,
                "state": 0.15,
            },
            threshold=0.45,
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
        )

    def to_yaml(self, path: str) -> None:
        import yaml

        data = {
            "score_map": self.score_map,
            "weights": self.weights,
            "threshold": self.threshold,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


# ──────────────────────────── 评判引擎 ────────────────────────────


class ScoringEngine:
    """加权评判引擎 (新四维)"""

    DIMENSIONS = ("position", "medium", "angle", "state")

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
        position: str,
        medium: str,
        angle: str,
        state: str,
    ) -> ScoringResult:
        """对新四维 VLM 输出进行加权评判

        angle=[N/A] 时该维度退出，剩余维度权重归一化重算。
        score_map 值支持 float 或 [low, high] 区间（用置信度中值插值）。
        """
        return self._score_with_conf(position, medium, angle, state)

    def score_interp(
        self,
        position: str,
        medium: str,
        angle: str,
        state: str,
        position_conf: float = 0.5,
        medium_conf: float = 0.5,
        angle_conf: float = 0.5,
        state_conf: float = 0.5,
    ) -> ScoringResult:
        """带置信度的区间插值评分

        score_map 中标签的值可以是 [low, high] 区间列表，
        最终分数 = low + conf * (high - low)。
        兼容旧版 float 值（直接使用）。
        """
        confs = {
            "position": position_conf,
            "medium": medium_conf,
            "angle": angle_conf,
            "state": state_conf,
        }
        return self._score_with_conf(position, medium, angle, state, confs)

    def _score_with_conf(
        self,
        position: str,
        medium: str,
        angle: str,
        state: str,
        confs: Optional[dict[str, float]] = None,
    ) -> ScoringResult:
        raw = {
            "position": position.strip(),
            "medium": medium.strip(),
            "angle": angle.strip(),
            "state": state.strip(),
        }

        # 计算各维度分数，None 表示跳过
        dim_scores = {}
        active_weights = {}
        for dim in self.DIMENSIONS:
            status = raw[dim]
            mapping = self.config.score_map[dim]
            s = mapping.get(status, self._fuzzy_match(status, mapping))
            if s is None:
                continue  # angle=[N/A] 跳过
            # 区间映射: [low, high] 用置信度中值插值，纯 float 直接使用
            if isinstance(s, (list, tuple)):
                low, high = s[0], s[1]
                conf = confs.get(dim, 0.5) if confs else 0.5
                s = low + conf * (high - low)
            dim_scores[dim] = s
            active_weights[dim] = self.config.weights[dim]

        # 归一化：active_weights 重新加权到总和 1.0
        w_sum = sum(active_weights.values())
        if w_sum == 0:
            return ScoringResult(
                is_compliant=False,
                final_score=0.0,
                dimension_scores=dim_scores,
                raw_statuses=raw,
            )

        final = sum(dim_scores[d] * active_weights[d] / w_sum for d in dim_scores)
        return ScoringResult(
            is_compliant=final >= self.config.threshold,
            final_score=round(final, 4),
            dimension_scores=dim_scores,
            raw_statuses=raw,
        )

    def judge(
        self,
        position: str,
        medium: str,
        angle: str,
        state: str,
    ) -> str:
        """返回 'yes' 或 'no'"""
        return "yes" if self.score(position, medium, angle, state).is_compliant else "no"

    # ── 一票否决（保留兼容） ──

    @staticmethod
    def veto_judge(
        composition: str,
        angle: str,
        distance: str,
        context: str,
    ) -> str:
        """原始一票否决逻辑（旧四维兼容，仅供旧脚本使用）"""
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
        pos_col: str = "position",
        med_col: str = "medium",
        ang_col: str = "angle",
        state_col: str = "state",
    ) -> dict:
        """从已有 CSV 结果文件批量重评估"""
        rows = self._load_csv(csv_path)
        tp = tn = fp = fn = 0
        for row in rows:
            gt = row[gt_col].strip().lower()
            gt = "yes" if gt in ("yes", "合规") else "no"
            pred = self.judge(row[pos_col], row[med_col], row[ang_col], row[state_col])
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
        pos_col: str = "position",
        med_col: str = "medium",
        ang_col: str = "angle",
        state_col: str = "state",
    ) -> list[dict]:
        """遍历阈值区间"""
        rows = self._load_csv(csv_path)
        scored = []
        for row in rows:
            gt = row[gt_col].strip().lower()
            gt = "yes" if gt in ("yes", "合规") else "no"
            result = self.score(row[pos_col], row[med_col], row[ang_col], row[state_col])
            scored.append((gt, result.final_score))

        results = []
        threshold = start
        while threshold <= stop:
            tp = tn = fp = fn = 0
            for gt, fs in scored:
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
        pos_col: str = "position",
        med_col: str = "medium",
        ang_col: str = "angle",
        state_col: str = "state",
    ) -> dict:
        """网格搜索最优权重和阈值"""
        rows = self._load_csv(csv_path)
        if weight_grid is None:
            weight_grid = {
                "position": [0.20, 0.30, 0.40, 0.50],
                "medium": [0.25, 0.35, 0.45],
                "angle": [0.10, 0.20, 0.30],
                "state": [0.05, 0.10, 0.15],
            }
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
                result = test_engine.score(row[pos_col], row[med_col], row[ang_col], row[state_col])
                scored.append((gt, result.final_score))

            for threshold in thresholds:
                tp = tn = fp = fn = 0
                for gt, fs in scored:
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

    @classmethod
    def from_yaml(cls, path: str) -> ScoringEngine:
        return cls(ScoringConfig.from_yaml(path))


# ──────────────────────────── CLI 入口 ────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="加权评判引擎 - 新四维版")
    sub = parser.add_subparsers(dest="command")

    eval_p = sub.add_parser("evaluate", help="用加权评判重新评估已有 CSV 结果")
    eval_p.add_argument("csv", help="结果 CSV 文件路径")
    eval_p.add_argument("-c", "--config", help="评判配置 YAML 路径")
    eval_p.add_argument("-t", "--threshold", type=float, help="覆盖默认阈值")

    sweep_p = sub.add_parser("sweep", help="扫描阈值区间寻找最优点")
    sweep_p.add_argument("csv", help="结果 CSV 文件路径")
    sweep_p.add_argument("-c", "--config", help="评判配置 YAML 路径")

    grid_p = sub.add_parser("grid", help="网格搜索最优权重和阈值")
    grid_p.add_argument("csv", help="结果 CSV 文件路径")
    grid_p.add_argument("-c", "--config", help="评判配置 YAML 路径")
    grid_p.add_argument("-o", "--optimize", default="f1", choices=["f1", "acc", "pre", "rec"])

    args = parser.parse_args()

    def _print_metrics(metrics: dict, threshold: float) -> None:
        print(f"\n{'=' * 20} 加权评判结果 (阈值={threshold}) {'=' * 20}")
        print(f"准确率: {metrics['acc']:.2%}  精确率: {metrics['pre']:.2%}")
        print(f"召回率: {metrics['rec']:.2%}  F1: {metrics['f1']:.4f}")
        print(f"TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
        print("=" * 55)

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


if __name__ == "__main__":
    main()