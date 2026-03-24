"""modules.experiment.metrics 单元测试"""

import csv
import os

from modules.experiment.metrics import (
    BinaryMetrics,
    calculate_metrics,
    normalize_label,
    print_metrics_report,
    update_leaderboard,
)


class TestBinaryMetrics:
    def test_perfect(self):
        m = BinaryMetrics.from_confusion_matrix(50, 50, 0, 0)
        assert m.accuracy == 1.0 and m.f1_score == 1.0

    def test_all_wrong(self):
        m = BinaryMetrics.from_confusion_matrix(0, 0, 50, 50)
        assert m.accuracy == 0.0 and m.f1_score == 0.0

    def test_zero(self):
        m = BinaryMetrics.from_confusion_matrix(0, 0, 0, 0)
        assert m.total == 0

    def test_to_dict(self):
        d = BinaryMetrics.from_confusion_matrix(10, 20, 3, 7).to_dict()
        assert "acc" in d and d["tp"] == 10

    def test_single_sample(self):
        m = BinaryMetrics.from_confusion_matrix(1, 0, 0, 0)
        assert m.accuracy == 1.0 and m.total == 1


class TestNormalizeLabelMetrics:
    def test_yes(self):
        assert normalize_label("yes") == "yes"

    def test_no(self):
        assert normalize_label("no") == "no"

    def test_unknown(self):
        assert normalize_label("garbage") == "unknown"

    def test_empty(self):
        assert normalize_label("") == "unknown"


class TestCalculateMetrics:
    def test_perfect(self):
        m = calculate_metrics(["yes"] * 50 + ["no"] * 50, ["yes"] * 50 + ["no"] * 50)
        assert m.accuracy == 1.0

    def test_all_wrong(self):
        m = calculate_metrics(["no"] * 50 + ["yes"] * 50, ["yes"] * 50 + ["no"] * 50)
        assert m.accuracy == 0.0

    def test_invalid_predictions(self):
        # "error" is the only truly invalid prediction
        # "unknown_label" contains "no" and normalizes to "no"
        m = calculate_metrics(["yes", "error", "xyz_label", "no"], ["yes", "yes", "no", "no"])
        assert m.invalid == 2 and m.total == 2

    def test_empty(self):
        m = calculate_metrics([], [])
        assert m.total == 0

    def test_latencies(self):
        m = calculate_metrics(["yes", "no"], ["yes", "no"], [1.5, 2.5])
        assert m.avg_latency == 2.0

    def test_chinese_labels(self):
        m = calculate_metrics(["合规", "违规"], ["yes", "no"])
        assert m.tp == 1 and m.tn == 1


class TestPrintMetricsReport:
    def test_no_crash(self, capsys):
        print_metrics_report(BinaryMetrics.from_confusion_matrix(5, 5, 2, 3))
        assert "准确率" in capsys.readouterr().out

    def test_with_latency(self, capsys):
        m = BinaryMetrics.from_confusion_matrix(1, 1, 0, 0)
        m.avg_latency = 2.5
        print_metrics_report(m)
        assert "2.5" in capsys.readouterr().out


class TestUpdateLeaderboard:
    def _create_exp(self, base, name, f1):
        d = os.path.join(base, f"exp_{name}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "all_experiments_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "f1",
                    "acc",
                    "pre",
                    "rec",
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                    "total",
                    "invalid",
                    "avg_lat",
                    "exp_name",
                    "segmentor",
                    "folders",
                    "timestamp",
                ],
            )
            w.writeheader()
            w.writerow(
                {
                    "f1": f1,
                    "acc": 0.5,
                    "pre": 0.5,
                    "rec": 0.5,
                    "tp": 25,
                    "tn": 25,
                    "fp": 25,
                    "fn": 25,
                    "total": 100,
                    "invalid": 0,
                    "avg_lat": 1.0,
                    "exp_name": name,
                    "segmentor": "t",
                    "folders": "t",
                    "timestamp": "T",
                }
            )

    def test_creates_leaderboard(self, tmp_path):
        self._create_exp(str(tmp_path), "e1", 0.8)
        update_leaderboard(str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), "leaderboard_top20.csv"))

    def test_ranking_order(self, tmp_path):
        self._create_exp(str(tmp_path), "low", 0.5)
        self._create_exp(str(tmp_path), "high", 0.9)
        update_leaderboard(str(tmp_path))
        with open(os.path.join(str(tmp_path), "leaderboard_top20.csv"), "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["f1"]) >= float(rows[1]["f1"])

    def test_empty_dir(self, tmp_path):
        update_leaderboard(str(tmp_path))
