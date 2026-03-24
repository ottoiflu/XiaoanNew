"""进程内 CLI 入口覆盖测试 — 直接调用 main() 或 runpy 以计入覆盖率"""

import csv
import runpy
from unittest.mock import patch

from modules.experiment.scoring import ScoringEngine
from modules.experiment.scoring import main as scoring_main


def _csv(tmp_path, name="data.csv"):
    """构建测试 CSV，确保覆盖 fn / fp 分支"""
    p = str(tmp_path / name)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth", "composition", "angle", "distance", "context"])
        # TP: yes -> compliant
        w.writerow(["a.jpg", "yes", "[合规]", "[合规]", "[完全合规]", "[合规]"])
        # FN: yes -> predicted no (line 223: fn += 1)
        w.writerow(["b.jpg", "yes", "[不合规-构图]", "[不合规-角度]", "[不合规-超界]", "[不合规-环境]"])
        # TN: no -> predicted no
        w.writerow(["c.jpg", "no", "[不合规-构图]", "[合规]", "[完全合规]", "[合规]"])
        # FP: no -> predicted yes (line 228: fp += 1)
        w.writerow(["d.jpg", "no", "[合规]", "[合规]", "[完全合规]", "[合规]"])
    return p


class TestScoringMainInProcess:
    """直接调用 scoring.main()，覆盖 lines 408-471"""

    def test_evaluate(self, tmp_path, capsys):
        csv_path = _csv(tmp_path)
        with patch("sys.argv", ["scoring", "evaluate", csv_path]):
            scoring_main()
        out = capsys.readouterr().out
        assert "加权评判结果" in out

    def test_evaluate_with_threshold(self, tmp_path, capsys):
        csv_path = _csv(tmp_path)
        from modules.experiment.scoring import ScoringConfig

        cfg_path = str(tmp_path / "sc.yaml")
        ScoringConfig.default().to_yaml(cfg_path)
        with patch("sys.argv", ["scoring", "evaluate", csv_path, "-c", cfg_path, "-t", "0.3"]):
            scoring_main()
        out = capsys.readouterr().out
        assert "加权评判结果" in out

    def test_sweep(self, tmp_path, capsys):
        csv_path = _csv(tmp_path)
        with patch("sys.argv", ["scoring", "sweep", csv_path]):
            scoring_main()
        out = capsys.readouterr().out
        assert "最优阈值" in out

    def test_grid(self, tmp_path, capsys):
        csv_path = _csv(tmp_path)
        with patch("sys.argv", ["scoring", "grid", csv_path, "-o", "f1"]):
            scoring_main()
        out = capsys.readouterr().out
        assert "网格搜索完成" in out

    def test_no_command(self, tmp_path, capsys):
        with patch("sys.argv", ["scoring"]):
            scoring_main()
        capsys.readouterr()  # 无子命令时打印帮助


class TestScoringBatchFnFp:
    """确保 fn / fp 分支被覆盖 (lines 223, 228)"""

    def test_fn_branch(self, tmp_path):
        """ground_truth=yes 但全维度不合规 -> fn"""
        p = str(tmp_path / "fn.csv")
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["image", "ground_truth", "composition", "angle", "distance", "context"])
            w.writerow(["x.jpg", "yes", "[不合规-构图]", "[不合规-角度]", "[不合规-超界]", "[不合规-环境]"])
        m = ScoringEngine().batch_evaluate(p)
        assert m["fn"] == 1 and m["tp"] == 0

    def test_fp_branch(self, tmp_path):
        """ground_truth=no 但全维度合规 -> fp"""
        p = str(tmp_path / "fp.csv")
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["image", "ground_truth", "composition", "angle", "distance", "context"])
            w.writerow(["x.jpg", "no", "[合规]", "[合规]", "[完全合规]", "[合规]"])
        m = ScoringEngine().batch_evaluate(p)
        assert m["fp"] == 1 and m["tn"] == 0


class TestConfigRunpy:
    """用 runpy 覆盖 config.py __main__ block (lines 169-201)"""

    def test_list(self, capsys):
        with patch("sys.argv", ["config", "list"]):
            runpy.run_module("modules.experiment.config", run_name="__main__", alter_sys=True)
        assert "可用的实验配置" in capsys.readouterr().out

    def test_show(self, capsys):
        with patch("sys.argv", ["config", "show", "default"]):
            runpy.run_module("modules.experiment.config", run_name="__main__", alter_sys=True)
        assert "exp_name" in capsys.readouterr().out

    def test_create(self, capsys):
        with patch("sys.argv", ["config", "create"]):
            runpy.run_module("modules.experiment.config", run_name="__main__", alter_sys=True)
        assert "模板" in capsys.readouterr().out


class TestPromptRunpy:
    """用 runpy 覆盖 prompt/manager.py __main__ block (lines 137-155)"""

    def test_list(self, capsys):
        with patch("sys.argv", ["manager", "list"]):
            runpy.run_module("modules.prompt.manager", run_name="__main__", alter_sys=True)

    def test_show(self, capsys):
        with patch("sys.argv", ["manager", "show", "cv_enhanced_p4"]):
            runpy.run_module("modules.prompt.manager", run_name="__main__", alter_sys=True)
        assert len(capsys.readouterr().out) > 0

    def test_info(self, capsys):
        with patch("sys.argv", ["manager", "info", "cv_enhanced_p4"]):
            runpy.run_module("modules.prompt.manager", run_name="__main__", alter_sys=True)
        assert "version" in capsys.readouterr().out


class TestMetricsRunpy:
    """用 runpy 覆盖 metrics.py __main__ block (lines 318-322)"""

    def test_main(self, capsys):
        runpy.run_module("modules.experiment.metrics", run_name="__main__", alter_sys=True)
        assert "测试报告" in capsys.readouterr().out
