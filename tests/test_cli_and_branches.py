"""CLI 入口 + 剩余分支覆盖测试"""

import csv
import os
import subprocess
import sys

import pytest

PYTHON = sys.executable


# ━━━━━━━━━━━━━━━━ scoring.py CLI ━━━━━━━━━━━━━━━━


def _make_scored_csv(tmp_path, rows=None):
    """构造带完整列的结果 CSV"""
    p = str(tmp_path / "data.csv")
    if rows is None:
        rows = [
            ["a.jpg", "yes", "[合规]", "[合规]", "[完全合规]", "[合规]"],
            ["b.jpg", "no", "[不合规-构图]", "[合规]", "[完全合规]", "[合规]"],
            ["c.jpg", "yes", "[合规]", "[不合规-角度]", "[完全合规]", "[合规]"],
            ["d.jpg", "no", "[合规]", "[合规]", "[不合规-超界]", "[不合规-环境]"],
        ]
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth", "composition", "angle", "distance", "context"])
        for row in rows:
            w.writerow(row)
    return p


class TestScoringCLI:
    def test_evaluate(self, tmp_path):
        csv_path = _make_scored_csv(tmp_path)
        r = subprocess.run(
            [PYTHON, "-m", "modules.experiment.scoring", "evaluate", csv_path],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0
        assert "加权评判结果" in r.stdout

    def test_evaluate_with_config(self, tmp_path):
        csv_path = _make_scored_csv(tmp_path)
        from modules.experiment.scoring import ScoringConfig

        cfg = ScoringConfig.default()
        cfg_path = str(tmp_path / "sc.yaml")
        cfg.to_yaml(cfg_path)
        r = subprocess.run(
            [PYTHON, "-m", "modules.experiment.scoring", "evaluate", csv_path, "-c", cfg_path, "-t", "0.5"],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0
        assert "加权评判结果" in r.stdout

    def test_sweep(self, tmp_path):
        csv_path = _make_scored_csv(tmp_path)
        r = subprocess.run(
            [PYTHON, "-m", "modules.experiment.scoring", "sweep", csv_path],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0
        assert "最优阈值" in r.stdout

    def test_grid(self, tmp_path):
        csv_path = _make_scored_csv(tmp_path)
        r = subprocess.run(
            [PYTHON, "-m", "modules.experiment.scoring", "grid", csv_path, "-o", "acc"],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0
        assert "网格搜索完成" in r.stdout

    def test_no_command_shows_help(self):
        r = subprocess.run(
            [PYTHON, "-m", "modules.experiment.scoring"],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0 or "usage" in r.stderr.lower() or "usage" in r.stdout.lower()


# ━━━━━━━━━━━━━━━━ experiment/config.py CLI ━━━━━━━━━━━━━━━━


class TestConfigCLI:
    def test_list(self):
        r = subprocess.run(
            [PYTHON, "-m", "modules.experiment.config", "list"],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0
        assert "可用的实验配置" in r.stdout

    def test_show(self):
        r = subprocess.run(
            [PYTHON, "-m", "modules.experiment.config", "show", "default"],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0
        assert "exp_name" in r.stdout

    def test_create(self, tmp_path):
        r = subprocess.run(
            [PYTHON, "-m", "modules.experiment.config", "create"],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0
        assert "模板" in r.stdout


# ━━━━━━━━━━━━━━━━ prompt/manager.py CLI ━━━━━━━━━━━━━━━━


class TestPromptManagerCLI:
    def test_list(self):
        r = subprocess.run(
            [PYTHON, "-m", "modules.prompt.manager", "list"],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0

    def test_show(self):
        r = subprocess.run(
            [PYTHON, "-m", "modules.prompt.manager", "show", "cv_enhanced_p4"],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0

    def test_info(self):
        r = subprocess.run(
            [PYTHON, "-m", "modules.prompt.manager", "info", "cv_enhanced_p4"],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0
        assert "version" in r.stdout


# ━━━━━━━━━━━━━━━━ experiment/metrics.py CLI ━━━━━━━━━━━━━━━━


class TestMetricsCLI:
    def test_main_block(self):
        r = subprocess.run(
            [PYTHON, "-m", "modules.experiment.metrics"],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
        )
        assert r.returncode == 0
        assert "测试报告" in r.stdout


# ━━━━━━━━━━━━━━━━ settings.py dotenv fallback ━━━━━━━━━━━━━━━━


class TestSettingsDotenvFallback:
    def test_without_dotenv(self):
        """当 python-dotenv 未安装时走 ImportError 分支"""
        code = (
            "import sys; "
            "sys.modules['dotenv'] = None; "
            "import importlib; "
            "import modules.config.settings as s; "
            "importlib.reload(s); "
            "print('ok')"
        )
        r = subprocess.run(
            [PYTHON, "-c", code],
            capture_output=True,
            text=True,
            cwd="/root/XiaoanNew",
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        # 即使 dotenv 不可用也不应崩溃
        assert r.returncode == 0 or "仅使用系统环境变量" in r.stderr + r.stdout


class TestSettingsEnvFiles:
    def test_env_file_loading(self, tmp_path):
        """有 .env 文件时应加载"""
        import importlib

        real_mod = importlib.import_module("modules.config.settings")
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_SETTING_XYZ=hello12345\n")
        old_root = real_mod.project_root
        try:
            real_mod.project_root = tmp_path
            real_mod._load_env_files()
        finally:
            real_mod.project_root = old_root


# ━━━━━━━━━━━━━━━━ scoring 剩余分支: batch_evaluate ground_truth 解析 ━━━━━━━━━━━━━━━━


class TestScoringBatchGroundTruth:
    def test_gt_from_folder_name_合规(self, tmp_path):
        """ground_truth 列包含中文 '合规' / '不合规' 时的解析"""
        p = _make_scored_csv(
            tmp_path,
            rows=[
                ["a.jpg", "合规", "[合规]", "[合规]", "[完全合规]", "[合规]"],
                ["b.jpg", "不合规", "[不合规-构图]", "[合规]", "[完全合规]", "[合规]"],
            ],
        )
        from modules.experiment.scoring import ScoringEngine

        m = ScoringEngine().batch_evaluate(p)
        assert m["tp"] == 1
        assert m["tn"] == 1

    def test_gt_no_folder(self, tmp_path):
        """ground_truth 为空字符串时的处理"""
        p = str(tmp_path / "empty_gt.csv")
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["image", "ground_truth", "composition", "angle", "distance", "context"])
            w.writerow(["x.jpg", "", "[合规]", "[合规]", "[完全合规]", "[合规]"])
        from modules.experiment.scoring import ScoringEngine

        m = ScoringEngine().batch_evaluate(p)
        # 空 gt 应归为 "no" 类
        assert m["total"] == 1


class TestScoringGridSearchDefault:
    def test_default_weight_grid(self, tmp_path):
        """grid_search 使用默认 weight_grid"""
        p = _make_scored_csv(tmp_path)
        from modules.experiment.scoring import ScoringEngine

        r = ScoringEngine().grid_search(p)
        assert "weights" in r
        assert "threshold" in r
        assert r["metric"] >= 0


class TestScoringMissingWeightDim:
    def test_missing_weight(self):
        from modules.experiment.scoring import ScoringConfig, ScoringEngine

        c = ScoringConfig.default()
        # 先删除一个维度的权重，然后把总和补到1.0
        del c.weights["angle"]
        remaining_sum = sum(c.weights.values())
        # 按比例缩放使总和为1.0
        for k in c.weights:
            c.weights[k] = c.weights[k] / remaining_sum
        with pytest.raises(ValueError, match="缺少维度权重"):
            ScoringEngine(c)
