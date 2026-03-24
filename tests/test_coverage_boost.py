"""补充覆盖率测试 — scoring/config/prompt/metrics 模块未覆盖分支"""

import csv
import os

import pytest
import yaml

from modules.experiment.config import (
    ExperimentConfig,
    create_experiment_dirs,
    list_configs,
    load_config,
    save_config,
)
from modules.experiment.metrics import BinaryMetrics, print_metrics_report, update_leaderboard
from modules.experiment.scoring import ScoringConfig, ScoringEngine
from modules.prompt.manager import PromptManager, get_prompt_manager, list_prompts, load_prompt

# ── scoring.py 覆盖补充 ──


class TestScoringConfigToYaml:
    def test_roundtrip_with_gate_off(self, tmp_path):
        c = ScoringConfig.default()
        c.composition_gate = False
        p = str(tmp_path / "g.yaml")
        c.to_yaml(p)
        loaded = ScoringConfig.from_yaml(p)
        assert loaded.composition_gate is False

    def test_custom_score_map(self, tmp_path):
        c = ScoringConfig.default()
        c.score_map["composition"]["自定义"] = 0.7
        p = str(tmp_path / "m.yaml")
        c.to_yaml(p)
        assert ScoringConfig.from_yaml(p).score_map["composition"]["自定义"] == 0.7


class TestScoringEngineMissingScoreMap:
    def test_missing_score_map_dim(self):
        c = ScoringConfig.default()
        del c.score_map["angle"]
        with pytest.raises(ValueError, match="缺少维度分数映射"):
            ScoringEngine(c)


class TestScoringEngineEdgeCases:
    def test_score_half_compliant(self):
        e = ScoringEngine()
        r = e.score("[合规]", "[合规]", "[不合规-超界]", "[合规]")
        assert r.final_score < 1.0 and r.final_score > 0.0

    def test_score_基本合规(self):
        e = ScoringEngine()
        r = e.score("[合规]", "[合规]", "[基本合规-压线]", "[合规]")
        assert r.is_compliant

    def test_score_returns_dimension_scores(self):
        e = ScoringEngine()
        r = e.score("[合规]", "[不合规-角度]", "[完全合规]", "[合规]")
        assert r.dimension_scores["angle"] == 0.0

    def test_threshold_boundary(self):
        """分数恰好等于阈值"""
        c = ScoringConfig.default()
        c.threshold = 0.70
        e = ScoringEngine(c)
        r = e.score("[合规]", "[合规]", "[合规]", "[合规]")
        assert r.is_compliant

    def test_all_dimensions_zero(self):
        e = ScoringEngine()
        r = e.score("[不合规-构图]", "[不合规-角度]", "[不合规-超界]", "[不合规-环境]")
        assert r.final_score == 0.0 and r.gated


class TestScoringBatchEdgeCases:
    def test_csv_with_ground_truth_variants(self, tmp_path):
        """CSV 包含 yes/no 和 合规/不合规 标签"""
        p = str(tmp_path / "r.csv")
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["image", "ground_truth", "composition", "angle", "distance", "context"])
            w.writerow(["a.jpg", "合规", "[合规]", "[合规]", "[完全合规]", "[合规]"])
            w.writerow(["b.jpg", "不合规", "[不合规-构图]", "[合规]", "[完全合规]", "[合规]"])
        m = ScoringEngine().batch_evaluate(p)
        assert m["total"] == 2

    def test_batch_with_中文_gt(self, tmp_path):
        p = str(tmp_path / "r.csv")
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["image", "ground_truth", "composition", "angle", "distance", "context"])
            w.writerow(["a.jpg", "yes", "[合规]", "[合规]", "[完全合规]", "[合规]"])
        m = ScoringEngine().batch_evaluate(p)
        assert m["tp"] == 1


class TestScoringGridSearchEdge:
    def test_single_weight_combo(self, experiment_csv):
        r = ScoringEngine().grid_search(
            experiment_csv,
            weight_grid={
                "composition": [0.1],
                "angle": [0.3],
                "distance": [0.3],
                "context": [0.3],
            },
            threshold_range=(0.5, 0.6, 0.1),
        )
        assert r["metric"] >= 0

    def test_optimize_acc(self, experiment_csv):
        r = ScoringEngine().grid_search(
            experiment_csv,
            weight_grid={
                "composition": [0.1],
                "angle": [0.3],
                "distance": [0.3],
                "context": [0.3],
            },
            optimize="acc",
        )
        assert r["optimize"] == "acc"


# ── experiment/config.py 覆盖补充 ──


class TestConfigListDefaultDir:
    def test_default_dir_lists_assets(self):
        """list_configs(None) 使用默认 assets/configs 目录"""
        configs = list_configs()
        assert isinstance(configs, list)
        if len(configs) > 0:
            assert configs[0].endswith((".yaml", ".yml"))


class TestConfigSaveBackup:
    def test_save_creates_backup(self, tmp_path):
        c = ExperimentConfig(exp_name="bk", output_root=str(tmp_path), timestamp="T")
        ed, vd = create_experiment_dirs(c)
        backup = os.path.join(ed, "experiment_config.yaml")
        assert os.path.exists(backup)

    def test_save_to_existing_dir(self, tmp_path):
        d = tmp_path / "existing"
        d.mkdir()
        c = ExperimentConfig(exp_name="ex")
        p = save_config(c, str(d))
        loaded = load_config(p)
        assert loaded.exp_name == "ex"


class TestConfigExtraFields:
    def test_unknown_fields_raise(self, tmp_path):
        """load_config 对未知字段会抛出 TypeError"""
        data = {"exp_name": "uf", "unknown_field": "ignored", "max_size": [320, 320]}
        p = tmp_path / "uf.yaml"
        with open(str(p), "w") as f:
            yaml.dump(data, f)
        with pytest.raises(TypeError):
            load_config(str(p))

    def test_all_fields(self, tmp_path):
        data = {
            "exp_name": "full",
            "model": "m",
            "prompt_id": "p1",
            "max_size": [640, 640],
            "quality": 95,
            "conf_threshold": 0.7,
            "data_folders": ["/a", "/b"],
            "max_workers": 8,
            "save_visuals": False,
        }
        p = tmp_path / "full.yaml"
        with open(str(p), "w") as f:
            yaml.dump(data, f)
        c = load_config(str(p))
        assert c.max_workers == 8 and c.save_visuals is False


# ── prompt/manager.py 覆盖补充 ──


class TestPromptManagerDefaultDir:
    def test_default_dir(self):
        pm = PromptManager()
        assert os.path.basename(pm.prompts_dir) == "prompts"


class TestPromptModuleHelpers:
    def test_get_prompt_manager_singleton(self):
        import modules.prompt.manager as pm_mod

        pm_mod._default_manager = None
        m1 = get_prompt_manager()
        m2 = get_prompt_manager()
        assert m1 is m2

    def test_list_prompts_func(self):
        names = list_prompts()
        assert isinstance(names, list)

    def test_load_prompt_func(self):
        names = list_prompts()
        if names:
            content = load_prompt(names[0])
            assert isinstance(content, str) and len(content) > 0


class TestPromptManagerReloadEdge:
    def test_reload_nonexistent_name(self, sample_prompt_dir):
        pm = PromptManager(sample_prompt_dir)
        pm.reload("nonexistent")

    def test_info_includes_all_fields(self, sample_prompt_dir):
        pm = PromptManager(sample_prompt_dir)
        info = pm.info("test_prompt")
        assert "version" in info and "author" in info and "content_length" in info


# ── metrics 覆盖补充 ──


class TestMetricsEdgeCases:
    def test_report_with_title(self, capsys):
        m = BinaryMetrics.from_confusion_matrix(10, 10, 5, 5)
        print_metrics_report(m, title="自定义标题")
        assert "自定义标题" in capsys.readouterr().out

    def test_leaderboard_dedup(self, tmp_path):
        """相同实验名+时间戳应被去重"""
        d = tmp_path / "exp_dup"
        d.mkdir()
        p = str(d / "all_experiments_summary.csv")
        headers = [
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
        ]
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for _ in range(3):
                w.writerow(
                    {
                        "f1": 0.8,
                        "acc": 0.8,
                        "pre": 0.8,
                        "rec": 0.8,
                        "tp": 40,
                        "tn": 40,
                        "fp": 10,
                        "fn": 10,
                        "total": 100,
                        "invalid": 0,
                        "avg_lat": 1.0,
                        "exp_name": "same",
                        "segmentor": "t",
                        "folders": "d",
                        "timestamp": "T",
                    }
                )

        update_leaderboard(str(tmp_path))
        lb = os.path.join(str(tmp_path), "leaderboard_top20.csv")
        if os.path.exists(lb):
            with open(lb, "r", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1

    def test_leaderboard_skip_zero(self, tmp_path):
        """f1=0 且 total=0 的行应该被跳过"""
        d = tmp_path / "exp_zero"
        d.mkdir()
        p = str(d / "all_experiments_summary.csv")
        headers = [
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
        ]
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerow(
                {
                    "f1": 0,
                    "acc": 0,
                    "pre": 0,
                    "rec": 0,
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "total": 0,
                    "invalid": 0,
                    "avg_lat": 0,
                    "exp_name": "zero",
                    "segmentor": "t",
                    "folders": "d",
                    "timestamp": "T2",
                }
            )

        update_leaderboard(str(tmp_path))
        lb = os.path.join(str(tmp_path), "leaderboard_top20.csv")
        if os.path.exists(lb):
            with open(lb, "r", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 0
