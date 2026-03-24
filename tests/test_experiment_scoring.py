"""modules.experiment.scoring 单元测试"""

import pytest

from modules.experiment.scoring import ScoringConfig, ScoringEngine


class TestScoringConfig:
    def test_default(self):
        c = ScoringConfig.default()
        assert abs(sum(c.weights.values()) - 1.0) < 0.01
        assert c.threshold == 0.60

    def test_from_yaml(self, scoring_config_yaml):
        c = ScoringConfig.from_yaml(scoring_config_yaml)
        assert c.threshold == 0.5

    def test_roundtrip(self, tmp_path):
        c = ScoringConfig.default()
        p = str(tmp_path / "out.yaml")
        c.to_yaml(p)
        assert ScoringConfig.from_yaml(p).threshold == c.threshold


class TestScoringEngineInit:
    def test_default(self):
        assert ScoringEngine().config is not None

    def test_bad_weights_sum(self):
        c = ScoringConfig.default()
        c.weights["angle"] = 0.9
        with pytest.raises(ValueError, match="权重总和"):
            ScoringEngine(c)

    def test_missing_dimension(self):
        c = ScoringConfig.default()
        del c.weights["angle"]
        with pytest.raises(ValueError):
            ScoringEngine(c)

    def test_from_yaml(self, scoring_config_yaml):
        assert isinstance(ScoringEngine.from_yaml(scoring_config_yaml), ScoringEngine)


class TestScore:
    @pytest.fixture()
    def engine(self):
        return ScoringEngine()

    def test_all_compliant(self, engine):
        r = engine.score("[合规]", "[合规]", "[完全合规]", "[合规]")
        assert r.is_compliant and not r.gated

    def test_gate_triggers(self, engine):
        r = engine.score("[不合规-构图]", "[合规]", "[完全合规]", "[合规]")
        assert not r.is_compliant and r.gated and r.final_score == 0.0

    def test_gate_disabled(self):
        c = ScoringConfig.default()
        c.composition_gate = False
        r = ScoringEngine(c).score("[不合规-构图]", "[合规]", "[完全合规]", "[合规]")
        assert not r.gated and r.final_score > 0

    def test_all_non_compliant(self, engine):
        assert not engine.score("[不合规-构图]", "[不合规-角度]", "[不合规-超界]", "[不合规-环境]").is_compliant

    def test_whitespace(self, engine):
        r = engine.score("  [合规]  ", " [合规] ", " [完全合规] ", " [合规] ")
        assert r.is_compliant

    def test_dimension_scores(self, engine):
        r = engine.score("[合规]", "[合规]", "[完全合规]", "[合规]")
        for d in ("composition", "angle", "distance", "context"):
            assert d in r.dimension_scores


class TestJudge:
    def test_yes(self):
        assert ScoringEngine().judge("[合规]", "[合规]", "[完全合规]", "[合规]") == "yes"

    def test_no(self):
        assert ScoringEngine().judge("[不合规-构图]", "[合规]", "[完全合规]", "[合规]") == "no"


class TestVetoJudge:
    def test_all_ok(self):
        assert ScoringEngine.veto_judge("[合规]", "[合规]", "[完全合规]", "[合规]") == "yes"

    def test_composition_veto(self):
        assert ScoringEngine.veto_judge("[不合规-构图]", "[合规]", "[完全合规]", "[合规]") == "no"

    def test_angle_veto(self):
        assert ScoringEngine.veto_judge("[合规]", "[不合规-角度]", "[完全合规]", "[合规]") == "no"

    def test_context_veto(self):
        assert ScoringEngine.veto_judge("[合规]", "[合规]", "[完全合规]", "[不合规-环境]") == "no"

    def test_distance_超界(self):
        assert ScoringEngine.veto_judge("[合规]", "[合规]", "[不合规-超界]", "[合规]") == "no"

    def test_压线_no_veto(self):
        assert ScoringEngine.veto_judge("[合规]", "[合规]", "[基本合规-压线]", "[合规]") == "yes"


class TestBatchEvaluate:
    def test_basic(self, experiment_csv):
        m = ScoringEngine().batch_evaluate(experiment_csv)
        assert "acc" in m and m["total"] > 0


class TestSweepThreshold:
    def test_returns_list(self, experiment_csv):
        r = ScoringEngine().sweep_threshold(experiment_csv, start=0.0, stop=1.0, step=0.5)
        assert isinstance(r, list) and len(r) >= 2 and "threshold" in r[0]


class TestGridSearch:
    def test_basic(self, experiment_csv):
        r = ScoringEngine().grid_search(
            experiment_csv,
            weight_grid={"composition": [0.1], "angle": [0.3], "distance": [0.3], "context": [0.3]},
            threshold_range=(0.3, 0.7, 0.2),
        )
        assert "weights" in r and "threshold" in r


class TestFuzzyMatch:
    def test_exact(self):
        assert ScoringEngine._fuzzy_match("[合规]", {"[合规]": 1.0}) == 1.0

    def test_fallback_不合规(self):
        assert ScoringEngine._fuzzy_match("不合规xyz", {}) == 0.0

    def test_fallback_合规(self):
        assert ScoringEngine._fuzzy_match("合规", {}) == 1.0

    def test_fallback_基本(self):
        assert ScoringEngine._fuzzy_match("基本合格", {}) == 0.5

    def test_unknown(self):
        assert ScoringEngine._fuzzy_match("????", {}) == 0.0
