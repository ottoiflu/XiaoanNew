"""modules.experiment.config 单元测试"""

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


class TestExperimentConfig:
    def test_defaults(self):
        c = ExperimentConfig(exp_name="test")
        assert c.quality == 80
        assert isinstance(c.max_size, tuple)
        assert c.timestamp

    def test_to_dict(self):
        d = ExperimentConfig(exp_name="t").to_dict()
        assert isinstance(d["max_size"], list)

    def test_exp_dir(self):
        c = ExperimentConfig(exp_name="e", output_root="/tmp/o", timestamp="T")
        assert c.exp_dir == "/tmp/o/exp_T_e"

    def test_vis_dir(self):
        c = ExperimentConfig(exp_name="e", output_root="/tmp/o", timestamp="T")
        assert c.vis_dir.endswith("visuals")


class TestLoadConfig:
    def test_valid(self, tmp_path):
        data = {"exp_name": "loaded", "max_size": [640, 640], "quality": 90}
        p = tmp_path / "t.yaml"
        with open(str(p), "w") as f:
            yaml.dump(data, f)
        c = load_config(str(p))
        assert c.exp_name == "loaded"
        assert c.max_size == (640, 640)

    def test_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent.yaml")

    def test_minimal(self, tmp_path):
        p = tmp_path / "m.yaml"
        with open(str(p), "w") as f:
            yaml.dump({"exp_name": "min"}, f)
        assert load_config(str(p)).quality == 80


class TestSaveConfig:
    def test_to_dir(self, tmp_path):
        c = ExperimentConfig(exp_name="s")
        path = save_config(c, str(tmp_path))
        assert os.path.exists(path) and path.endswith(".yaml")

    def test_to_file(self, tmp_path):
        p = str(tmp_path / "c.yaml")
        save_config(ExperimentConfig(exp_name="s"), p)
        assert os.path.exists(p)

    def test_roundtrip(self, tmp_path):
        c = ExperimentConfig(exp_name="rt", quality=75, max_size=(512, 512))
        loaded = load_config(save_config(c, str(tmp_path)))
        assert loaded.exp_name == "rt" and loaded.max_size == (512, 512)


class TestCreateExperimentDirs:
    def test_creates(self, tmp_path):
        c = ExperimentConfig(exp_name="mk", output_root=str(tmp_path), timestamp="T")
        ed, vd = create_experiment_dirs(c)
        assert os.path.isdir(ed) and os.path.isdir(vd)

    def test_no_visuals(self, tmp_path):
        c = ExperimentConfig(exp_name="nv", output_root=str(tmp_path), save_visuals=False, timestamp="T2")
        ed, vd = create_experiment_dirs(c)
        assert os.path.isdir(ed) and not os.path.isdir(vd)


class TestListConfigs:
    def test_lists_yaml(self, tmp_path):
        (tmp_path / "a.yaml").touch()
        (tmp_path / "b.yml").touch()
        (tmp_path / "c.txt").touch()
        r = list_configs(str(tmp_path))
        assert "a.yaml" in r and "b.yml" in r and "c.txt" not in r

    def test_empty(self, tmp_path):
        assert list_configs(str(tmp_path)) == []

    def test_nonexistent(self):
        assert list_configs("/nonexistent") == []

    def test_sorted(self, tmp_path):
        for n in ("z.yaml", "a.yaml", "m.yaml"):
            (tmp_path / n).touch()
        assert list_configs(str(tmp_path)) == ["a.yaml", "m.yaml", "z.yaml"]
