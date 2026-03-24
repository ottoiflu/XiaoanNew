"""modules.experiment.io 单元测试"""

import csv

from modules.experiment.io import (
    ResultWriter,
    append_summary,
    collect_image_tasks,
    load_all_labels,
    load_labels,
)


class TestLoadLabels:
    def test_basic(self, sample_labels_dir):
        labels = load_labels(sample_labels_dir)
        assert labels[("img001.jpg", sample_labels_dir)] == "yes"
        assert labels[("img002.jpg", sample_labels_dir)] == "no"

    def test_chinese(self, sample_labels_dir):
        labels = load_labels(sample_labels_dir)
        assert labels[("img003.jpg", sample_labels_dir)] == "yes"

    def test_missing_file(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert load_labels(str(d)) == {}

    def test_custom_fn(self, sample_labels_dir):
        labels = load_labels(sample_labels_dir, normalize_fn=lambda x: x.strip().upper())
        assert labels[("img001.jpg", sample_labels_dir)] == "YES"

    def test_malformed_lines(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        with open(str(d / "labels.txt"), "w") as f:
            f.write("good.jpg, yes\nbad_no_comma\n\nanother.jpg, no\n")
        assert len(load_labels(str(d))) == 2


class TestLoadAllLabels:
    def test_merge(self, tmp_path):
        for name, content in [("d1", "a.jpg, yes"), ("d2", "b.jpg, no")]:
            d = tmp_path / name
            d.mkdir()
            with open(str(d / "labels.txt"), "w") as f:
                f.write(content)
        assert len(load_all_labels([str(tmp_path / "d1"), str(tmp_path / "d2")])) == 2

    def test_empty(self):
        assert load_all_labels([]) == {}


class TestCollectImageTasks:
    def test_collect(self, sample_labels_dir):
        tasks = collect_image_tasks([sample_labels_dir])
        names = [t[0] for t in tasks]
        assert "img001.jpg" in names

    def test_excludes_non_images(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "a.jpg").touch()
        (d / "b.txt").touch()
        names = [t[0] for t in collect_image_tasks([str(d)])]
        assert "a.jpg" in names and "b.txt" not in names

    def test_nonexistent_skipped(self, tmp_path):
        assert collect_image_tasks([str(tmp_path / "nope")]) == []

    def test_empty(self, tmp_path):
        d = tmp_path / "e"
        d.mkdir()
        assert collect_image_tasks([str(d)]) == []


class TestResultWriter:
    def test_write_read(self, tmp_path):
        p = str(tmp_path / "r.csv")
        with ResultWriter(p, ["name", "score"]) as w:
            w.write_row(["img1", 0.9])
            w.write_row(["img2", 0.5])
        with open(p, "r", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        assert rows[1] == ["img1", "0.9"]

    def test_rows_property(self, tmp_path):
        with ResultWriter(str(tmp_path / "r.csv"), ["a"]) as w:
            w.write_row([1])
            w.write_row([2])
            assert len(w.rows) == 2

    def test_closes_file(self, tmp_path):
        w = ResultWriter(str(tmp_path / "c.csv"), ["x"])
        with w:
            w.write_row(["v"])
        assert w._file.closed

    def test_unicode(self, tmp_path):
        p = str(tmp_path / "u.csv")
        with ResultWriter(p, ["名称"]) as w:
            w.write_row(["图片1"])
        with open(p, "r", encoding="utf-8-sig") as f:
            assert "图片1" in f.read()


class TestAppendSummary:
    def test_creates_new(self, tmp_path):
        p = str(tmp_path / "s.csv")
        append_summary(p, {"acc": 0.9})
        with open(p, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["acc"] == "0.9"

    def test_appends(self, tmp_path):
        p = str(tmp_path / "s.csv")
        append_summary(p, {"acc": 0.9})
        append_summary(p, {"acc": 0.8})
        with open(p, "r", encoding="utf-8-sig") as f:
            lines = [line for line in f.read().strip().split("\n") if line]
        assert len(lines) == 3

    def test_extra_fields(self, tmp_path):
        p = str(tmp_path / "s.csv")
        append_summary(p, {"acc": 0.9}, extra_fields={"name": "t"})
        with open(p, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["name"] == "t"
