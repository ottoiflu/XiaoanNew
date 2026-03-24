"""实验 I/O 工具

提供标签加载、CSV 结果写入、汇总指标追加等实验通用功能。
"""

from __future__ import annotations

import csv
import os
from typing import Callable, Optional

from utils.vlm_parser import normalize_label


def load_labels(
    folder_path: str,
    normalize_fn: Callable[[str], str] = normalize_label,
) -> dict[tuple[str, str], str]:
    """从 labels.txt 加载图片-标签映射

    Args:
        folder_path: 包含 labels.txt 的目录
        normalize_fn: 标签标准化函数
    Returns:
        {(image_name, folder_path): "yes"/"no"}
    """
    labels = {}
    label_file = os.path.join(folder_path, "labels.txt")
    if not os.path.exists(label_file):
        return labels
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                labels[(parts[0].strip(), folder_path)] = normalize_fn(parts[1])
    return labels


def load_all_labels(
    folders: list[str],
    normalize_fn: Callable[[str], str] = normalize_label,
) -> dict[tuple[str, str], str]:
    """从多个目录加载并合并标签"""
    merged = {}
    for folder in folders:
        merged.update(load_labels(folder, normalize_fn))
    return merged


def collect_image_tasks(
    folders: list[str],
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> list[tuple[str, str]]:
    """收集所有目录中的图片文件列表

    Returns:
        [(image_name, folder_path), ...]
    """
    tasks = []
    for folder in folders:
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith(extensions):
                tasks.append((f, folder))
    return tasks


class ResultWriter:
    """实验结果 CSV 写入器（上下文管理器）"""

    def __init__(self, csv_path: str, headers: list[str]):
        self.csv_path = csv_path
        self.headers = headers
        self._rows: list[list] = []
        self._file = None
        self._writer = None

    def __enter__(self):
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8-sig")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.headers)
        return self

    def __exit__(self, *exc):
        if self._file:
            self._file.close()
        return False

    def write_row(self, row: list) -> None:
        if self._writer:
            self._writer.writerow(row)
            self._file.flush()
        self._rows.append(row)

    @property
    def rows(self) -> list[list]:
        return self._rows


def append_summary(
    summary_path: str,
    metrics: dict,
    extra_fields: Optional[dict] = None,
) -> None:
    """追加一行汇总指标到汇总 CSV"""
    row = dict(metrics)
    if extra_fields:
        row.update(extra_fields)

    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
