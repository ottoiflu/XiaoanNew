"""Benchmark v2 数据划分工具

按固定种子将 fourdim_gt_v2.json 划分为 train/val/test。
保证各 split 内 yes/no 均衡。
输出 split.json（id→split）。

用法：
    uv run python scripts/tool/split_benchmark.py
"""

import json
import os
import random

_PROJECT_ROOT = "/root/otto/XiaoanNew"
DATA_PATH = os.path.join(_PROJECT_ROOT, "data/benchmark/benchmark_v2/fourdim_gt_v2.json")
OUT_PATH = os.path.join(_PROJECT_ROOT, "data/benchmark/benchmark_v2/split.json")

SEED = 42
TRAIN_RATIO = 0.5   # 300
VAL_RATIO = 0.25    # 150
# TEST = remainder → 150

with open(DATA_PATH, encoding="utf-8") as f:
    items = json.load(f)

print(f"Total: {len(items)}")

# 按 gt 分层
yes_items = [d for d in items if d["gt"] == "yes"]
no_items  = [d for d in items if d["gt"] == "no"]
print(f"yes={len(yes_items)} no={len(no_items)}")

rng = random.Random(SEED)
rng.shuffle(yes_items)
rng.shuffle(no_items)

split = {}
key_map = {}

def assign_split(items_list, train_end, val_end, key_map):
    """将排序后列表前 train_end 张分给 train, train_end~val_end 给 val, 其余 test"""
    for i, item in enumerate(items_list):
        key = item["id"]
        # 处理重复 ID：加 _dup_N 后缀
        if key in split:
            dup_idx = 1
            while f"{key}_dup{dup_idx}" in split:
                dup_idx += 1
            key = f"{key}_dup{dup_idx}"
        key_map[key] = item["id"]  # 记录原始 ID
        if i < train_end:
            split[key] = "train"
        elif i < val_end:
            split[key] = "val"
        else:
            split[key] = "test"

# 每类: train=150, val=75, test=75 (共 300/150/150)
assign_split(yes_items, 150, 225, key_map)
assign_split(no_items, 150, 225, key_map)

# 验证
train = [k for k, v in split.items() if v == "train"]
val   = [k for k, v in split.items() if v == "val"]
test  = [k for k, v in split.items() if v == "test"]
print(f"\ntrain={len(train)} val={len(val)} test={len(test)}")

# 各 split 内 gt 平衡
gt_lookup = {d["id"]: d["gt"] for d in items}

def count_gt(id_list):
    yes_c = 0
    for iid in id_list:
        orig_id = key_map.get(iid, iid)
        if gt_lookup.get(orig_id) == "yes":
            yes_c += 1
    return yes_c, len(id_list) - yes_c

for name, lst in [("train", train), ("val", val), ("test", test)]:
    y, n = count_gt(lst)
    print(f"  {name}: yes={y} no={n}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(split, f, ensure_ascii=False, indent=2)
print(f"\nSplit saved: {OUT_PATH} ({len(split)} entries)")

# 保存 ID 映射
id_map_path = os.path.join(os.path.dirname(OUT_PATH), "id_map.json")
with open(id_map_path, "w", encoding="utf-8") as f:
    json.dump(key_map, f, ensure_ascii=False, indent=2)
print(f"ID map saved: {id_map_path}")