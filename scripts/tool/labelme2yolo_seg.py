"""LabelMe 标注转 YOLO 实例分割格式，并按比例划分训练集和验证集。"""

import json
import random
import shutil
from pathlib import Path

CLASS_MAP = {
    "Electric bike": 0,
    "Curb": 1,
    "parking lane": 2,
    "Tactile paving": 3,
}

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def rectangle_to_polygon(points):
    """将 rectangle 的两点表示转为四点多边形。"""
    x1, y1 = points[0]
    x2, y2 = points[1]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def convert_one(json_path, img_w, img_h):
    """将单个 LabelMe JSON 转为 YOLO seg 格式的文本行列表。"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in CLASS_MAP:
            print(f"  [skip] unknown label '{label}' in {json_path.name}")
            continue

        cls_id = CLASS_MAP[label]
        points = shape["points"]

        if shape["shape_type"] == "rectangle":
            points = rectangle_to_polygon(points)

        if len(points) < 3:
            print(f"  [skip] too few points ({len(points)}) in {json_path.name}")
            continue

        normalized = []
        for px, py in points:
            nx = max(0.0, min(1.0, px / img_w))
            ny = max(0.0, min(1.0, py / img_h))
            normalized.extend([f"{nx:.6f}", f"{ny:.6f}"])

        lines.append(f"{cls_id} " + " ".join(normalized))

    return lines


def main():
    """执行 LabelMe -> YOLO 格式转换和数据集划分。"""
    project_root = Path(__file__).resolve().parents[2]
    src_dir = project_root / "data" / "all_labeled_data"
    out_dir = project_root / "data" / "yolo_seg_dataset"

    if out_dir.exists():
        print(f"target dir exists, clearing: {out_dir}")
        shutil.rmtree(out_dir)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    json_files = sorted(src_dir.glob("*.json"))
    pairs = []
    skipped = 0
    for jf in json_files:
        img_file = jf.with_suffix(".jpg")
        if not img_file.exists():
            img_file = jf.with_suffix(".png")
        if not img_file.exists():
            print(f"[warn] image not found: {jf.stem}.*")
            skipped += 1
            continue
        pairs.append((jf, img_file))

    print(f"Found {len(pairs)} valid pairs, skipped {skipped}")

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    total_labels = 0
    empty_labels = 0
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for jf, img_file in split_pairs:
            with open(jf, encoding="utf-8") as f:
                meta = json.load(f)
            img_w = meta["imageWidth"]
            img_h = meta["imageHeight"]

            lines = convert_one(jf, img_w, img_h)
            total_labels += len(lines)
            if not lines:
                empty_labels += 1

            dst_img = out_dir / "images" / split_name / img_file.name
            shutil.copy2(img_file, dst_img)

            txt_name = jf.stem + ".txt"
            dst_label = out_dir / "labels" / split_name / txt_name
            with open(dst_label, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
                if lines:
                    f.write("\n")

    yaml_content = f"""# YOLOv8 instance segmentation dataset config
path: {out_dir}
train: images/train
val: images/val

names:
  0: Electric bike
  1: Curb
  2: parking lane
  3: Tactile paving

# Train: {len(train_pairs)} | Val: {len(val_pairs)} | Labels: {total_labels}
"""
    yaml_path = out_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print("\nDone:")
    print(f"  Output: {out_dir}")
    print(f"  Total labels: {total_labels}")
    print(f"  Empty label files: {empty_labels}")
    print(f"  dataset.yaml: {yaml_path}")

    sample_label = list((out_dir / "labels" / "train").glob("*.txt"))[0]
    print(f"\nSample check ({sample_label.name}):")
    with open(sample_label) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = parts[0]
            n_points = (len(parts) - 1) // 2
            cls_name = list(CLASS_MAP.keys())[int(cls)]
            print(f"  class={cls} ({cls_name}), polygon_vertices={n_points}")


if __name__ == "__main__":
    main()
