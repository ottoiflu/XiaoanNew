#!/usr/bin/env python3
"""把人工复核产出的降级清单落盘到 by_scene/。

读取 review.html 导出的 scene_review_demotions.csv（列：comp, byscene_stem, decision），
将被降级的图片从 by_scene/standard/{comp}/ 移动到 by_scene/longtail/{comp}/，
并同步更新 manifest.csv 的 scene 列。原图（compliance 各源目录）不动，仅调整场景分桶。

用法：
    python3 apply_scene_review.py --demotions scene_review_demotions.csv          # 预览（默认 dry-run）
    python3 apply_scene_review.py --demotions scene_review_demotions.csv --apply   # 实际执行
"""
import argparse
import csv
import glob
import os
import shutil

DEFAULT_ROOT = "/root/otto/XiaoanNew/data/compliance"


def load_demotions(path):
    """读取降级清单，返回 [(comp, byscene_stem), ...]。"""
    out = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("decision", "").strip() == "longtail":
                out.append((row["comp"].strip(), row["byscene_stem"].strip()))
    return out


def update_manifest(manifest_path, demoted_keys, apply):
    """把 manifest 中命中的行 scene 从 standard 改为 longtail。

    demoted_keys: set of (source_folder, original_stem)
    """
    if not os.path.exists(manifest_path):
        print(f"[警告] 未找到 manifest: {manifest_path}")
        return 0
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    changed = 0
    for r in rows:
        key = (r["source_folder"], os.path.splitext(r["file"])[0])
        if r["scene"] == "standard" and key in demoted_keys:
            r["scene"] = "longtail"
            changed += 1

    if apply and changed:
        shutil.copy2(manifest_path, manifest_path + ".bak")
        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return changed


def main():
    ap = argparse.ArgumentParser(description="落盘场景复核降级清单")
    ap.add_argument("--demotions", required=True, help="scene_review_demotions.csv 路径")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="compliance 目录")
    ap.add_argument("--apply", action="store_true", help="实际执行（默认仅预览）")
    args = ap.parse_args()

    by_scene = os.path.join(args.root, "by_scene")
    demotions = load_demotions(args.demotions)
    print(f">>> 降级清单条目: {len(demotions)}  模式: {'执行' if args.apply else '预览(dry-run)'}")

    moved, missing = 0, 0
    demoted_keys = set()
    for comp, stem in demotions:
        src_dir = os.path.join(by_scene, "standard", comp)
        dst_dir = os.path.join(by_scene, "longtail", comp)
        matches = glob.glob(os.path.join(src_dir, stem + ".*"))
        if not matches:
            print(f"[缺失] standard/{comp}/{stem}.* 未找到（可能已移动）")
            missing += 1
            continue
        # byscene_stem = {source_folder}__{original_stem}
        if "__" in stem:
            folder, orig_stem = stem.split("__", 1)
            demoted_keys.add((folder, orig_stem))
        for src in matches:
            dst = os.path.join(dst_dir, os.path.basename(src))
            print(f"  move standard/{comp}/{os.path.basename(src)} -> longtail/{comp}/")
            if args.apply:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src, dst)
            moved += 1

    manifest = os.path.join(by_scene, "manifest.csv")
    changed = update_manifest(manifest, demoted_keys, args.apply)

    print("\n==================== 汇总 ====================")
    print(f"  移动文件: {moved}  缺失: {missing}  manifest 改动行: {changed}")
    if not args.apply:
        print("  这是预览。确认无误后加 --apply 实际执行。")
    else:
        print(f"  已执行。manifest 备份: {manifest}.bak")


if __name__ == "__main__":
    main()
