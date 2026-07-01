#!/usr/bin/env python3
"""为新四维人工标注备料：合并 benchmark v1(88) + 旧标注(100)，
生成缩略图 + 带「旧标签按新四维预填」的标注清单。

新四维 schema：
  position 停放位置 : [合规]/[基本合规-压线]/[不合规-超界]/[无参照]
  medium   禁停介质 : [合规]/[不合规-盲道]/[不合规-绿化]/[不合规-禁停区]
  angle    角度     : [合规]/[不合规-斜停]/[N/A]
  state    车辆状态 : [正立]/[倒伏]

输出（服务器）：
  data/labeling/fourdim/thumbs/<id>.jpg
  data/labeling/fourdim/label_manifest.json
"""
import csv
import glob
import json
import os

from PIL import Image

ROOT = "/root/otto/XiaoanNew/data/compliance"
BENCH = "/root/otto/XiaoanNew/data/benchmark_v1/manifest_v1.csv"
OUT = "/root/otto/XiaoanNew/data/labeling/fourdim"
THUMBS = os.path.join(OUT, "thumbs")
os.makedirs(THUMBS, exist_ok=True)


def stem(fn):
    return os.path.splitext(os.path.basename(fn))[0]


def prefill_from_old(ann):
    """旧四维(composition/angle/distance/context) → 新四维预填 + 旧 reason 提示。"""
    pf, reasons = {}, {}
    comp, ang = ann.get("composition", {}), ann.get("angle", {})
    dist, ctx = ann.get("distance", {}), ann.get("context", {})
    # 角度：直接映射
    a = ang.get("status", "")
    if a == "[合规]":
        pf["angle"] = "[合规]"
    elif "不合规" in a:
        pf["angle"] = "[不合规-斜停]"
    if ang.get("reason"):
        reasons["angle"] = ang["reason"]
    # 位置：旧"无参照"优先置 [无参照]（修掉旧版误判超界），否则按 distance 映射
    cs, ds = comp.get("status", ""), dist.get("status", "")
    if "无参照" in cs:
        pf["position"] = "[无参照]"
    elif ds == "[完全合规]":
        pf["position"] = "[合规]"
    elif "压线" in ds:
        pf["position"] = "[基本合规-压线]"
    elif "超界" in ds:
        pf["position"] = "[不合规-超界]"
    rp = dist.get("reason", "")
    if comp.get("reason"):
        rp = (rp + " ｜旧构图: " + comp["reason"]).strip(" ｜")
    if rp:
        reasons["position"] = rp
    # 介质：旧合规→合规；旧不合规-环境 留空（需重选盲道/绿化/禁停区），带 reason
    if ctx.get("status", "") == "[合规]":
        pf["medium"] = "[合规]"
    if ctx.get("reason"):
        reasons["medium"] = ctx["reason"]
    # 车辆状态：旧无此维，不预填
    return pf, reasons


items = {}

# ---- benchmark 88：物理文件在 by_scene/standard/{yes,no}/，gt=文件夹 ----
std_index = {}
for gt in ("yes", "no"):
    for p in glob.glob(os.path.join(ROOT, "by_scene", "standard", gt, "*")):
        std_index[stem(p)] = (p, gt)

with open(BENCH, encoding="utf-8-sig") as f:
    rd = csv.DictReader(f)
    cols = rd.fieldnames or []
    fcol = next((c for c in cols if c.lower() in ("image", "file", "filename", "name", "path")), cols[0])
    for row in rd:
        s = stem(row[fcol])
        if s in std_index:
            p, gt = std_index[s]
            items[s] = {"id": s, "src": p, "gt": gt, "source": "benchmark", "prefill": {}, "reasons": {}}

n_bench = len(items)

# ---- 旧标注 100 ----
ann_all = json.load(open(os.path.join(ROOT, "annotations_corrected.json"), encoding="utf-8"))["annotations"]
for fn, ann in ann_all.items():
    s = stem(fn)
    sf = ann.get("source_folder", "")
    path = os.path.join(ROOT, sf, fn)
    if not os.path.exists(path):
        g = glob.glob(os.path.join(ROOT, sf, s + ".*"))
        path = g[0] if g else None
    pf, reasons = prefill_from_old(ann)
    if s in items:  # 与 benchmark 重叠：合并预填
        items[s]["prefill"].update(pf)
        items[s]["reasons"].update(reasons)
        items[s]["source"] = "both"
    elif path:
        items[s] = {"id": s, "src": path, "gt": ann.get("compliance", ""),
                    "source": "old", "prefill": pf, "reasons": reasons}

# ---- 缩略图 + 清单 ----
manifest, err = [], 0
for s, rec in items.items():
    try:
        im = Image.open(rec["src"]).convert("RGB")
        im.thumbnail((1280, 1280))
        im.save(os.path.join(THUMBS, s + ".jpg"), "JPEG", quality=85)
    except Exception as e:
        err += 1
        print("THUMB ERR", s, e)
        continue
    manifest.append({"id": s, "thumb": "thumbs/" + s + ".jpg", "gt": rec["gt"],
                     "source": rec["source"], "prefill": rec["prefill"], "reasons": rec["reasons"]})

json.dump(manifest, open(os.path.join(OUT, "label_manifest.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

n_old = sum(1 for m in manifest if m["source"] in ("old", "both"))
n_both = sum(1 for m in manifest if m["source"] == "both")
n_pf = sum(1 for m in manifest if m["prefill"])
print(f"benchmark匹配={n_bench}  旧标注={n_old}  重叠={n_both}  总计={len(manifest)}  有预填={n_pf}  缩略图错误={err}")
print("输出:", os.path.join(OUT, "label_manifest.json"))
