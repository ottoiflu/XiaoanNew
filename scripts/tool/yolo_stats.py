#!/usr/bin/env python3
"""YOLO 全量检出率统计 + 无标线场景抽样可视化。"""
import json, os, sys, collections, random
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/root/otto/XiaoanNew")
from modules.config.settings import settings
from modules.cv.yolov8_inference import load_yolov8_seg

GT="/root/otto/XiaoanNew/data/benchmark/benchmark_v4/fourdim_gt_v4.json"
DATA_ROOT="/root/otto/XiaoanNew/data"
OUT_DIR="/root/otto/XiaoanNew/outputs/benchmark_output/v4/yolo_check"
os.makedirs(OUT_DIR, exist_ok=True)

gt=json.load(open(GT))
N=len(gt)
print(f"gt 条目: {N}")
print("[Load] YOLOv8-Seg ...")
seg=load_yolov8_seg(settings.YOLO_WEIGHTS, device=settings.INFERENCE_DEVICE)

label_present=collections.Counter()
label_total=collections.Counter()
per_img=[]
no_line=[]

def find_img(src):
    p=os.path.join(DATA_ROOT, src)
    if os.path.exists(p): return p
    for ext in (".JPEG",".jpeg",".png",".jpg"):
        alt=os.path.splitext(p)[0]+ext
        if os.path.exists(alt): return alt
    return None

for i, entry in enumerate(gt):
    img_path=find_img(entry.get("src",""))
    if not img_path:
        per_img.append({}); continue
    img_bytes=open(img_path,"rb").read()
    res=seg.predict(img_bytes, visual=False, retina_masks=True)
    objs=res["objects"]
    cnt=collections.Counter(o["label"] for o in objs)
    per_img.append(dict(cnt))
    for lb in cnt: label_present[lb]+=1
    for lb,n in cnt.items(): label_total[lb]+=n
    if cnt.get("parking lane",0)==0:
        no_line.append((entry["id"], img_path, dict(cnt)))
    if (i+1)%200==0: print(f"  {i+1}/{N}", flush=True)

print(f"\n=== YOLO 各类别检出率 (共{N}张) ===")
for lb, n in label_present.most_common():
    print(f"  {lb:20s}: {n}/{N} = {n/N:.1%}  (平均{label_total[lb]/N:.2f}个/图)")

has_line=sum(1 for c in per_img if c.get("parking lane",0)>0)
has_curb=sum(1 for c in per_img if c.get("Curb",0)>0)
both=sum(1 for c in per_img if c.get("parking lane",0)>0 and c.get("Curb",0)>0)
neither=sum(1 for c in per_img if c.get("parking lane",0)==0 and c.get("Curb",0)==0)
print(f"\n标线检出: {has_line}/{N}={has_line/N:.1%}")
print(f"路缘检出: {has_curb}/{N}={has_curb/N:.1%}")
print(f"标线+路缘都有: {both}/{N}={both/N:.1%}")
print(f"标线路缘都无: {neither}/{N}={neither/N:.1%}")

nl_curb=sum(1 for _,_,c in no_line if c.get("Curb",0)>0)
print(f"\n无标线场景({len(no_line)}张) 路缘检出: {nl_curb}/{len(no_line)}={nl_curb/len(no_line):.1%}")

random.seed(42)
sample=random.sample(no_line, min(10, len(no_line)))
print(f"\n抽样 {len(sample)} 张无标线图可视化...")
COLORS={"Electric bike":(0,255,0),"Curb":(255,128,0),"parking lane":(255,255,0),"Tactile paving":(255,0,0),"Green belt":(0,128,51)}
for idx,(eid,img_path,cnt) in enumerate(sample):
    pil=Image.open(img_path).convert("RGB")
    img_bytes=open(img_path,"rb").read()
    res=seg.predict(img_bytes, visual=False, retina_masks=True)
    objs=res["objects"]
    arr=np.array(pil).astype(np.float32)
    for o in objs:
        lb=o["label"]; color=COLORS.get(lb,(128,128,128))
        mask=o.get("mask")
        if isinstance(mask,np.ndarray) and mask.sum()>0:
            m=mask.astype(bool)
            if m.shape==arr.shape[:2]:
                for c in range(3):
                    arr[:,:,c][m]=arr[:,:,c][m]*0.5+color[c]*0.5
    overlay=Image.fromarray(arr.astype(np.uint8))
    draw=ImageDraw.Draw(overlay)
    for o in objs:
        lb=o["label"]; color=COLORS.get(lb,(128,128,128))
        bx1,by1,bx2,by2=o["bbox"]
        draw.rectangle([bx1,by1,bx2,by2],outline=color,width=3)
        draw.text((bx1,max(0,by1-15)),lb,fill=color)
    out=os.path.join(OUT_DIR, f"noline_{idx:02d}.jpg")
    overlay.save(out, quality=85)
    occ=collections.Counter(o["label"] for o in objs)
    print(f"  {idx}: {eid[:50]} | {dict(occ)}")

print(f"\n可视化目录: {OUT_DIR}")
