"""用2D图像PCA覆盖VLM angle,重算scoring和准确率。不需要深度估计,只用mask像素坐标。"""
import sys, os, csv, json, numpy as np, time
sys.path.insert(0, "/root/otto/XiaoanNew")
from modules.config.settings import settings
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.cv.image_utils import combine_masks
from modules.experiment.scoring import ScoringEngine
from PIL import Image
from collections import Counter

seg = load_yolov8_seg(settings.YOLO_WEIGHTS, device=settings.INFERENCE_DEVICE)
scoring = ScoringEngine.from_yaml("/root/otto/XiaoanNew/assets/configs/scoring_new4d_gs_best.yaml")
gt = {x["id"]: x for x in json.load(open("/root/otto/XiaoanNew/data/benchmark/benchmark_v4/fourdim_gt_v4.json"))}

rows = list(csv.DictReader(open("/root/otto/XiaoanNew/outputs/benchmark_output/v4/baseline_qwen36/results.csv", encoding="utf-8-sig", errors="replace")))
fieldnames = list(rows[0].keys())
print(f"总: {len(rows)}")

def calc_stats(rows):
    tp=tn=fp=fn=0
    for r in rows:
        g=r.get("gt",""); p=r.get("pred","")
        if g=="yes":
            if p=="yes":tp+=1
            else:fn+=1
        else:
            if p=="no":tn+=1
            else:fp+=1
    n=tp+tn+fp+fn
    return f"Acc={ (tp+tn)/n:.3f} VR={tn/(tn+fp) if tn+fp else 0:.3f} TP{tp}/TN{tn}/FP{fp}/FN{fn}"

def dim_acc(rows, dim, pk):
    c=t=0
    for r in rows:
        rid=r.get("id","")
        if rid not in gt: continue
        g=gt[rid].get(dim,""); pv=r.get(pk,"")
        if g and pv and not pv.startswith("0.") and pv not in ("err","parse_fail"):
            t+=1
            if g==pv:c+=1
    return f"{c}/{t}={c/t if t else 0:.3f}"

print(f"原始: {calc_stats(rows)}")
print(f"原始 angle: {dim_acc(rows,'angle','angle_status')}")

def pca_2d_image(mask):
    """2D图像PCA: 在mask像素坐标(u,v)上做PCA"""
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return None, None
    uv = np.column_stack([xs, ys]).astype(float)
    c = uv.mean(axis=0)
    U, S, Vt = np.linalg.svd(uv - c, full_matrices=False)
    return c, Vt[0]

def angle_2d(v1, v2):
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def get_centermost_bike(objects, W, H):
    cx, cy = W/2.0, H/2.0
    best, bd = None, float("inf")
    for obj in objects:
        if obj["label"] != "Electric bike": continue
        bx1,by1,bx2,by2 = obj["bbox"]
        d = ((bx1+bx2)/2-cx)**2 + ((by1+by2)/2-cy)**2
        if d < bd: bd, best = d, obj
    return best

n_cv=0; n_fail=0; t0=time.time()
for i, r in enumerate(rows):
    rid=r.get("id","")
    g=gt.get(rid,{})
    src=g.get("src","")
    if not src:
        n_fail+=1; continue
    img_path=os.path.join("/root/otto/XiaoanNew/data", src)
    if not os.path.exists(img_path):
        for ext in (".JPEG",".jpeg",".png"):
            alt=os.path.splitext(img_path)[0]+ext
            if os.path.exists(alt): img_path=alt; break
    if not os.path.exists(img_path):
        n_fail+=1; continue
    try:
        img_bytes=open(img_path,"rb").read()
        pil=Image.open(img_path).convert("RGB")
        W,H=pil.size
        result=seg.predict(img_bytes, visual=False, retina_masks=True, max_input_size=None)
        objects=result["objects"]

        # 最居中车
        bike_obj=get_centermost_bike(objects, W, H)
        if bike_obj is None or bike_obj.get("mask") is None:
            n_fail+=1; continue
        bike_mask=bike_obj["mask"]

        # 参照线
        parking_mask=combine_masks(objects,"parking lane")
        curb_mask=combine_masks(objects,"Curb")
        ref_mask=parking_mask if parking_mask is not None and parking_mask.sum()>50 else curb_mask
        if ref_mask is None or ref_mask.sum()<50:
            n_fail+=1; continue

        # 2D图像PCA(只用mask像素坐标,不用深度)
        bc,ba=pca_2d_image(bike_mask)
        rc,ra=pca_2d_image(ref_mask)
        if ba is None or ra is None:
            n_fail+=1; continue

        ang=angle_2d(ba, ra)
        if ang>90: ang=180-ang
        cv_angle="[合规]" if 60<=ang<=90 else "[不合规-斜停]"

        r["angle_status"]=cv_angle
        pos=r.get("position_status",""); med=r.get("medium_status",""); st=r.get("state_status","[正立]")
        if pos and med and cv_angle and st and not pos.startswith("0.") and pos not in ("err","parse_fail"):
            sr=scoring.score(pos,med,cv_angle,st)
            r["pred"]="yes" if sr.is_compliant else "no"
            r["final_score"]=str(sr.final_score)
        n_cv+=1
    except Exception as e:
        n_fail+=1
    if (i+1)%200==0:
        print(f"  {i+1}/{len(rows)} CV覆盖{n_cv} 失败{n_fail} ({time.time()-t0:.0f}s)")

print(f"\nCV覆盖(2D图像PCA): {n_cv}, 保留VLM: {n_fail}, 耗时{time.time()-t0:.0f}s")
print(f"覆盖后: {calc_stats(rows)}")
print(f"覆盖后 angle: {dim_acc(rows,'angle','angle_status')}")

with open("/root/otto/XiaoanNew/outputs/benchmark_output/v4/baseline_qwen36/results_cv_2d_angle.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)
print("已保存 results_cv_2d_angle.csv")
