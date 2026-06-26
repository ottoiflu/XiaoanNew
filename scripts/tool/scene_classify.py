"""按场景对 compliance 评测图片分类：标准场景 vs 复杂长尾场景。

两段式判定：
1. YOLOv8-Seg 出几何事实（是否检出停车线、主车面积占比）。无车/车过小直接判长尾，省 VLM 调用。
2. 其余图片带 CV 事实交 VLM 视觉精判（要素清晰 + 车辆完整 + 明显车道线）。

保留 yes/no（合规/违规）维度，叠加场景维度，输出到 by_scene/{scene}/{yes|no}/，并写 manifest.csv。
原图不动，非破坏性。
"""

import argparse
import collections
import csv
import glob
import os
import shutil
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from modules.config.settings import settings
from modules.cv.image_utils import encode_image_to_base64
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.vlm.client import create_client_pool
from modules.vlm.retry import chat_completion_with_retry

ROOT = "/root/otto/XiaoanNew/data/compliance"
# 源目录 -> 合规标签（跳过 yes_val/no_val，它们是 _all 的子集）
FOLDERS = {
    "yes_val_all": "yes",
    "positive_extra": "yes",
    "no_val_all": "no",
    "negative_extra": "no",
}
OUT = os.path.join(ROOT, "by_scene")
MODEL = settings.VLM_MODEL

PROMPT = """你是共享单车停放图像的场景分类器。请判断这张图属于哪一类：
- standard（标准场景）：必须同时满足三点：(1) 画面要素清晰，无明显过曝/过暗/模糊；(2) 电动车完整，车把、车座、前后轮基本都在画面内；(3) 有明显可辨认的停车线或车道线。
- longtail（复杂长尾场景）：上述任意一条不满足，例如停车线磨损/被遮挡/缺失、夜间逆光、车辆残缺或被截断、拍摄视角极端、密集停放遮挡等。
CV 辅助信息：检测到停车线={lane}，检测到路缘={curb}，主车面积占比={area:.3f}。
只输出一个英文单词：standard 或 longtail。"""

seg = load_yolov8_seg(settings.YOLO_WEIGHTS, device=settings.INFERENCE_DEVICE)
seg_lock = threading.Lock()
clients = create_client_pool()


def classify(task):
    path, folder, comp = task
    try:
        with seg_lock:
            r = seg.predict(path, conf=0.25, retina_masks=False, visual=False)
        objs = r["objects"]
        lane = any(o["label"] == "parking lane" and o["confidence"] > 0.3 for o in objs)
        curb = any(o["label"] == "Curb" for o in objs)
        area = max([o["area_ratio"] for o in objs if o["label"] == "Electric bike"], default=0.0)
        # 短路：无车或车过小（不完整/太远）→ 长尾
        if area < 0.03:
            return (path, folder, comp, "longtail", lane, area, "yolo")
        b64 = encode_image_to_base64(r["image_raw"], (768, 768), 80)
        client = clients[abs(hash(path)) % len(clients)]
        res = chat_completion_with_retry(
            client,
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT.format(lane=lane, curb=curb, area=area)},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            max_tokens=50,
            temperature=0.0,
        )
        ans = (res.choices[0].message.content or "").strip().lower()
        scene = "standard" if "standard" in ans else "longtail"
        return (path, folder, comp, scene, lane, area, "vlm")
    except Exception as e:  # noqa: BLE001 - 分类失败兜底为长尾并记录
        return (path, folder, comp, "longtail", False, 0.0, f"err:{type(e).__name__}")


def main():
    ap = argparse.ArgumentParser(description="按场景分类 compliance 图片")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（冒烟测试用）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tasks = []
    for folder, comp in FOLDERS.items():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            for p in glob.glob(os.path.join(ROOT, folder, ext)):
                tasks.append((p, folder, comp))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f">>> 待分类图片: {len(tasks)}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in tqdm(ex.map(classify, tasks), total=len(tasks), desc="分类中"):
            results.append(r)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "source_folder", "compliance", "scene", "has_lane", "bike_area", "by"])
        for path, folder, comp, scene, lane, area, by in results:
            w.writerow([os.path.basename(path), folder, comp, scene, lane, round(area, 4), by])

    cnt = collections.Counter()
    for path, folder, comp, scene, _, _, _ in results:
        dst = os.path.join(OUT, scene, comp)
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(path, os.path.join(dst, f"{folder}__{os.path.basename(path)}"))
        cnt[(scene, comp)] += 1

    print("\n==================== 场景分布 ====================")
    total = sum(cnt.values())
    for scene in ("standard", "longtail"):
        for comp in ("yes", "no"):
            print(f"  {scene:9s} / {comp:3s} : {cnt[(scene, comp)]}")
    print(f"  合计: {total}")
    errs = [r for r in results if r[6].startswith("err")]
    vlm_n = sum(1 for r in results if r[6] == "vlm")
    print(f"  VLM 精判: {vlm_n} | YOLO 短路: {total - vlm_n - len(errs)} | 错误: {len(errs)}")
    print(f"  manifest: {os.path.join(OUT, 'manifest.csv')}")


if __name__ == "__main__":
    main()
