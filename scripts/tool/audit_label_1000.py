"""bike_audit_1000 半自动预标注脚本

对 data/raw_collected/bike_audit_1000/ 的 1000 张申诉样本做：
  1. 场景分类（standard / longtail）：复用 scene_classify.py 两段式逻辑
     - YOLO 几何短路（无主车/车过小 → longtail，省 VLM）
     - VLM 视觉精判（要素清晰 + 车完整 + 明显车道线 → standard）
  2. 合规预测（pred_compliance=yes/no）：复用生产判断流程
     - YOLOv8-Seg 分割 + 几何计算
     - 主车选取：距画面中心最近（与 app.py 一致）
     - cv_enhanced_p5 prompt + scoring_optimized_cv_p4 打分
  3. 输出到 data/labeling/bike_audit_1000/ （严禁并进 compliance/by_scene）
     - manifest.csv：image, scene, pred_compliance, final_score, vlm_reason, car_return_tag
     - thumbnails/<image>：≤1280px 缩略图，供复核界面

用法：
    # 冒烟测试（10 张）：
    uv run python scripts/tool/audit_label_1000.py --smoke

    # 全量：
    uv run python scripts/tool/audit_label_1000.py

    # 自定义数量和并发：
    uv run python scripts/tool/audit_label_1000.py --limit 100 --workers 6
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from modules.config.settings import settings
from modules.cv.image_utils import (
    calculate_iou_and_overlap,
    combine_masks,
    draw_wireframe_visual,
    encode_image_to_base64,
)
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.experiment.scoring import ScoringEngine
from modules.prompt.manager import load_prompt
from modules.vlm.client import create_client_pool
from modules.vlm.parser import parse_vlm_response
from modules.vlm.retry import chat_completion_with_retry

# ================================================================
# 路径配置
# ================================================================
_SRC_DIR   = os.path.join(_PROJECT_ROOT, "data/raw_collected/bike_audit_1000")
_SRC_MANIFEST = os.path.join(_SRC_DIR, "manifest.csv")
_OUT_DIR   = os.path.join(_PROJECT_ROOT, "data/labeling/bike_audit_1000")
_THUMB_DIR = os.path.join(_OUT_DIR, "thumbnails")
_OUT_MANIFEST = os.path.join(_OUT_DIR, "manifest.csv")

_SCORING_CONFIG = os.path.join(_PROJECT_ROOT, "assets/configs/scoring_optimized_cv_p4.yaml")
_PROMPT_ID      = "cv_enhanced_p5"

_THUMB_MAX = 1280  # 缩略图最长边

# ================================================================
# 场景分类 Prompt（与 scene_classify.py 保持一致）
# ================================================================
_SCENE_PROMPT = (
    "你是共享单车停放图像的场景分类器。请判断这张图属于哪一类：\n"
    "- standard（标准场景）：必须同时满足三点：(1) 画面要素清晰，无明显过曝/过暗/模糊；"
    "(2) 电动车完整，车把、车座、前后轮基本都在画面内；(3) 有明显可辨认的停车线或车道线。\n"
    "- longtail（复杂长尾场景）：上述任意一条不满足，例如停车线磨损/被遮挡/缺失、"
    "夜间逆光、车辆残缺或被截断、拍摄视角极端、密集停放遮挡等。\n"
    "CV 辅助信息：检测到停车线={lane}，检测到路缘={curb}，主车面积占比={area:.3f}。\n"
    "只输出一个英文单词：standard 或 longtail。"
)

# ================================================================
# 全局模型（在模块级加载一次，多线程共享）
# ================================================================
print("[audit_label_1000] 加载 YOLOv8-Seg 模型...")
_segmentor = load_yolov8_seg(settings.YOLO_WEIGHTS, device=settings.INFERENCE_DEVICE)
_segmentor.conf_threshold = 0.25  # 场景分类阶段用低阈值，合规判断阶段仍用此分割结果
_seg_lock = threading.Lock()

_scoring_engine = ScoringEngine.from_yaml(_SCORING_CONFIG)
_vlm_clients = create_client_pool()

print(f"[audit_label_1000] VLM 客户端: {len(_vlm_clients)} 个")


# ================================================================
# 工具函数
# ================================================================

def _pick_client(path: str):
    """按文件名哈希轮询客户端，分散请求压力"""
    return _vlm_clients[abs(hash(path)) % len(_vlm_clients)]


def _make_thumbnail(src_path: str, fname: str) -> None:
    """生成 ≤1280px 缩略图，保存到 _THUMB_DIR"""
    try:
        with Image.open(src_path) as img:
            img.thumbnail((_THUMB_MAX, _THUMB_MAX), Image.LANCZOS)
            img.save(os.path.join(_THUMB_DIR, fname), "JPEG", quality=85)
    except Exception as e:
        print(f"[缩略图] {fname} 生成失败: {e}")


# ================================================================
# 核心处理逻辑
# ================================================================

def process_image(task: dict) -> dict:
    """
    对单张图片执行：场景分类 + 合规预测。

    Args:
        task: {filename, image_path, car_return_tag}

    Returns:
        {filename, scene, scene_by, pred_compliance, final_score,
         vlm_reason, car_return_tag, latency, error}
    """
    fname      = task["filename"]
    image_path = task["image_path"]
    car_tag    = task["car_return_tag"]
    start_t    = time.time()

    result = {
        "filename":        fname,
        "scene":           "longtail",
        "scene_by":        "err",
        "pred_compliance": "error",
        "final_score":     0.0,
        "vlm_reason":      "",
        "car_return_tag":  car_tag,
        "latency":         0.0,
        "error":           "",
    }

    try:
        # ----------------------------------------------------------
        # 阶段 1：YOLOv8-Seg 分割（场景分类和合规判断共用一次推理）
        # ----------------------------------------------------------
        with _seg_lock:
            seg_result = _segmentor.predict(image_path, visual=False, retina_masks=True)

        objects = seg_result["objects"]
        raw_img = seg_result["image_raw"]
        H, W    = seg_result["image_size"]

        # 统计类别
        class_counts = {"Electric bike": 0, "Curb": 0, "parking lane": 0, "Tactile paving": 0}
        for obj in objects:
            if obj["label"] in class_counts:
                class_counts[obj["label"]] += 1

        lane  = any(o["label"] == "parking lane" and o["confidence"] > 0.3 for o in objects)
        curb  = any(o["label"] == "Curb" for o in objects)
        areas = [o.get("area_ratio", 0.0) for o in objects if o["label"] == "Electric bike"]
        area  = max(areas, default=0.0)

        # ----------------------------------------------------------
        # 阶段 1a：场景分类短路（无车或车过小 → longtail，省 VLM）
        # ----------------------------------------------------------
        if area < 0.03:
            result["scene"]    = "longtail"
            result["scene_by"] = "yolo"
        else:
            # 场景分类 VLM 精判
            try:
                b64_raw = encode_image_to_base64(raw_img, (768, 768), 80)
                client  = _pick_client(fname + "_scene")
                res = chat_completion_with_retry(
                    client,
                    model=settings.VLM_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": _SCENE_PROMPT.format(lane=lane, curb=curb, area=area)},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64_raw}"}},
                        ],
                    }],
                    max_tokens=50,
                    temperature=0.0,
                )
                ans = (res.choices[0].message.content or "").strip().lower()
                result["scene"]    = "standard" if "standard" in ans else "longtail"
                result["scene_by"] = "vlm"
            except Exception as se:
                result["scene"]    = "longtail"
                result["scene_by"] = f"err:{type(se).__name__}"

        # ----------------------------------------------------------
        # 阶段 2：合规预测（复用生产流程）
        # ----------------------------------------------------------

        # 主车选取：距画面中心最近（与 app.py 保持一致）
        img_cx, img_cy = W / 2.0, H / 2.0
        main_bike_mask, main_bike_dist = None, float("inf")
        cv_detections = []
        for obj in objects:
            cv_detections.append({
                "id": obj["id"], "label": obj["label"],
                "confidence": obj["confidence"], "bbox": obj["bbox"],
            })
            if obj["label"] == "Electric bike":
                bx1, by1, bx2, by2 = obj["bbox"]
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                d = (bcx - img_cx) ** 2 + (bcy - img_cy) ** 2
                if d < main_bike_dist:
                    main_bike_dist = d
                    main_bike_mask = obj.get("mask")

        # 几何指标计算
        geo = {
            "main_vehicle_detected": main_bike_mask is not None,
            "overlap_with_parking_lane":  0.0,
            "iou_with_parking_lane":      0.0,
            "overlap_with_tactile_paving": 0.0,
            "status_inference": "unknown",
        }
        if main_bike_mask is not None:
            p_mask = combine_masks(objects, "parking lane")
            if p_mask is not None:
                iou, overlap = calculate_iou_and_overlap(main_bike_mask, p_mask)
                geo["iou_with_parking_lane"]     = iou
                geo["overlap_with_parking_lane"] = overlap
            t_mask = combine_masks(objects, "Tactile paving")
            if t_mask is not None:
                _, ov_t = calculate_iou_and_overlap(main_bike_mask, t_mask)
                geo["overlap_with_tactile_paving"] = ov_t
            if geo["overlap_with_parking_lane"] > 0.8:
                geo["status_inference"] = "Likely Compliant (High Overlap)"
            elif geo["overlap_with_parking_lane"] < 0.1:
                geo["status_inference"] = "Likely Out of Bounds"

        # 组装 Prompt（与 run_benchmark_v1.py / app.py 一致）
        detection_info = {
            "image_size":       [H, W],
            "detected_objects": cv_detections,
            "class_summary":    class_counts,
            "geometry_analysis": geo,
        }
        full_prompt = (
            load_prompt(_PROMPT_ID)
            + "\n\n# YOLOv8-Seg Detection & Geometry Analysis\n```json\n"
            + json.dumps(detection_info, ensure_ascii=False, indent=2)
            + "\n```"
        )

        # 线框图编码
        vis_img = draw_wireframe_visual(raw_img, objects)
        b64_raw_comp = encode_image_to_base64(raw_img, (768, 768), 80)
        b64_vis      = encode_image_to_base64(vis_img, (768, 768), 80)

        # VLM 合规调用
        client = _pick_client(fname + "_comp")
        resp = chat_completion_with_retry(
            client,
            model=settings.VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",      "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_raw_comp}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_vis}"}},
                ],
            }],
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
        )
        vlm_text   = resp.choices[0].message.content or ""
        vlm_result = parse_vlm_response(vlm_text)

        if not vlm_result.is_valid:
            result["pred_compliance"] = "parse_error"
            result["vlm_reason"]      = vlm_result.parse_error[:200]
        else:
            sr = _scoring_engine.score(*vlm_result.statuses)
            result["pred_compliance"] = "yes" if sr.is_compliant else "no"
            result["final_score"]     = round(sr.final_score, 4)
            result["vlm_reason"]      = str(vlm_result.reason)[:300]

    except Exception as e:
        import traceback
        traceback.print_exc()
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    result["latency"] = round(time.time() - start_t, 3)

    # 生成缩略图（不阻塞主流程，失败静默）
    _make_thumbnail(image_path, fname)

    return result


# ================================================================
# 主流程
# ================================================================

def load_src_manifest(path: str) -> dict[str, str]:
    """加载源 manifest，返回 {image_file: car_return_tag}"""
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["image_file"]] = row.get("car_return_tag", "")
    return mapping


def main():
    """主入口"""
    ap = argparse.ArgumentParser(description="bike_audit_1000 半自动预标注")
    ap.add_argument("--smoke",   action="store_true", help="冒烟模式：只跑前 10 张")
    ap.add_argument("--limit",   type=int, default=0,  help="只跑前 N 张")
    ap.add_argument("--workers", type=int, default=6,  help="并发线程数")
    args = ap.parse_args()

    # 构建任务列表
    car_tag_map = load_src_manifest(_SRC_MANIFEST)
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
    all_images = sorted([
        f for f in os.listdir(_SRC_DIR)
        if os.path.splitext(f)[1] in exts
    ])
    limit = 10 if args.smoke else (args.limit if args.limit > 0 else len(all_images))
    images = all_images[:limit]
    print(f"[audit_label_1000] 待处理: {len(images)} 张 (总计 {len(all_images)} 张)")

    tasks = [
        {
            "filename":      fname,
            "image_path":    os.path.join(_SRC_DIR, fname),
            "car_return_tag": car_tag_map.get(fname, ""),
        }
        for fname in images
    ]

    # 创建输出目录
    os.makedirs(_OUT_DIR,   exist_ok=True)
    os.makedirs(_THUMB_DIR, exist_ok=True)

    # 并行处理
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_image, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="预标注"):
            results.append(fut.result())

    # 按原始顺序排序
    fname_order = {f: i for i, f in enumerate(images)}
    results.sort(key=lambda r: fname_order.get(r["filename"], 9999))

    # 写 manifest
    headers = ["filename", "scene", "scene_by", "pred_compliance",
               "final_score", "vlm_reason", "car_return_tag", "latency", "error"]
    with open(_OUT_MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # 统计
    total = len(results)
    std_cnt   = sum(1 for r in results if r["scene"] == "standard")
    long_cnt  = sum(1 for r in results if r["scene"] == "longtail")
    yes_cnt   = sum(1 for r in results if r["pred_compliance"] == "yes")
    no_cnt    = sum(1 for r in results if r["pred_compliance"] == "no")
    err_cnt   = sum(1 for r in results if r["error"])
    vlm_scene = sum(1 for r in results if r["scene_by"] == "vlm")
    yolo_sc   = sum(1 for r in results if r["scene_by"] == "yolo")

    print("\n==================== 预标注统计 ====================")
    print(f"  总计:      {total} 张")
    print(f"  场景:      standard={std_cnt}  longtail={long_cnt}")
    print(f"  合规预测:  yes={yes_cnt}  no={no_cnt}  error={err_cnt}")
    print(f"  场景判定:  VLM精判={vlm_scene}  YOLO短路={yolo_sc}")
    print(f"  manifest:  {_OUT_MANIFEST}")
    print(f"  缩略图:    {_THUMB_DIR}/")

    if args.smoke:
        print("\n==================== 冒烟样例 (前5行) ====================")
        for r in results[:5]:
            print(
                f"  {r['filename']} | scene={r['scene']}({r['scene_by']}) "
                f"| pred={r['pred_compliance']} score={r['final_score']} "
                f"| car_tag={r['car_return_tag']} | {r['latency']}s"
            )
        print("\n[smoke] 完成，请确认输出合理后用 --全量 跑完整批次")


if __name__ == "__main__":
    main()
