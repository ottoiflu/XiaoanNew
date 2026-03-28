"""优化版批量对比实验运行器

三层缓存架构：
  1. YOLO 推理 + 图像编码预计算 (一次推理，所有 CV 实验复用)
  2. VLM 调用去重 (同 prompt 只调一次，不同评分离线重算)
  3. 原始四维状态持久化 (支持任意评分方式重放)

用法:
    uv run scripts/run_contrast_batch_v2.py
    uv run scripts/run_contrast_batch_v2.py --experiments 0,1,2
    uv run scripts/run_contrast_batch_v2.py --list
    uv run scripts/run_contrast_batch_v2.py --replay outputs/contrast_experiments/vlm_cache_xxx.json --scoring assets/configs/scoring_optimized_cv_p4.yaml
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

from PIL import Image
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from modules.config.settings import get_settings
from modules.cv.image_utils import (
    calculate_iou_and_overlap,
    combine_masks,
    draw_wireframe_visual,
    encode_image_to_base64,
)
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.experiment.io import (
    ResultWriter,
    collect_image_tasks,
    load_all_labels,
)
from modules.experiment.metrics import calculate_metrics, print_metrics_report
from modules.experiment.scoring import ScoringEngine
from modules.prompt.manager import load_prompt
from modules.vlm.client import create_client_pool
from modules.vlm.parser import VLMResult, normalize_label, parse_vlm_response
from modules.vlm.retry import chat_completion_with_retry

# ================= 全局常量 =================

MODEL = "qwen/qwen3-vl-30b-a3b-instruct"
MAX_SIZE = (768, 768)
QUALITY = 80
SCORING_DEFAULT = "assets/configs/scoring_default.yaml"
SCORING_OPT = "assets/configs/scoring_optimized_cv_p4.yaml"

DATA_FOLDERS = [
    os.path.join(_PROJECT_ROOT, "data/Compliance_test_data/no_val"),
    os.path.join(_PROJECT_ROOT, "data/Compliance_test_data/yes_val"),
]

OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs/contrast_experiments")
SEGMENTOR_WEIGHTS = os.path.join(_PROJECT_ROOT, "assets/weights/best.pt")

# ================= 实验矩阵 =================

EXPERIMENTS = [
    # --- 纯VLM 基线 (veto) ---
    {"name": "vlm_p6_veto", "mode": "pure_vlm", "prompt_id": "standard_p6", "scoring": None},
    {"name": "vlm_p5_veto", "mode": "pure_vlm", "prompt_id": "standard_p5", "scoring": None},
    {"name": "vlm_p4_veto", "mode": "pure_vlm", "prompt_id": "standard_p4", "scoring": None},
    {"name": "vlm_p4_1_veto", "mode": "pure_vlm", "prompt_id": "standard_p4_1", "scoring": None},
    {"name": "vlm_p4_2_veto", "mode": "pure_vlm", "prompt_id": "standard_p4_2", "scoring": None},
    # --- VLM+CV (veto) ---
    {"name": "cv_p6_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p6", "scoring": None},
    {"name": "cv_p5_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p5", "scoring": None},
    {"name": "cv_p4_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4", "scoring": None},
    {"name": "cv_p4_1_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4_1", "scoring": None},
    {"name": "cv_p4_2_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4_2", "scoring": None},
    {"name": "cv_p4_3_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4_3", "scoring": None},
    {"name": "cv_p7_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p7", "scoring": None},
    # --- 纯VLM 加权 ---
    {"name": "vlm_p6_weighted", "mode": "pure_vlm", "prompt_id": "standard_p6", "scoring": SCORING_DEFAULT},
    {"name": "vlm_p5_weighted", "mode": "pure_vlm", "prompt_id": "standard_p5", "scoring": SCORING_DEFAULT},
    {"name": "vlm_p4_weighted", "mode": "pure_vlm", "prompt_id": "standard_p4", "scoring": SCORING_DEFAULT},
    {"name": "vlm_p4_opt_weighted", "mode": "pure_vlm", "prompt_id": "standard_p4", "scoring": SCORING_OPT},
    # --- VLM+CV 加权 ---
    {"name": "cv_p6_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p6", "scoring": SCORING_DEFAULT},
    {"name": "cv_p5_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p5", "scoring": SCORING_DEFAULT},
    {"name": "cv_p4_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4", "scoring": SCORING_DEFAULT},
    {"name": "cv_p4_opt_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4", "scoring": SCORING_OPT},
    {"name": "cv_p4_1_opt_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4_1", "scoring": SCORING_OPT},
    {"name": "cv_p4_2_opt_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4_2", "scoring": SCORING_OPT},
    {"name": "cv_p4_3_opt_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4_3", "scoring": SCORING_OPT},
    {"name": "cv_p7_opt_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p7", "scoring": SCORING_OPT},
    # --- 补充缺失对称实验 ---
    {"name": "vlm_p7_veto", "mode": "pure_vlm", "prompt_id": "standard_p7", "scoring": None},
    {"name": "vlm_p7_weighted", "mode": "pure_vlm", "prompt_id": "standard_p7", "scoring": SCORING_DEFAULT},
    {"name": "vlm_p4_3_veto", "mode": "pure_vlm", "prompt_id": "standard_p4_3", "scoring": None},
    {"name": "vlm_p4_3_opt_weighted", "mode": "pure_vlm", "prompt_id": "standard_p4_3", "scoring": SCORING_OPT},
    # --- p8 实验组 ---
    {"name": "vlm_p8_veto", "mode": "pure_vlm", "prompt_id": "standard_p8", "scoring": None},
    {"name": "vlm_p8_opt_weighted", "mode": "pure_vlm", "prompt_id": "standard_p8", "scoring": SCORING_OPT},
    {"name": "cv_p8_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p8", "scoring": None},
    {"name": "cv_p8_opt_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p8", "scoring": SCORING_OPT},
]

CSV_HEADERS = [
    "image",
    "folder",
    "pred",
    "gt",
    "composition",
    "angle",
    "distance",
    "context",
    "reason",
    "latency",
    "weighted_score",
]


# ================= 阶段1: 预计算缓存 =================


def precompute_yolo_cache(image_tasks, segmentor, vis_dir):
    """对所有图片执行一次 YOLO 推理，缓存分割结果和编码图像。"""
    cache = {}
    print(f"\n>>> [预计算] YOLO 推理 + 图像编码，共 {len(image_tasks)} 张")
    os.makedirs(vis_dir, exist_ok=True)

    for img_name, folder in tqdm(image_tasks, desc="[YOLO预计算]"):
        image_path = os.path.join(folder, img_name)
        key = (img_name, folder)
        try:
            seg_result = segmentor.predict(image_path)
            raw_img = seg_result["image_raw"]
            objects = seg_result["objects"]
            h_val, w_val = seg_result["image_size"]

            vis_img = draw_wireframe_visual(raw_img, objects)
            vis_path = os.path.join(vis_dir, img_name)
            Image.fromarray(vis_img).save(vis_path)

            b64_raw = encode_image_to_base64(raw_img, MAX_SIZE, QUALITY)
            b64_vis = encode_image_to_base64(vis_img, MAX_SIZE, QUALITY)

            class_counts = {
                "Electric bike": 0,
                "Curb": 0,
                "parking lane": 0,
                "Tactile paving": 0,
            }
            det_objects = []
            main_bike_mask, main_bike_conf = None, -1

            for obj in objects:
                det_objects.append(
                    {
                        "id": obj["id"],
                        "label": obj["label"],
                        "confidence": obj["confidence"],
                        "bbox": obj["bbox"],
                    }
                )
                if obj["label"] in class_counts:
                    class_counts[obj["label"]] += 1
                if obj["label"] == "Electric bike" and obj["confidence"] > main_bike_conf:
                    main_bike_conf = obj["confidence"]
                    main_bike_mask = obj.get("mask")

            geo = {
                "main_vehicle_detected": False,
                "overlap_with_parking_lane": 0.0,
                "iou_with_parking_lane": 0.0,
                "overlap_with_tactile_paving": 0.0,
            }
            if main_bike_mask is not None:
                geo["main_vehicle_detected"] = True
                parking_mask = combine_masks(objects, "parking lane")
                if parking_mask is not None:
                    iou, overlap = calculate_iou_and_overlap(main_bike_mask, parking_mask)
                    geo["iou_with_parking_lane"] = iou
                    geo["overlap_with_parking_lane"] = overlap
                tactile_mask = combine_masks(objects, "Tactile paving")
                if tactile_mask is not None:
                    _, overlap_t = calculate_iou_and_overlap(main_bike_mask, tactile_mask)
                    geo["overlap_with_tactile_paving"] = overlap_t

            detection_info = {
                "image_size": [h_val, w_val],
                "detected_objects": det_objects,
                "class_summary": class_counts,
                "geometry_analysis": geo,
            }

            cache[key] = {
                "b64_raw": b64_raw,
                "b64_vis": b64_vis,
                "detection_info": detection_info,
            }
        except Exception as e:
            print(f"  [WARN] {img_name}: {e}")
            cache[key] = None

    valid = sum(1 for v in cache.values() if v is not None)
    print(f">>> [预计算完成] 成功 {valid}/{len(image_tasks)}")
    return cache


def precompute_vlm_images(image_tasks):
    """纯VLM模式：只做 base64 编码缓存。"""
    cache = {}
    print(f"\n>>> [预计算] 纯VLM 图像编码，共 {len(image_tasks)} 张")
    for img_name, folder in tqdm(image_tasks, desc="[编码]"):
        key = (img_name, folder)
        image_path = os.path.join(folder, img_name)
        try:
            cache[key] = encode_image_to_base64(
                image_path,
                max_size=MAX_SIZE,
                quality=QUALITY,
            )
        except Exception as e:
            print(f"  [WARN] {img_name}: {e}")
            cache[key] = None
    return cache


# ================= 阶段2: VLM 调用 (按 prompt group 去重) =================


def _call_vlm_pure(args):
    """纯VLM 单图 VLM 调用"""
    img_name, folder, client, prompt_text, b64_img = args
    start_t = time.time()
    try:
        res = chat_completion_with_retry(
            client,
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                        },
                    ],
                }
            ],
            max_tokens=600,
            temperature=0.1,
        )
        vlm_out = res.choices[0].message.content
        vlm_result = parse_vlm_response(vlm_out)
        latency = round(time.time() - start_t, 3)
        return (img_name, folder, vlm_result, latency)
    except Exception as e:
        return (
            img_name,
            folder,
            VLMResult(parse_error=str(e)),
            round(time.time() - start_t, 3),
        )


def _call_vlm_cv(args):
    """VLM+CV 单图 VLM 调用 (使用预计算的 YOLO 缓存)"""
    img_name, folder, client, prompt_text, cv_cache_entry, strip_geo = args
    start_t = time.time()
    try:
        b64_raw = cv_cache_entry["b64_raw"]
        b64_vis = cv_cache_entry["b64_vis"]
        detection_info = cv_cache_entry["detection_info"]

        if strip_geo:
            info_copy = {
                "image_size": detection_info["image_size"],
                "detected_objects": [
                    {"label": o["label"], "confidence": round(o["confidence"], 2)}
                    for o in detection_info["detected_objects"]
                ],
                "class_summary": detection_info["class_summary"],
                "geometry_analysis": {
                    "main_vehicle_detected": detection_info["geometry_analysis"]["main_vehicle_detected"],
                },
            }
        else:
            info_copy = detection_info

        structured_info = json.dumps(info_copy, ensure_ascii=False, indent=2)
        full_prompt = (
            prompt_text + "\n\n# YOLOv8-Seg Detection & Geometry Analysis\n```json\n" + structured_info + "\n```"
        )

        res = chat_completion_with_retry(
            client,
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_raw}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_vis}"},
                        },
                    ],
                }
            ],
            max_tokens=1000,
            temperature=0.1,
            top_p=0.9,
        )
        vlm_out = res.choices[0].message.content
        vlm_result = parse_vlm_response(vlm_out)
        latency = round(time.time() - start_t, 3)
        return (img_name, folder, vlm_result, latency)
    except Exception as e:
        return (
            img_name,
            folder,
            VLMResult(parse_error=str(e)),
            round(time.time() - start_t, 3),
        )


def run_vlm_group(
    group_key,
    image_tasks,
    clients,
    prompt_text,
    cv_cache,
    vlm_img_cache,
    max_workers,
    strip_geo=False,
):
    """对一个 (mode, prompt_id) 组执行 VLM 调用。"""
    mode = group_key[0]
    results = {}

    if mode == "pure_vlm":
        tasks = []
        for i, (img_name, folder) in enumerate(image_tasks):
            b64 = vlm_img_cache.get((img_name, folder))
            if b64 is None:
                results[(img_name, folder)] = (
                    VLMResult(parse_error="编码失败"),
                    0.0,
                )
                continue
            client = clients[i % len(clients)]
            tasks.append((img_name, folder, client, prompt_text, b64))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        ) as executor:
            for r in tqdm(
                executor.map(_call_vlm_pure, tasks),
                total=len(tasks),
                desc=f"[VLM:{group_key[1]}]",
            ):
                results[(r[0], r[1])] = (r[2], r[3])
    else:
        tasks = []
        for i, (img_name, folder) in enumerate(image_tasks):
            entry = cv_cache.get((img_name, folder))
            if entry is None:
                results[(img_name, folder)] = (
                    VLMResult(parse_error="YOLO预计算失败"),
                    0.0,
                )
                continue
            client = clients[i % len(clients)]
            tasks.append(
                (img_name, folder, client, prompt_text, entry, strip_geo),
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
        ) as executor:
            for r in tqdm(
                executor.map(_call_vlm_cv, tasks),
                total=len(tasks),
                desc=f"[VLM:{group_key[1]}]",
            ):
                results[(r[0], r[1])] = (r[2], r[3])

    return results


# ================= 阶段3: 评分重放 + 结果写入 =================


def _build_scoring_engine(scoring_path):
    """构建评分引擎"""
    if scoring_path:
        full_path = os.path.join(_PROJECT_ROOT, scoring_path) if not os.path.isabs(scoring_path) else scoring_path
        return ScoringEngine.from_yaml(full_path)
    return None


def _judge(vlm_result, scoring_engine):
    """根据评分模式进行合规判定"""
    comp, ang, dist, cont = vlm_result.statuses
    if scoring_engine:
        sr = scoring_engine.score(comp, ang, dist, cont)
        return ("yes" if sr.is_compliant else "no"), sr.final_score
    return ScoringEngine.veto_judge(comp, ang, dist, cont), 0.0


def evaluate_and_write(exp_config, vlm_results, global_labels, image_tasks):
    """使用 VLM 原始结果 + 指定评分方式生成实验报告。"""
    exp_name = exp_config["name"]
    scoring_engine = _build_scoring_engine(exp_config["scoring"])
    scoring_label = "weighted" if scoring_engine else "veto"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_ROOT, f"{timestamp}_{exp_name}")
    os.makedirs(exp_dir, exist_ok=True)
    out_csv = os.path.join(exp_dir, f"{exp_name}.csv")

    final_rows = []
    with ResultWriter(out_csv, CSV_HEADERS) as writer:
        for img_name, folder in image_tasks:
            gt = global_labels.get((img_name, folder), "N/A")
            key = (img_name, folder)
            vlm_result, latency = vlm_results.get(
                key,
                (VLMResult(parse_error="缺失"), 0.0),
            )

            if not vlm_result.is_valid:
                row = [
                    img_name,
                    os.path.basename(folder),
                    "error",
                    gt,
                    "err",
                    "err",
                    "err",
                    "err",
                    vlm_result.parse_error,
                    0,
                    0.0,
                ]
            else:
                pred, w_score = _judge(vlm_result, scoring_engine)
                row = [
                    img_name,
                    os.path.basename(folder),
                    pred,
                    gt,
                    vlm_result.composition,
                    vlm_result.angle,
                    vlm_result.distance,
                    vlm_result.context,
                    vlm_result.reason,
                    latency,
                    w_score,
                ]
            writer.write_row(row)
            final_rows.append(row)

    predictions = [normalize_label(r[2]) for r in final_rows]
    ground_truths = [normalize_label(r[3]) for r in final_rows]
    latencies = [r[9] for r in final_rows if isinstance(r[9], (int, float)) and r[9] > 0]
    metrics_result = calculate_metrics(predictions, ground_truths, latencies)

    print(f"\n  [{exp_name}] scoring={scoring_label}")
    print_metrics_report(metrics_result)

    m = metrics_result.to_dict()
    m.update(
        {
            "exp_name": exp_name,
            "mode": exp_config["mode"],
            "prompt_id": exp_config["prompt_id"],
            "scoring": scoring_label,
            "timestamp": timestamp,
        }
    )
    return m


# ================= VLM 缓存持久化 =================


def save_vlm_cache(all_vlm_results, output_path):
    """将 VLM 原始四维状态序列化到 JSON。"""
    serializable = {}
    for group_key, results in all_vlm_results.items():
        gk = f"{group_key[0]}|{group_key[1]}"
        serializable[gk] = {}
        for (img, folder), (vlm_result, latency) in results.items():
            ik = f"{img}|{folder}"
            serializable[gk][ik] = {
                "composition": vlm_result.composition,
                "angle": vlm_result.angle,
                "distance": vlm_result.distance,
                "context": vlm_result.context,
                "reason": vlm_result.reason,
                "parse_error": vlm_result.parse_error,
                "latency": latency,
            }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f">>> VLM 缓存已保存: {output_path}")


def load_vlm_cache(cache_path):
    """从 JSON 加载 VLM 缓存。"""
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for gk, items in data.items():
        parts = gk.split("|", 1)
        group_key = (parts[0], parts[1])
        result[group_key] = {}
        for ik, vals in items.items():
            img_parts = ik.split("|", 1)
            vlm_result = VLMResult(
                composition=vals["composition"],
                angle=vals["angle"],
                distance=vals["distance"],
                context=vals["context"],
                reason=vals["reason"],
                parse_error=vals.get("parse_error", ""),
            )
            result[group_key][(img_parts[0], img_parts[1])] = (
                vlm_result,
                vals["latency"],
            )
    return result


# ================= 汇总 =================

SUMMARY_FIELDS = [
    "exp_name",
    "mode",
    "prompt_id",
    "scoring",
    "acc",
    "pre",
    "rec",
    "f1",
    "tp",
    "tn",
    "fp",
    "fn",
    "total",
    "invalid",
    "avg_lat",
    "timestamp",
]


def write_summary(all_metrics, output_path):
    """写入汇总 CSV"""
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SUMMARY_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)
    print(f"\n>>> summary: {output_path}")


def print_comparison_table(all_metrics):
    """打印对比表"""
    print(f"\n{'=' * 110}")
    print("Contrast Experiment Summary (Optimized Runner v2)")
    print(f"{'=' * 110}")
    header = (
        f"{'Experiment':<30} {'Mode':<10} {'Prompt':<22} {'Score':<10} "
        f"{'Acc':>7} {'Pre':>7} {'Rec':>7} {'F1':>7} "
        f"{'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}"
    )
    print(header)
    print("-" * 110)
    for m in all_metrics:
        line = (
            f"{m['exp_name']:<30} {m['mode']:<10} "
            f"{m['prompt_id']:<22} {m['scoring']:<10} "
            f"{m.get('acc', 0):>7.2%} {m.get('pre', 0):>7.2%} "
            f"{m.get('rec', 0):>7.2%} {m.get('f1', 0):>7.2%} "
            f"{m.get('tp', 0):>4} {m.get('tn', 0):>4} "
            f"{m.get('fp', 0):>4} {m.get('fn', 0):>4}"
        )
        print(line)
    print(f"{'=' * 110}")


# ================= 主程序 =================


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="优化版批量对比实验运行器")
    parser.add_argument(
        "-e",
        "--experiments",
        type=str,
        default=None,
        help="实验索引(逗号分隔)，如 '0,1,2'",
    )
    parser.add_argument("-l", "--list", action="store_true", help="列出实验配置")
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="VLM 缓存 JSON 路径，跳过 VLM 调用直接重放评分",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default=None,
        help="重放模式下覆盖评分配置路径",
    )
    parser.add_argument(
        "--save-cache",
        action="store_true",
        default=True,
        help="保存 VLM 缓存 (默认开启)",
    )
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    if args.list:
        print("Available experiments:")
        for i, exp in enumerate(EXPERIMENTS):
            scoring_label = "weighted" if exp["scoring"] else "veto"
            strip = "strip" if exp.get("strip_geometry") else ""
            print(
                f"  [{i:>2}] {exp['name']:<30} mode={exp['mode']:<10} "
                f"prompt={exp['prompt_id']:<22} scoring={scoring_label:<8} "
                f"{strip}"
            )
        return

    if args.experiments:
        indices = [int(x.strip()) for x in args.experiments.split(",")]
        experiments_to_run = [EXPERIMENTS[i] for i in indices]
    else:
        experiments_to_run = EXPERIMENTS

    settings = get_settings()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    global_labels = load_all_labels(DATA_FOLDERS)
    image_tasks = collect_image_tasks(DATA_FOLDERS)

    print("=" * 60)
    print(">>> Optimized Batch Runner v2")
    print(f">>> {len(experiments_to_run)} experiments, {len(image_tasks)} images")
    print(f">>> Model: {MODEL}")

    # ─── 重放模式 ───
    if args.replay:
        print(f">>> [重放模式] 加载缓存: {args.replay}")
        all_vlm_results = load_vlm_cache(args.replay)
        all_metrics = []
        for exp in experiments_to_run:
            if args.scoring:
                exp = dict(exp, scoring=args.scoring)
            group_key = (exp["mode"], exp["prompt_id"])
            if group_key not in all_vlm_results:
                print(f"  [SKIP] {exp['name']}: 缓存中无 {group_key}")
                continue
            m = evaluate_and_write(
                exp,
                all_vlm_results[group_key],
                global_labels,
                image_tasks,
            )
            all_metrics.append(m)
        if all_metrics:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            write_summary(
                all_metrics,
                os.path.join(OUTPUT_ROOT, f"summary_replay_{ts}.csv"),
            )
            print_comparison_table(all_metrics)
        return

    # ─── 正常模式 ───
    clients = create_client_pool()
    print(f">>> {len(clients)} API clients")

    # 按 (mode, prompt_id) 聚合实验
    prompt_groups = defaultdict(list)
    for exp in experiments_to_run:
        gk = (exp["mode"], exp["prompt_id"])
        prompt_groups[gk].append(exp)

    unique_vlm = len(prompt_groups)
    total_exp = len(experiments_to_run)
    saved = total_exp - unique_vlm
    print(f">>> VLM 去重: {total_exp} 实验 -> {unique_vlm} 组 VLM 调用 (节省 {saved} 组)")

    # 预计算
    need_cv = any(e["mode"] == "vlm_cv" for e in experiments_to_run)
    need_pure = any(e["mode"] == "pure_vlm" for e in experiments_to_run)

    cv_cache = {}
    vlm_img_cache = {}

    if need_cv:
        print(">>> 加载 YOLOv8-Seg ...")
        segmentor = load_yolov8_seg(
            weights_path=SEGMENTOR_WEIGHTS,
            device="cuda:0",
        )
        segmentor.conf_threshold = 0.6
        vis_dir = os.path.join(OUTPUT_ROOT, "shared_visuals")
        cv_cache = precompute_yolo_cache(image_tasks, segmentor, vis_dir)

    if need_pure:
        vlm_img_cache = precompute_vlm_images(image_tasks)

    # VLM 调用 (按组)
    all_vlm_results = {}
    for i, (group_key, exps) in enumerate(prompt_groups.items()):
        mode, prompt_id = group_key
        prompt_text = load_prompt(prompt_id)
        strip_geo = any(e.get("strip_geometry", False) for e in exps)

        print(
            f"\n>>> [{i + 1}/{len(prompt_groups)}] VLM: mode={mode} prompt={prompt_id}",
        )
        vlm_results = run_vlm_group(
            group_key,
            image_tasks,
            clients,
            prompt_text,
            cv_cache,
            vlm_img_cache,
            settings.MAX_WORKERS,
            strip_geo,
        )
        all_vlm_results[group_key] = vlm_results

    # 保存 VLM 缓存
    if args.save_cache:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_path = os.path.join(OUTPUT_ROOT, f"vlm_cache_{ts}.json")
        save_vlm_cache(all_vlm_results, cache_path)

    # 评分重放
    print(f"\n>>> [评分重放] {len(experiments_to_run)} 个实验")
    all_metrics = []
    for exp in experiments_to_run:
        group_key = (exp["mode"], exp["prompt_id"])
        vlm_results = all_vlm_results[group_key]
        m = evaluate_and_write(
            exp,
            vlm_results,
            global_labels,
            image_tasks,
        )
        all_metrics.append(m)

    if all_metrics:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        write_summary(
            all_metrics,
            os.path.join(OUTPUT_ROOT, f"summary_{ts}.csv"),
        )
        print_comparison_table(all_metrics)


if __name__ == "__main__":
    main()
