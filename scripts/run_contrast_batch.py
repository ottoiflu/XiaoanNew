"""批量对比实验运行器

支持纯 VLM 和 VLM+CV 两种模式，一票否决和加权评分两种评判方式。
在同一进程中顺序执行多组实验，输出统一对比表。

用法:
    uv run scripts/run_contrast_batch.py
    uv run scripts/run_contrast_batch.py --experiments 0,1,2
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
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
from modules.vlm.parser import normalize_label, parse_vlm_response
from modules.vlm.retry import chat_completion_with_retry

# ================= 全局常量 =================

MODEL = "qwen/qwen3-vl-30b-a3b-instruct"
MAX_SIZE = (768, 768)
QUALITY = 80
SCORING_YAML = "assets/configs/scoring_default.yaml"

DATA_FOLDERS = [
    os.path.join(_PROJECT_ROOT, "data/Compliance_test_data/no_val"),
    os.path.join(_PROJECT_ROOT, "data/Compliance_test_data/yes_val"),
]

OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs/contrast_experiments")
SEGMENTOR_WEIGHTS = os.path.join(_PROJECT_ROOT, "assets/weights/best.pt")

# ================= 实验矩阵定义 =================

EXPERIMENTS = [
    # --- Group A: 纯VLM vs VLM+CV, 一票否决 ---
    {"name": "vlm_p6_veto", "mode": "pure_vlm", "prompt_id": "standard_p6", "scoring": None},
    {"name": "cv_p6_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p6", "scoring": None},
    {"name": "vlm_p5_veto", "mode": "pure_vlm", "prompt_id": "standard_p5", "scoring": None},
    {"name": "cv_p5_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p5", "scoring": None},
    {"name": "cv_p7_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p7", "scoring": None},
    # --- Group B: 纯VLM vs VLM+CV, 加权评分 ---
    {"name": "vlm_p6_weighted", "mode": "pure_vlm", "prompt_id": "standard_p6", "scoring": SCORING_YAML},
    {"name": "cv_p6_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p6", "scoring": SCORING_YAML},
    {"name": "vlm_p5_weighted", "mode": "pure_vlm", "prompt_id": "standard_p5", "scoring": SCORING_YAML},
    {"name": "cv_p5_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p5", "scoring": SCORING_YAML},
    {"name": "cv_p7_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p7", "scoring": SCORING_YAML},
    # --- Group C: 额外对照 ---
    {"name": "cv_p4_veto", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4", "scoring": None},
    {"name": "cv_p4_weighted", "mode": "vlm_cv", "prompt_id": "cv_enhanced_p4", "scoring": SCORING_YAML},
    # --- Group E: p4 纯VLM对照 ---
    {"name": "vlm_p4_veto", "mode": "pure_vlm", "prompt_id": "standard_p4", "scoring": None},
    {"name": "vlm_p4_weighted", "mode": "pure_vlm", "prompt_id": "standard_p4", "scoring": SCORING_YAML},
    # --- Group D: VLM+CV 精简几何数据 (strip_geometry) ---
    {
        "name": "cv_p7_veto_minimal",
        "mode": "vlm_cv",
        "prompt_id": "cv_enhanced_p7",
        "scoring": None,
        "strip_geometry": True,
    },
    {
        "name": "cv_p7_weighted_minimal",
        "mode": "vlm_cv",
        "prompt_id": "cv_enhanced_p7",
        "scoring": SCORING_YAML,
        "strip_geometry": True,
    },
    {
        "name": "cv_p6_veto_minimal",
        "mode": "vlm_cv",
        "prompt_id": "cv_enhanced_p6",
        "scoring": None,
        "strip_geometry": True,
    },
    {
        "name": "cv_p6_weighted_minimal",
        "mode": "vlm_cv",
        "prompt_id": "cv_enhanced_p6",
        "scoring": SCORING_YAML,
        "strip_geometry": True,
    },
]

# ================= CSV 表头 =================

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


# ================= 核心处理函数 =================


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


def process_pure_vlm(args):
    """纯 VLM 模式处理单张图片"""
    image_name, folder_path, client, labels_dict, prompt_text, scoring_engine = args
    image_path = os.path.join(folder_path, image_name)
    gt = labels_dict.get((image_name, folder_path), "N/A")
    start_t = time.time()

    try:
        b64_img = encode_image_to_base64(image_path, max_size=MAX_SIZE, quality=QUALITY)

        res = chat_completion_with_retry(
            client,
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ],
                }
            ],
            max_tokens=600,
            temperature=0.1,
        )
        vlm_out = res.choices[0].message.content
        vlm_result = parse_vlm_response(vlm_out)

        if not vlm_result.is_valid:
            return _error_row(image_name, folder_path, gt, vlm_result.parse_error)

        pred, w_score = _judge(vlm_result, scoring_engine)
        return [
            image_name,
            os.path.basename(folder_path),
            pred,
            gt,
            vlm_result.composition,
            vlm_result.angle,
            vlm_result.distance,
            vlm_result.context,
            vlm_result.reason,
            round(time.time() - start_t, 3),
            w_score,
        ]
    except Exception as e:
        return _error_row(image_name, folder_path, gt, str(e))


def process_vlm_cv(args):
    """VLM+CV 模式处理单张图片"""
    image_name, folder_path, client, labels_dict, prompt_text, scoring_engine, segmentor, vis_dir = args[:8]
    strip_geometry = args[8] if len(args) > 8 else False
    image_path = os.path.join(folder_path, image_name)
    gt = labels_dict.get((image_name, folder_path), "N/A")
    start_t = time.time()

    try:
        seg_result = segmentor.predict(image_path)
        raw_img = seg_result["image_raw"]
        objects = seg_result["objects"]
        H, W = seg_result["image_size"]

        vis_img = draw_wireframe_visual(raw_img, objects)
        vis_path = os.path.join(vis_dir, image_name)
        Image.fromarray(vis_img).save(vis_path)

        b64_raw = encode_image_to_base64(raw_img, MAX_SIZE, QUALITY)
        b64_vis = encode_image_to_base64(vis_img, MAX_SIZE, QUALITY)

        class_counts = {"Electric bike": 0, "Curb": 0, "parking lane": 0, "Tactile paving": 0}
        detection_info = {"image_size": [H, W], "detected_objects": [], "geometry_analysis": {}}
        main_bike_mask, main_bike_conf = None, -1

        for obj in objects:
            detection_info["detected_objects"].append(
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

        detection_info["class_summary"] = class_counts

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

        detection_info["geometry_analysis"] = geo

        if strip_geometry:
            detection_info["geometry_analysis"] = {
                "main_vehicle_detected": geo["main_vehicle_detected"],
            }
            detection_info["detected_objects"] = [
                {"label": o["label"], "confidence": round(o["confidence"], 2)}
                for o in detection_info["detected_objects"]
            ]
        structured_info = json.dumps(detection_info, ensure_ascii=False, indent=2)

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
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_raw}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_vis}"}},
                    ],
                }
            ],
            max_tokens=1000,
            temperature=0.1,
            top_p=0.9,
        )
        vlm_out = res.choices[0].message.content
        vlm_result = parse_vlm_response(vlm_out)

        if not vlm_result.is_valid:
            return _error_row(image_name, folder_path, gt, vlm_result.parse_error)

        pred, w_score = _judge(vlm_result, scoring_engine)
        return [
            image_name,
            os.path.basename(folder_path),
            pred,
            gt,
            vlm_result.composition,
            vlm_result.angle,
            vlm_result.distance,
            vlm_result.context,
            vlm_result.reason,
            round(time.time() - start_t, 3),
            w_score,
        ]
    except Exception as e:
        import traceback

        traceback.print_exc()
        return _error_row(image_name, folder_path, gt, str(e))


def _error_row(image_name, folder_path, gt, error_msg):
    """生成错误行"""
    return [
        image_name,
        os.path.basename(folder_path),
        "error",
        gt,
        "err",
        "err",
        "err",
        "err",
        error_msg,
        0,
        0.0,
    ]


# ================= 单次实验运行 =================


def run_single_experiment(exp_config, clients, segmentor, global_labels, image_tasks, settings):
    """执行单次实验并返回指标字典"""
    exp_name = exp_config["name"]
    mode = exp_config["mode"]
    prompt_id = exp_config["prompt_id"]
    scoring_path = exp_config["scoring"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_ROOT, f"{timestamp}_{exp_name}")
    vis_dir = os.path.join(exp_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    scoring_engine = _build_scoring_engine(scoring_path)
    scoring_label = "weighted" if scoring_engine else "veto"
    prompt_text = load_prompt(prompt_id)

    print(f"\n{'=' * 60}")
    print(f">>> [{exp_name}] mode={mode} prompt={prompt_id} scoring={scoring_label}")
    print(f">>> dir: {exp_dir}")

    if mode == "pure_vlm":
        all_tasks = []
        for i, (img_name, folder) in enumerate(image_tasks):
            client = clients[i % len(clients)]
            all_tasks.append((img_name, folder, client, global_labels, prompt_text, scoring_engine))
        worker_fn = process_pure_vlm
    else:
        strip_geo = exp_config.get("strip_geometry", False)
        all_tasks = []
        for i, (img_name, folder) in enumerate(image_tasks):
            client = clients[i % len(clients)]
            all_tasks.append(
                (img_name, folder, client, global_labels, prompt_text, scoring_engine, segmentor, vis_dir, strip_geo)
            )
        worker_fn = process_vlm_cv

    print(f">>> {len(all_tasks)} images")

    out_csv = os.path.join(exp_dir, f"{exp_name}.csv")
    final_results = []

    with ResultWriter(out_csv, CSV_HEADERS) as writer:
        with concurrent.futures.ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
            for row in tqdm(
                executor.map(worker_fn, all_tasks),
                total=len(all_tasks),
                desc=f"[{exp_name}]",
            ):
                writer.write_row(row)
                final_results.append(row)

    predictions = [normalize_label(r[2]) for r in final_results]
    ground_truths = [normalize_label(r[3]) for r in final_results]
    latencies = [r[9] for r in final_results if isinstance(r[9], (int, float)) and r[9] > 0]
    metrics_result = calculate_metrics(predictions, ground_truths, latencies)
    print_metrics_report(metrics_result)

    m = metrics_result.to_dict()
    m.update(
        {
            "exp_name": exp_name,
            "mode": mode,
            "prompt_id": prompt_id,
            "scoring": scoring_label,
            "timestamp": timestamp,
        }
    )
    return m


# ================= 汇总输出 =================

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
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)
    print(f"\n>>> summary saved: {output_path}")


def print_comparison_table(all_metrics):
    """打印对比表"""
    print(f"\n{'=' * 100}")
    print("Contrast Experiment Summary")
    print(f"{'=' * 100}")
    header = f"{'Experiment':<30} {'Mode':<10} {'Scoring':<10} {'Acc':>7} {'Pre':>7} {'Rec':>7} {'F1':>7} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}"
    print(header)
    print("-" * 100)
    for m in all_metrics:
        line = (
            f"{m['exp_name']:<30} {m['mode']:<10} {m['scoring']:<10} "
            f"{m.get('acc', 0):>7.2%} {m.get('pre', 0):>7.2%} {m.get('rec', 0):>7.2%} {m.get('f1', 0):>7.2%} "
            f"{m.get('tp', 0):>4} {m.get('tn', 0):>4} {m.get('fp', 0):>4} {m.get('fn', 0):>4}"
        )
        print(line)
    print(f"{'=' * 100}")


# ================= 主程序 =================


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量对比实验运行器")
    parser.add_argument(
        "--experiments",
        "-e",
        type=str,
        default=None,
        help="实验索引(逗号分隔)，如 '0,1,2'。默认全部。",
    )
    parser.add_argument("--list", "-l", action="store_true", help="列出所有实验配置")
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    if args.list:
        print("Available experiments:")
        for i, exp in enumerate(EXPERIMENTS):
            scoring_label = "weighted" if exp["scoring"] else "veto"
            print(
                f"  [{i:>2}] {exp['name']:<30} mode={exp['mode']:<10} prompt={exp['prompt_id']:<20} scoring={scoring_label}"
            )
        return

    if args.experiments:
        indices = [int(x.strip()) for x in args.experiments.split(",")]
        experiments_to_run = [EXPERIMENTS[i] for i in indices]
    else:
        experiments_to_run = EXPERIMENTS

    settings = get_settings()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("=" * 60)
    print(">>> Batch contrast experiment starting")
    print(f">>> {len(experiments_to_run)} experiments queued")
    print(f">>> Model: {MODEL}")

    clients = create_client_pool()
    print(f">>> {len(clients)} API clients ready")

    need_cv = any(e["mode"] == "vlm_cv" for e in experiments_to_run)
    segmentor = None
    if need_cv:
        print(">>> Loading YOLOv8-Seg model...")
        segmentor = load_yolov8_seg(weights_path=SEGMENTOR_WEIGHTS, device="cuda:0")
        segmentor.conf_threshold = 0.6
        print(">>> YOLOv8-Seg model loaded")

    global_labels = load_all_labels(DATA_FOLDERS)
    image_tasks = collect_image_tasks(DATA_FOLDERS)
    print(f">>> {len(image_tasks)} images, {len(global_labels)} labels")
    print("=" * 60)

    all_metrics = []
    for i, exp in enumerate(experiments_to_run):
        print(f"\n>>> [{i + 1}/{len(experiments_to_run)}] Starting: {exp['name']}")
        try:
            metrics = run_single_experiment(exp, clients, segmentor, global_labels, image_tasks, settings)
            all_metrics.append(metrics)
        except Exception as e:
            print(f">>> Experiment {exp['name']} failed: {e}")
            import traceback

            traceback.print_exc()

    if all_metrics:
        summary_path = os.path.join(OUTPUT_ROOT, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        write_summary(all_metrics, summary_path)
        print_comparison_table(all_metrics)


if __name__ == "__main__":
    main()
