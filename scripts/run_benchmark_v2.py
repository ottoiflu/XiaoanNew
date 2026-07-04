"""Benchmark v2 基线运行脚本

从 data/benchmark/benchmark_v2/fourdim_gt_v2.json 读取 600 张图片
（300 yes + 300 no），支持两种模式：

pure 模式：
    仅向 VLM 发送原图 + cv_enhanced_p5 prompt，不注入任何 YOLO/CV 数据。
cv 模式：
    走完整 check_parking 合规判定链路——YOLOv8-Seg 分割→主车取距画面中心最近→
    几何 IoU→线框图+结构化 JSON 注入 prompt→VLM→scoring p4 阈值 0.35。
两模式均绕过车牌验证（benchmark 图无车牌）。

用法：
    uv run python scripts/run_benchmark_v2.py --mode pure --workers 16 \\
        --out outputs/benchmark_output/v2/base_prompt
    uv run python scripts/run_benchmark_v2.py --mode cv --workers 8 \\
        --out outputs/benchmark_output/v2/cv_enhanced
    uv run python scripts/run_benchmark_v2.py --mode pure --smoke 5   # 冒烟测试
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
from datetime import datetime
from statistics import median

from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from modules.config.settings import get_settings
from modules.cv.image_utils import encode_image_to_base64
from modules.experiment.scoring import ScoringEngine
from modules.prompt.manager import load_prompt
from modules.vlm.parser import normalize_label, parse_vlm_response
from modules.vlm.retry import chat_completion_with_retry

# ================================================================
# 配置常量
# ================================================================
MODEL = "qwen/qwen3-vl-30b-a3b-instruct"
MAX_SIZE = (768, 768)
QUALITY = 80
PROMPT_ID = "cv_enhanced_v2_new4dim"
SCORING_PATH = os.path.join(_PROJECT_ROOT, "assets/configs/scoring_new4d_gs_best.yaml")
BENCHMARK_PATH = os.path.join(_PROJECT_ROOT, "data/benchmark/benchmark_v2/fourdim_gt_v2.json")

CSV_HEADERS = [
    "id", "gt", "pred", "final_score",
    "position_status", "medium_status", "angle_status", "state_status",
    "pos_score", "med_score", "ang_score", "state_score",
    "vlm_reason", "latency_sec",
    "pos_conf", "med_conf", "ang_conf", "state_conf",
    "iou_parking", "overlap_parking", "overlap_tactile",
    "contact_tactile", "contact_parking",
    "tactile_detected", "curb_detected", "parking_lane_detected",
    "main_bike_conf",
]

# CV 模式懒加载占位
_segmentor = None
_depth_assist = None

# ================================================================
# 数据集加载
# ================================================================


def load_benchmark(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    yes_cnt = sum(1 for d in data if d["gt"] == "yes")
    no_cnt = sum(1 for d in data if d["gt"] == "no")
    print(f"[Benchmark v2] 数据集加载完成: {len(data)} 张 (yes={yes_cnt}, no={no_cnt})")
    return data


def resolve_image_path(item: dict) -> str:
    """src 是相对 data/ 的路径，部分扩展名为 .JPEG"""
    return os.path.join(_PROJECT_ROOT, "data", item["src"])


# ================================================================
# Pure 模式：单图处理（纯 VLM，无 CV 注入）
# ================================================================


def process_pure(args: tuple) -> list:
    # CV defaults (pure mode doesn't run YOLO)
    geo = {"iou_with_parking_lane": 0.0, "overlap_with_parking_lane": 0.0, "overlap_with_tactile_paving": 0.0}
    real_contact = {"tactile": False, "parking_lane": False}
    class_counts = {"Tactile paving": 0, "Curb": 0, "parking lane": 0}
    main_bike_conf = 0.0
    item, client, scoring_engine, config = args
    image_path = resolve_image_path(item)
    image_id = item["id"]
    gt = item["gt"]
    start_t = time.time()

    try:
        b64_img = encode_image_to_base64(
            image_path, config["max_size"], config["quality"]
        )
        prompt_text = load_prompt(config["prompt_id"])

        resp = chat_completion_with_retry(
            client,
            model=config["model"],
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
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            logprobs=True,
            top_logprobs=5,
        )
        vlm_out = resp.choices[0].message.content
        vlm_result = parse_vlm_response(vlm_out)
        # 提取 logprobs
        logprobs_data = None
        if hasattr(resp.choices[0], "logprobs") and resp.choices[0].logprobs:
            logprobs_data = resp.choices[0].logprobs

        if not vlm_result.is_valid:
            lat = round(time.time() - start_t, 3)
            return [
                image_id, gt, "error", 0.0,
                "parse_fail", "parse_fail", "parse_fail", "parse_fail",
                0.0, 0.0, 0.0, 0.0,
                vlm_result.parse_error[:300], lat,
            ]

        sr = scoring_engine.score_interp(
            vlm_result.position, vlm_result.medium,
            vlm_result.angle, vlm_result.state,
            position_conf=vlm_result.position_confidence,
            medium_conf=vlm_result.medium_confidence,
            angle_conf=vlm_result.angle_confidence,
            state_conf=vlm_result.state_confidence,
        )
        pred = "yes" if sr.is_compliant else "no"
        lat = round(time.time() - start_t, 3)

        return [
            image_id, gt, pred, round(sr.final_score, 4),
            vlm_result.position, vlm_result.medium,
            vlm_result.angle, vlm_result.state,
            round(sr.dimension_scores.get("position", 0.0), 4),
            round(sr.dimension_scores.get("medium", 0.0), 4),
            round(sr.dimension_scores.get("angle", 0.0), 4),
            round(sr.dimension_scores.get("state", 0.0), 4),
            str(vlm_result.reason)[:300], lat,
            round(vlm_result.position_confidence, 4),
            round(vlm_result.medium_confidence, 4),
            round(vlm_result.angle_confidence, 4),
            round(vlm_result.state_confidence, 4),
            round(float(geo.get("iou_with_parking_lane", 0.0)), 4),
            round(float(geo.get("overlap_with_parking_lane", 0.0)), 4),
            round(float(geo.get("overlap_with_tactile_paving", 0.0)), 4),
            int(real_contact.get("tactile", False)),
            int(real_contact.get("parking_lane", False)),
            int(class_counts.get("Tactile paving", 0) > 0),
            int(class_counts.get("Curb", 0) > 0),
            int(class_counts.get("parking lane", 0) > 0),
            round(float(main_bike_conf), 4),
        ]
    except Exception as e:
        import traceback

        traceback.print_exc()
        lat = round(time.time() - start_t, 3)
        return [
            image_id, gt, "error", 0.0,
            "err", "err", "err", "err",
            0.0, 0.0, 0.0, 0.0,
            str(e)[:200], lat,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0, 0,
            0, 0, 0,
            0.0,
        ]


# ================================================================
# CV 模式：懒加载 YOLO
# ================================================================


def ensure_segmentor():
    global _segmentor
    if _segmentor is not None:
        return
    from modules.cv.yolov8_inference import load_yolov8_seg

    weights = os.path.join(_PROJECT_ROOT, "assets/weights/best.pt")
    print("[Benchmark v2] 加载 YOLOv8-Seg 模型...")
    _segmentor = load_yolov8_seg(weights, device="cuda:0")
    _segmentor.conf_threshold = 0.6
    print("[Benchmark v2] YOLOv8-Seg 加载完成")


def ensure_depth_assist():
    """懒加载深度估计模型"""
    global _depth_assist
    if _depth_assist is not None:
        return
    from modules.cv.depth_assist import DepthAssist
    _depth_assist = DepthAssist(device="cuda:0")


# ================================================================
# CV 模式：单图处理（YOLOv8-Seg + CV 注入 + VLM）
# ================================================================


def process_cv(args: tuple) -> list:
    from modules.cv.image_utils import (
        calculate_iou_and_overlap,
        combine_masks,
        draw_wireframe_visual,
    )
    from modules.cv.depth_assist import check_contact
    from PIL import Image as PILImage

    item, client, scoring_engine, config, vis_dir = args
    image_path = resolve_image_path(item)
    image_id = item["id"]
    gt = item["gt"]
    start_t = time.time()

    try:
        # 1. YOLOv8-Seg
        seg_result = _segmentor.predict(image_path, visual=False, retina_masks=True)
        raw_img = seg_result["image_raw"]
        objects = seg_result["objects"]
        H, W = seg_result["image_size"]

        # 1b. 深度估计 + 3D 接触判断
        depth_map = _depth_assist.estimate(raw_img)
        main_bike_mask_first = None
        for obj in objects:
            if obj["label"] == "Electric bike" and obj.get("mask") is not None:
                main_bike_mask_first = obj["mask"].astype(bool)
                break
        blind_mask = combine_masks(objects, "Tactile paving")
        parking_mask = combine_masks(objects, "parking lane")
        real_contact = check_contact(main_bike_mask_first, blind_mask, parking_mask, depth_map)

        # 2. 线框图（深度辅助：仅高亮真实接触）
        vis_img = draw_wireframe_visual(raw_img, objects, real_contact=real_contact)
        if vis_dir:
            PILImage.fromarray(vis_img).save(
                os.path.join(vis_dir, os.path.basename(image_path))
            )

        # 3. 主车选取：距画面中心最近
        img_cx, img_cy = W / 2.0, H / 2.0
        class_counts = {
            "Electric bike": 0,
            "Curb": 0,
            "parking lane": 0,
            "Tactile paving": 0,
        }
        main_bike_mask, main_bike_dist, main_bike_conf = None, float("inf"), 0.0
        cv_detections = []

        for obj in objects:
            lbl = obj["label"]
            cv_detections.append(
                {
                    "id": obj["id"],
                    "label": lbl,
                    "confidence": obj["confidence"],
                    "bbox": obj["bbox"],
                }
            )
            if lbl in class_counts:
                class_counts[lbl] += 1
            if lbl == "Electric bike":
                bx1, by1, bx2, by2 = obj["bbox"]
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                d = (bcx - img_cx) ** 2 + (bcy - img_cy) ** 2
                if d < main_bike_dist:
                    main_bike_dist = d
                    main_bike_mask = obj.get("mask")

        # 4. 几何指标
        geo = {
            "main_vehicle_detected": main_bike_mask is not None,
            "overlap_with_parking_lane": 0.0,
            "iou_with_parking_lane": 0.0,
            "overlap_with_tactile_paving": 0.0,
            "status_inference": "unknown",
        }
        if main_bike_mask is not None:
            p_mask = combine_masks(objects, "parking lane")
            if p_mask is not None:
                iou, overlap = calculate_iou_and_overlap(main_bike_mask, p_mask)
                geo["iou_with_parking_lane"] = iou
                geo["overlap_with_parking_lane"] = overlap
            t_mask = combine_masks(objects, "Tactile paving")
            if t_mask is not None:
                _, ov_t = calculate_iou_and_overlap(main_bike_mask, t_mask)
                geo["overlap_with_tactile_paving"] = ov_t
            if geo["overlap_with_parking_lane"] > 0.8:
                geo["status_inference"] = "Likely Compliant (High Overlap)"
            elif geo["overlap_with_parking_lane"] < 0.1:
                geo["status_inference"] = "Likely Out of Bounds"

        # 5. 编码图像
        b64_raw = encode_image_to_base64(
            raw_img, config["max_size"], config["quality"]
        )
        b64_vis = encode_image_to_base64(
            vis_img, config["max_size"], config["quality"]
        )

        # 6. 组装 Prompt + 原图 + 视觉标注图（2D 标注版，不注入数值 JSON）
        detection_info = {
            "image_size": [H, W],
            "detected_objects": cv_detections,
            "class_summary": class_counts,
            "geometry_analysis": geo,
        }

        # 7. VLM（只发 prompt + 原图 + 视觉标注图，不拼 JSON 数值块）
        resp = chat_completion_with_retry(
            client,
            model=config["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": load_prompt(config["prompt_id"])},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_raw}"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_vis}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            logprobs=True,
            top_logprobs=5,
        )
        vlm_out = resp.choices[0].message.content
        vlm_result = parse_vlm_response(vlm_out)
        # 提取 logprobs
        logprobs_data = None
        if hasattr(resp.choices[0], "logprobs") and resp.choices[0].logprobs:
            logprobs_data = resp.choices[0].logprobs

        if not vlm_result.is_valid:
            lat = round(time.time() - start_t, 3)
            return [
                image_id, gt, "error", 0.0,
                "parse_fail", "parse_fail", "parse_fail", "parse_fail",
                0.0, 0.0, 0.0, 0.0,
                vlm_result.parse_error[:300], lat,
            ]

        sr = scoring_engine.score_interp(
            vlm_result.position, vlm_result.medium,
            vlm_result.angle, vlm_result.state,
            position_conf=vlm_result.position_confidence,
            medium_conf=vlm_result.medium_confidence,
            angle_conf=vlm_result.angle_confidence,
            state_conf=vlm_result.state_confidence,
        )
        pred = "yes" if sr.is_compliant else "no"
        lat = round(time.time() - start_t, 3)

        return [
            image_id, gt, pred, round(sr.final_score, 4),
            vlm_result.position, vlm_result.medium,
            vlm_result.angle, vlm_result.state,
            round(sr.dimension_scores.get("position", 0.0), 4),
            round(sr.dimension_scores.get("medium", 0.0), 4),
            round(sr.dimension_scores.get("angle", 0.0), 4),
            round(sr.dimension_scores.get("state", 0.0), 4),
            str(vlm_result.reason)[:300], lat,
            round(vlm_result.position_confidence, 4),
            round(vlm_result.medium_confidence, 4),
            round(vlm_result.angle_confidence, 4),
            round(vlm_result.state_confidence, 4),
            round(float(geo.get("iou_with_parking_lane", 0.0)), 4),
            round(float(geo.get("overlap_with_parking_lane", 0.0)), 4),
            round(float(geo.get("overlap_with_tactile_paving", 0.0)), 4),
            int(real_contact.get("tactile", False)),
            int(real_contact.get("parking_lane", False)),
            int(class_counts.get("Tactile paving", 0) > 0),
            int(class_counts.get("Curb", 0) > 0),
            int(class_counts.get("parking lane", 0) > 0),
            round(float(main_bike_conf), 4),
        ]
    except Exception as e:
        import traceback

        traceback.print_exc()
        lat = round(time.time() - start_t, 3)
        return [
            image_id, gt, "error", 0.0,
            "err", "err", "err", "err",
            0.0, 0.0, 0.0, 0.0,
            str(e)[:200], lat,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0, 0,
            0, 0, 0,
            0.0,
        ]


# ================================================================
# 指标汇总
# ================================================================


def compute_summary(rows: list[list]) -> dict:
    tp = tn = fp = fn = 0
    latencies = []
    for row in rows:
        g = normalize_label(row[1])
        p = normalize_label(row[2])
        if g == "yes":
            tp += p == "yes"
            fn += p != "yes"
        elif g == "no":
            tn += p == "no"
            fp += p != "no"
        lat = row[-1]
        if isinstance(lat, (int, float)) and lat > 0:
            latencies.append(lat)

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    pre = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
    viol_rec = tn / (tn + fp) if (tn + fp) else 0

    lat_sorted = sorted(latencies)
    n = len(lat_sorted)
    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(pre, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "violation_recall": round(viol_rec, 4),
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "error_count": sum(1 for r in rows if len(r) > 2 and r[2] == "error"),
        "parse_success_rate": round(1 - sum(1 for r in rows if len(r) > 2 and r[2] == "error") / total, 4) if total else 0,
        "latency": {
            "total_sec": round(sum(latencies), 2),
            "mean_sec": round(sum(lat_sorted) / n, 3) if n else 0,
            "median_sec": round(median(lat_sorted), 3) if n else 0,
            "p95_sec": round(lat_sorted[int(n * 0.95)], 3) if n else 0,
            "p99_sec": round(lat_sorted[int(n * 0.99)], 3) if n else 0,
            "min_sec": round(lat_sorted[0], 3) if n else 0,
            "max_sec": round(lat_sorted[-1], 3) if n else 0,
            "count": n,
        },
    }


def format_report(s: dict) -> str:
    lines = [
        "=== Confusion Matrix ===",
        "               Pred YES   Pred NO",
        f"  GT YES (合规)    {s['tp']:4d}      {s['fn']:4d}",
        f"  GT NO  (违规)    {s['fp']:4d}      {s['tn']:4d}",
        "",
        f"  Accuracy:         {s['accuracy']:.4f}",
        f"  Precision:        {s['precision']:.4f}",
        f"  Recall (合规):    {s['recall']:.4f}",
        f"  F1:               {s['f1']:.4f}",
        f"  Violation Recall: {s['violation_recall']:.4f}",
        "",
        "=== Latency Stats (sec) ===",
        f"  Total:   {s['latency']['total_sec']:.2f}",
        f"  Mean:    {s['latency']['mean_sec']:.3f}",
        f"  Median:  {s['latency']['median_sec']:.3f}",
        f"  P95:     {s['latency']['p95_sec']:.3f}",
        f"  P99:     {s['latency']['p99_sec']:.3f}",
        f"  Min:     {s['latency']['min_sec']:.3f}",
        f"  Max:     {s['latency']['max_sec']:.3f}",
    ]
    return "\n".join(lines)


# ================================================================
# 主入口
# ================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark v2 基线运行脚本")
    parser.add_argument(
        "--mode",
        choices=["pure", "cv"],
        required=True,
        help="pure=纯VLM, cv=YOLO+CV+VLM全链路",
    )
    parser.add_argument("--workers", type=int, default=8, help="并发数")
    parser.add_argument(
        "--out", type=str, default=None, help="输出目录（相对项目根）"
    )
    parser.add_argument(
        "--smoke", type=int, default=0, help="冒烟：只处理前 N 张"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="prompt ID (覆盖默认)"
    )
    parser.add_argument(
        "--scoring", type=str, default=None, help="scoring 配置名 (覆盖默认)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="VLM model ID (覆盖默认)"
    )
    parser.add_argument(
        "--logprobs", action="store_true", help="启用 logprobs 模式 (B版)"
    )
    parser.add_argument(
        "--data", type=str, default=None, help="Benchmark JSON 路径 (覆盖默认)"
    )
    args = parser.parse_args()

    # 加载数据集
    data_path = args.data if args.data else BENCHMARK_PATH
    items = load_benchmark(data_path)
    if args.smoke > 0:
        items = items[: args.smoke]
        print(f"[Benchmark v2] 冒烟模式：仅处理 {len(items)} 张")

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out:
        out_dir = os.path.join(_PROJECT_ROOT, args.out)
    else:
        mode_tag = "pure" if args.mode == "pure" else "cv"
        out_dir = os.path.join(
            _PROJECT_ROOT, f"outputs/benchmark_output/v2/{mode_tag}_{timestamp}"
        )
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "visuals") if args.mode == "cv" else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"[Benchmark v2] 模式={args.mode}, workers={args.workers}")
    print(f"[Benchmark v2] 输出: {out_dir}")

    # 初始化公用组件
    prompt_id = args.prompt if args.prompt else PROMPT_ID
    scoring_path = os.path.join(_PROJECT_ROOT, f"assets/configs/{args.scoring}.yaml") if args.scoring else SCORING_PATH
    config = {
        "model": args.model if args.model else MODEL,
        "prompt_id": prompt_id,
        "max_size": MAX_SIZE,
        "quality": QUALITY,
    }
    scoring_engine = ScoringEngine.from_yaml(scoring_path)
    settings = get_settings()
    from openai import OpenAI

    client = OpenAI(base_url=settings.API_BASE_URL, api_key=settings.VLM_API_KEY)

    # CV 模式：加载 YOLO + 深度模型
    if args.mode == "cv":
        ensure_segmentor()
        ensure_depth_assist()

    # 构造任务参数
    if args.mode == "pure":
        task_args = [(item, client, scoring_engine, config) for item in items]
        worker_fn = process_pure
    else:
        task_args = [
            (item, client, scoring_engine, config, vis_dir) for item in items
        ]
        worker_fn = process_cv

    # 并行处理
    out_csv = os.path.join(out_dir, "results.csv")
    all_rows = []
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as ex:
            for row in tqdm(
                ex.map(worker_fn, task_args),
                total=len(task_args),
                desc=f"Benchmark [{args.mode}]",
            ):
                writer.writerow(row)
                all_rows.append(row)

    # 汇总指标
    summary = compute_summary(all_rows)
    report = format_report(summary)
    print("\n" + report)

    # 写汇总文件
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "mode": args.mode,
                    "model": args.model if args.model else MODEL,
                    "prompt_id": prompt_id,
                    "scoring": os.path.basename(scoring_path),
                    "threshold": scoring_engine.config.threshold,
                    "workers": args.workers,
                    "dataset": os.path.basename(data_path),
                    "image_count": len(items),
                    "timestamp": timestamp,
                },
                "metrics": {k: v for k, v in summary.items() if k != "latency"},
                "latency": summary["latency"],
                "parse_success_rate": summary.get("parse_success_rate", 1.0),
                "error_count": summary.get("error_count", 0),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        os.path.join(out_dir, "confusion_matrix.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report)

    print(f"\n[Benchmark v2] 完成 -> {out_csv}")
    print(f"              -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
