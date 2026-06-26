"""Benchmark v1 基线运行脚本

从冻结清单 data/benchmark_v1/manifest_v1.csv 读取 88 张图片，
使用当前系统配置（cv_enhanced_p5 + scoring_optimized_cv_p4）跑基线。

主车选取策略：距画面中心最近（与 app.py 保持一致，非 contrast_VLM_CV_test_v2.py 的最高置信度）。

用法：
    cd /root/otto/XiaoanNew
    uv run python scripts/run_benchmark_v1.py --config assets/configs/benchmark_v1_run.yaml
    uv run python scripts/run_benchmark_v1.py --dry-run   # 只检查清单，不调用 VLM

输出：
    outputs/benchmark_runs/exp_<timestamp>_baseline_cv_enhanced_p5_scoring_cv_p4/
        results.csv          # 逐图预测结果
        summary.json         # 总体指标 + 逐维度准确率
        confusion_matrix.txt # 混淆矩阵
"""

import argparse
import concurrent.futures
import csv
import json
import os
import sys
import time
from datetime import datetime

import yaml
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
from modules.experiment.scoring import ScoringEngine
from modules.prompt.manager import load_prompt
from modules.vlm.parser import normalize_label, parse_vlm_response
from modules.vlm.retry import chat_completion_with_retry

# ================================================================
# 默认配置
# ================================================================
_DEFAULT_CONFIG = {
    "exp_name": "baseline_cv_enhanced_p5_scoring_cv_p4",
    "model": "qwen/qwen3-vl-30b-a3b-instruct",
    "prompt_id": "cv_enhanced_p5",
    "max_size": (768, 768),
    "quality": 80,
    "segmentor_weights": os.path.join(_PROJECT_ROOT, "assets/weights/best.pt"),
    "segmentor_device": "cuda:0",
    "conf_threshold": 0.6,
    "scoring_config": os.path.join(_PROJECT_ROOT, "assets/configs/scoring_optimized_cv_p4.yaml"),
    "benchmark_manifest": os.path.join(_PROJECT_ROOT, "data/benchmark_v1/manifest_v1.csv"),
    "output_root": os.path.join(_PROJECT_ROOT, "outputs/benchmark_runs"),
    "max_workers": 8,
}

CSV_HEADERS = [
    "image",
    "gt",
    "pred",
    "correct",
    # VLM 输出的四维状态标签
    "composition_status",
    "angle_status",
    "distance_status",
    "context_status",
    # ScoringEngine 映射后的四维数值分（用于错误归因：看是哪一维把综合分带偏的）
    "comp_score",
    "angle_score",
    "dist_score",
    "ctx_score",
    # 综合加权分
    "final_score",
    # CV 检测信息
    "num_detections",
    "electric_bike",
    "curb",
    "parking_lane",
    "tactile_paving",
    "main_vehicle_detected",
    # VLM 原因摘要
    "vlm_reason",
    "latency",
]


# ================================================================
# 清单加载
# ================================================================

def load_manifest(manifest_path: str) -> list[dict]:
    """加载 benchmark 清单，返回 [{filename, gt, rel_path}, ...]"""
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ================================================================
# 单图处理
# ================================================================

def process_image(args: tuple) -> list:
    """处理单张图片：YOLOv8-Seg + VLM，返回 CSV 行"""
    item, config, segmentor, client, vis_dir = args
    fname = item["filename"]
    gt = item["gt"]
    image_path = os.path.join(_PROJECT_ROOT, item["rel_path"])
    start_t = time.time()

    try:
        # 1. YOLOv8-Seg 实例分割
        seg_result = segmentor.predict(image_path, visual=False, retina_masks=True)
        raw_img = seg_result["image_raw"]
        objects = seg_result["objects"]
        H, W = seg_result["image_size"]

        # 2. 可视化线框图（保存到 vis_dir）
        vis_img = draw_wireframe_visual(raw_img, objects)
        if vis_dir:
            Image.fromarray(vis_img).save(os.path.join(vis_dir, fname))

        # 3. 主车选取：距画面中心最近（与 app.py 保持一致）
        img_cx, img_cy = W / 2.0, H / 2.0
        class_counts = {"Electric bike": 0, "Curb": 0, "parking lane": 0, "Tactile paving": 0}
        main_bike_mask, main_bike_dist = None, float("inf")
        cv_detections = []

        for obj in objects:
            lbl = obj["label"]
            cv_detections.append({"id": obj["id"], "label": lbl, "confidence": obj["confidence"], "bbox": obj["bbox"]})
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
        b64_raw = encode_image_to_base64(raw_img, config["max_size"], config["quality"])
        b64_vis = encode_image_to_base64(vis_img, config["max_size"], config["quality"])

        # 6. 组装 Prompt
        detection_info = {
            "image_size": [H, W],
            "detected_objects": cv_detections,
            "class_summary": class_counts,
            "geometry_analysis": geo,
        }
        import json as _json
        full_prompt = (
            load_prompt(config["prompt_id"])
            + "\n\n# YOLOv8-Seg Detection & Geometry Analysis\n```json\n"
            + _json.dumps(detection_info, ensure_ascii=False, indent=2)
            + "\n```"
        )

        # 7. VLM 调用
        resp = chat_completion_with_retry(
            client,
            model=config["model"],
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
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
        )
        vlm_text = resp.choices[0].message.content
        vlm_result = parse_vlm_response(vlm_text)

        if not vlm_result.is_valid:
            pred, score = "error", 0.0
            comp, ang, dist, ctx = "parse_fail", "parse_fail", "parse_fail", "parse_fail"
            comp_s = ang_s = dist_s = ctx_s = 0.0
            reason = vlm_result.parse_error
        else:
            sr = config["_scoring_engine"].score(*vlm_result.statuses)
            pred = "yes" if sr.is_compliant else "no"
            score = sr.final_score
            comp = vlm_result.composition
            ang  = vlm_result.angle
            dist = vlm_result.distance
            ctx  = vlm_result.context
            # 四维单独数值分（错误归因用）
            comp_s = sr.dimension_scores.get("composition", 0.0)
            ang_s  = sr.dimension_scores.get("angle",       0.0)
            dist_s = sr.dimension_scores.get("distance",    0.0)
            ctx_s  = sr.dimension_scores.get("context",     0.0)
            reason = str(vlm_result.reason)[:300]

        correct = "1" if normalize_label(pred) == normalize_label(gt) else "0"
        latency = round(time.time() - start_t, 3)

        return [
            fname, gt, pred, correct,
            comp, ang, dist, ctx,
            round(comp_s, 4), round(ang_s, 4), round(dist_s, 4), round(ctx_s, 4),
            round(score, 4),
            len(objects),
            class_counts.get("Electric bike", 0),
            class_counts.get("Curb", 0),
            class_counts.get("parking lane", 0),
            class_counts.get("Tactile paving", 0),
            int(geo["main_vehicle_detected"]),
            reason, latency,
        ]

    except Exception as e:
        import traceback
        traceback.print_exc()
        latency = round(time.time() - start_t, 3)
        return [fname, gt, "error", "0",
                "err", "err", "err", "err",
                0.0, 0.0, 0.0, 0.0, 0.0,
                0, 0, 0, 0, 0, 0, str(e)[:200], latency]


# ================================================================
# 指标汇总
# ================================================================

def compute_summary(rows: list[list]) -> dict:
    """计算总体准确率、混淆矩阵、逐维度分布"""
    tp = tn = fp = fn = 0

    for row in rows:
        gt   = normalize_label(row[1])
        pred = normalize_label(row[2])
        if gt == "yes":
            if pred == "yes":
                tp += 1
            else:
                fn += 1
        elif gt == "no":
            if pred == "no":
                tn += 1
            else:
                fp += 1

    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total else 0
    pre  = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * pre * rec / (pre + rec) if (pre + rec) else 0
    # 违规召回：正确识别出的 no / 全部 no
    viol_rec = tn / (tn + fp) if (tn + fp) else 0

    return {
        "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(pre, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "violation_recall": round(viol_rec, 4),
    }


def print_confusion_matrix(summary: dict) -> str:
    """格式化混淆矩阵文本"""
    lines = [
        "=== Confusion Matrix ===",
        "               Pred YES   Pred NO",
        f"  GT YES (合规)    {summary['tp']:4d}      {summary['fn']:4d}",
        f"  GT NO  (违规)    {summary['fp']:4d}      {summary['tn']:4d}",
        "",
        f"  Accuracy:         {summary['accuracy']:.4f}",
        f"  Precision:        {summary['precision']:.4f}",
        f"  Recall (合规):    {summary['recall']:.4f}",
        f"  F1:               {summary['f1']:.4f}",
        f"  Violation Recall: {summary['violation_recall']:.4f}",
    ]
    return "\n".join(lines)


# ================================================================
# 主流程
# ================================================================

def load_run_config(config_path: str) -> dict:
    """加载运行配置 YAML，合并默认值"""
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = dict(_DEFAULT_CONFIG)
    cfg.update(data)
    if isinstance(cfg.get("max_size"), list):
        cfg["max_size"] = tuple(cfg["max_size"])
    # 路径相对化
    for key in ("scoring_config", "benchmark_manifest", "segmentor_weights"):
        if cfg.get(key) and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(_PROJECT_ROOT, cfg[key])
    return cfg


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="Benchmark v1 基线运行脚本")
    parser.add_argument("--config", "-c", default=None, help="运行配置 YAML 路径")
    parser.add_argument("--dry-run", action="store_true", help="只验证清单，不调用 VLM")
    args = parser.parse_args()

    cfg = load_run_config(args.config) if args.config else dict(_DEFAULT_CONFIG)

    # 加载清单
    manifest_path = cfg["benchmark_manifest"]
    items = load_manifest(manifest_path)
    yes_cnt = sum(1 for r in items if r["gt"] == "yes")
    no_cnt  = sum(1 for r in items if r["gt"] == "no")
    print(f"[Benchmark v1] 清单加载完成: {len(items)} 张 (yes={yes_cnt}, no={no_cnt})")

    # 验证图片存在
    missing = [r for r in items if not os.path.exists(os.path.join(_PROJECT_ROOT, r["rel_path"]))]
    if missing:
        print(f"[警告] {len(missing)} 张图片在本地不存在（服务器路径），dry-run 继续，实际运行需在服务器执行")
        for m in missing[:3]:
            print(f"  缺: {m['rel_path']}")

    if args.dry_run:
        print("\n[dry-run] 清单验证完成，未调用 VLM。")
        print(f"预计 VLM 调用次数: {len(items)}")
        print(f"配置: prompt={cfg['prompt_id']}, scoring={os.path.basename(cfg['scoring_config'])}, threshold=0.35")
        print(f"输出目录: {cfg['output_root']}")
        return

    # 加载模型
    print(f"[Benchmark] 加载 YOLOv8-Seg: {cfg['segmentor_weights']}")
    segmentor = load_yolov8_seg(cfg["segmentor_weights"], device=cfg["segmentor_device"])
    segmentor.conf_threshold = cfg["conf_threshold"]

    scoring_engine = ScoringEngine.from_yaml(cfg["scoring_config"])
    cfg["_scoring_engine"] = scoring_engine

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(cfg["output_root"], f"exp_{timestamp}_{cfg['exp_name']}")
    vis_dir = os.path.join(exp_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)
    out_csv = os.path.join(exp_dir, "results.csv")
    print(f"[Benchmark] 输出目录: {exp_dir}")

    # 创建 VLM 客户端（单客户端，不分发多 key）
    settings = get_settings()
    from openai import OpenAI
    client = OpenAI(base_url=settings.API_BASE_URL, api_key=settings.VLM_API_KEY)

    # 并行处理
    task_args = [(item, cfg, segmentor, client, vis_dir) for item in items]
    all_rows = []
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["max_workers"]) as ex:
            for row in tqdm(ex.map(process_image, task_args), total=len(task_args), desc="Benchmark"):
                writer.writerow(row)
                all_rows.append(row)

    # 汇总指标
    summary = compute_summary(all_rows)
    cm_text = print_confusion_matrix(summary)
    print("\n" + cm_text)

    # 写入汇总文件
    with open(os.path.join(exp_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
            "manifest": manifest_path,
            "metrics": summary,
            "timestamp": timestamp,
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(exp_dir, "confusion_matrix.txt"), "w", encoding="utf-8") as f:
        f.write(cm_text)

    print(f"\n[Benchmark] 完成。结果: {out_csv}")
    print(f"  summary: {exp_dir}/summary.json")


if __name__ == "__main__":
    main()
