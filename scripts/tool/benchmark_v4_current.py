#!/usr/bin/env python
"""Benchmark v4 当前上线管线完整实验脚本

忠实复刻 app.py check_parking (lines 399-720) 的逻辑，批量跑 benchmark v4 (1152张)，
捕获所有 CV 中间结果（消歧/角度双解/路缘降级/各 overlap/override 等）供 few-shot prompt 分析。

不 import app.py 的 check_parking（避免 Flask 上下文依赖），独立实现。
不修改 app.py / 其他 modules 文件，仅 import 复用。

输出:
  CSV:     {out_dir}/v4_current_results.csv
  JSON:    {out_dir}/v4_current_results.json
  summary: {out_dir}/summary.md

用法:
  .venv/bin/python scripts/tool/benchmark_v4_current.py --smoke 5 --out-dir /tmp/smoke_v4
  .venv/bin/python scripts/tool/benchmark_v4_current.py            # 全量 1152
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from openai import OpenAI

# ── 项目路径 ──
PROJECT_ROOT = "/root/XiaoanNew"
sys.path.insert(0, PROJECT_ROOT)

from modules.config.settings import settings
from modules.cv.angle_inference import (
    analyze_angle,
    pca_2d_image,
    cluster_lines,
    angle_between,
    cv_disambiguate,
    cv_judge_angle,
)
from modules.cv.blind_lane_check import check_blind_lane, _get_depth_assist
from modules.cv.image_utils import (
    calculate_iou_and_overlap,
    combine_masks,
    compress_image,
    draw_wireframe_visual,
    encode_image_to_base64,
)
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.experiment.scoring import ScoringEngine
from modules.prompt.manager import load_prompt
from modules.vlm.parser import parse_vlm_response
from modules.vlm.retry import chat_completion_with_retry

# ── 配置 ──
GT_PATH = "/root/autodl-tmp/XiaoanNew_data/benchmark/benchmark_v4/fourdim_gt_v4.json"
IMG_ROOT = "/root/autodl-tmp/XiaoanNew_data"
YOLO_WEIGHTS = "/root/XiaoanNew/assets/weights/new_best.pt"
SCORING_YAML = "/root/XiaoanNew/assets/configs/scoring_new4d_gs_best.yaml"
PROMPT_ID = "cv_enhanced_v2_newdim_v2"
DEFAULT_OUT_DIR = "/root/autodl-tmp/XiaoanNew_outputs/benchmark_v4_current"
DEFAULT_WORKERS = 8
VLM_MAX_TOKENS = 1024

# ── 全局模型（一次性加载） ──
_yolo = None
_vlm_client = None
_scoring = None
_prompt_text = None
_vlm_model = None
_yolo_lock = threading.Lock()  # YOLO 推理串行化，避免 ultralytics 多线程竞态
_print_lock = threading.Lock()


# ──────────────────────────── 模型加载 ────────────────────────────


def _load_models():
    global _yolo, _vlm_client, _scoring, _prompt_text, _vlm_model
    print("[init] 加载 YOLOv8-Seg...", flush=True)
    _yolo = load_yolov8_seg(YOLO_WEIGHTS, device="cuda:0")  # conf_threshold=0.35 默认
    print("[init] 初始化 VLM client / ScoringEngine / prompt...", flush=True)
    _vlm_client = OpenAI(base_url=settings.API_BASE_URL, api_key=settings.VLM_API_KEY)
    _scoring = ScoringEngine.from_yaml(SCORING_YAML)
    _prompt_text = load_prompt(PROMPT_ID)
    _vlm_model = settings.VLM_MODEL
    # 预加载 DepthAssist 单例，避免多线程首调竞态
    try:
        print("[init] 预加载 DepthAssist...", flush=True)
        _get_depth_assist()
    except Exception as e:
        print(f"[init] DepthAssist 预加载失败（盲道深度检查将退化）: {e}", flush=True)
        # fail-fast: 置 sentinel 让后续 DepthAssist() 构造早返回、estimate() 立即抛错，
        # 避免每张盲道图重复 HF 5 次重试（~25s/次）拖慢批跑。
        # 结果与线上在此服务器一致：depth override 永不触发（仅 green_belt override 生效）。
        import modules.cv.depth_assist as _da_mod
        if _da_mod._DEPTH_PIPE is None:
            _da_mod._DEPTH_PIPE = False  # False is not None → __init__ 早返回
    print(
        f"[init] OK: vlm_model={_vlm_model}, prompt={PROMPT_ID}, "
        f"scoring=scoring_new4d_gs_best (threshold={_scoring.config.threshold})",
        flush=True,
    )


# ──────────────────────────── 规则降级判断（复刻 app.py） ────────────────────────────


def _rule_based_judgment(parking_lane: bool, curb: bool, tactile: bool):
    if tactile:
        return False, 0.3, "停车违规：检测到盲道"
    if parking_lane:
        return True, 0.7, "规范停车（检测到停车线）"
    if curb:
        return True, 0.65, "停车位置确认（检测到马路牙子）"
    return True, 0.5, "停车位置确认（车牌清晰）"


# ──────────────────────────── per-cluster 双解提取 ────────────────────────────


def _extract_line_results(objects, main_bike_mask, W, H):
    """复用 angle_inference 内部 helper 提取 per-cluster 双解结果。

    analyze_angle 仅暴露 line_results[0]，此处复算完整 list 以输出 judgment_1/angle_1/judgment_2/angle_2。
    逻辑与 analyze_angle 内部一致（pca + cluster + disambiguate + judge + 路缘降级）。
    """
    if main_bike_mask is None:
        return []
    bike_center, bike_dir = pca_2d_image(main_bike_mask)
    if bike_center is None or bike_dir is None:
        return []

    line_objects = [o for o in objects if o["label"] == "parking lane"]
    curb_mask = combine_masks(objects, "Curb")
    if curb_mask is not None and curb_mask.sum() < 50:
        curb_mask = None

    line_pcas = []
    for lo in line_objects:
        lm = lo.get("mask")
        if lm is None:
            continue
        c, d = pca_2d_image(lm)
        if c is not None and d is not None:
            line_pcas.append((c, d, lo))

    curb_pca = None
    if curb_mask is not None:
        cc, cd = pca_2d_image(curb_mask)
        if cc is not None and cd is not None:
            curb_pca = (cc, cd)

    clusters = cluster_lines(line_pcas)
    has_curb = curb_pca is not None

    line_results = []
    for cluster in clusters:
        ang_to_bike = angle_between(cluster["direction"], bike_dir)
        curb_angle_to_line = None
        if has_curb:
            curb_angle_to_line = angle_between(cluster["direction"], curb_pca[1])
        if has_curb and curb_angle_to_line is not None:
            std, disambig, summary = cv_disambiguate(ang_to_bike, curb_angle_to_line, has_curb)
        else:
            std, disambig, summary = cv_disambiguate(ang_to_bike, None, False)
        cvj, _ = cv_judge_angle(ang_to_bike, std)
        line_results.append({
            "angle_to_bike": round(ang_to_bike, 1),
            "std": std,
            "disambiguation": disambig,
            "cv_judgment": cvj,
            "summary": summary,
        })

    # 路缘降级：无标线但有路缘
    if not line_results and curb_pca is not None:
        ang_to_curb = angle_between(curb_pca[1], bike_dir)
        curb_cvj = "[合规]" if ang_to_curb >= 60 else "[不合规-斜停]"
        line_results.append({
            "angle_to_bike": round(ang_to_curb, 1),
            "std": "60-90°",
            "disambiguation": "无标线降级路缘(默认垂直车位)",
            "cv_judgment": curb_cvj,
            "summary": "路缘降级",
        })
    return line_results


# ──────────────────────────── 单图处理 ────────────────────────────


def _empty_row(gid: str, src: str, gt_is_valid: bool) -> dict:
    """构造默认字段行"""
    return {
        "id": gid, "src": src,
        "gt_is_valid": gt_is_valid,
        "pred_is_valid": None, "pred_final_score": None, "pred_message": "",
        "vlm_position": "", "vlm_medium": "", "vlm_angle": "", "vlm_state": "",
        "vlm_position_conf": "", "vlm_medium_conf": "", "vlm_angle_conf": "", "vlm_state_conf": "",
        "dim_score_position": "", "dim_score_medium": "", "dim_score_angle": "", "dim_score_state": "",
        "det_electric_bike": False, "det_curb": False, "det_parking_lane": False,
        "det_tactile": False, "det_green_belt": False,
        "class_count_electric_bike": 0, "class_count_curb": 0,
        "class_count_parking_lane": 0, "class_count_tactile": 0, "class_count_green_belt": 0,
        "iou_parking": 0.0, "overlap_parking": 0.0, "overlap_tactile": 0.0, "overlap_green_belt": 0.0,
        "cv_angle_judgment": "[N/A]", "cv_angle_to_bike": 0.0, "cv_curb_fallback": False,
        "cv_n_line_types": 0, "cv_disambiguation": "",
        "cv_judgment_1": "", "cv_judgment_2": "", "cv_angle_1": "", "cv_angle_2": "",
        "blind_override": False, "blind_iou": 0.0, "blind_depth_diff": "", "blind_reason": "",
        "green_belt_override": False, "green_belt_overlap": 0.0, "green_belt_reason": "",
        "vlm_raw_response": "",
        "error": "",
    }


def _process_one(item: dict) -> tuple[dict, dict]:
    """处理单张图，返回 (csv_row, json_full)。"""
    gid = item["id"]
    src = item["src"]
    gt_is_valid = (item.get("gt", "").lower() == "yes")

    row = _empty_row(gid, src, gt_is_valid)
    row.update({
        "gt_position": item.get("position", ""),
        "gt_medium": item.get("medium", ""),
        "gt_angle": item.get("angle", ""),
        "gt_state": item.get("state", ""),
    })
    full = dict(row)
    full["cv_detections"] = []
    full["vlm_raw_response_full"] = ""
    full["cv_summary"] = ""
    full["cv_line_results"] = []

    img_path = os.path.join(IMG_ROOT, src)
    try:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
    except Exception as e:
        row["error"] = f"read_img_fail: {e}"
        full.update({k: row[k] for k in row})
        return row, full

    # ── 预处理：裁剪下方区域 + 压缩（复刻 app.py 399-440） ──
    processed_bytes = img_bytes
    try:
        pil_image = Image.open(io.BytesIO(img_bytes))
        w, h = pil_image.size
        if w > 0:
            box_h = w / 3.0
            center_y = h * 0.7
            y1 = max(0, center_y - box_h / 2)
            y2 = min(h, center_y + box_h / 2)
            cropped = pil_image.crop((0, y1, w, y2))
            buf = io.BytesIO()
            fmt = pil_image.format if pil_image.format else "JPEG"
            cropped.save(buf, format=fmt)
            processed_bytes = buf.getvalue()
            processed_bytes = compress_image(processed_bytes)  # 长边1024 JPEG q85
    except Exception as e:
        processed_bytes = img_bytes  # 裁剪失败用原图

    # ── YOLO 分割（串行化） ──
    try:
        with _yolo_lock:
            seg_result = _yolo.predict(
                processed_bytes, visual=False, retina_masks=True, max_input_size=1280
            )
    except Exception as e:
        row["error"] = f"yolo_fail: {e}"
        full.update({k: row[k] for k in row})
        return row, full

    raw_img = seg_result["image_raw"]
    objects = seg_result["objects"]
    H, W = seg_result["image_size"]

    # ── 主车选取 + class_counts + found 标志 ──
    class_counts = {
        "Electric bike": 0, "Curb": 0, "parking lane": 0,
        "Tactile paving": 0, "Green belt": 0,
    }
    parking_lane_found = curb_found = tactile_found = green_belt_found = False
    main_bike_mask = None
    main_bike_center_dist = float("inf")
    img_cx, img_cy = W / 2.0, H / 2.0
    cv_detections = []

    for obj in objects:
        label = obj["label"]
        cv_detections.append({
            "id": obj["id"], "label": label,
            "confidence": obj["confidence"], "bbox": obj["bbox"],
        })
        if label in class_counts:
            class_counts[label] += 1
        if label == "parking lane":
            parking_lane_found = True
        elif label == "Curb":
            curb_found = True
        elif label == "Tactile paving":
            tactile_found = True
        elif label == "Green belt":
            green_belt_found = True
        if label == "Electric bike":
            bx1, by1, bx2, by2 = obj["bbox"]
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            cd = (bcx - img_cx) ** 2 + (bcy - img_cy) ** 2
            if cd < main_bike_center_dist:
                main_bike_center_dist = cd
                main_bike_mask = obj.get("mask")

    row["det_electric_bike"] = main_bike_mask is not None
    row["det_curb"] = curb_found
    row["det_parking_lane"] = parking_lane_found
    row["det_tactile"] = tactile_found
    row["det_green_belt"] = green_belt_found
    row["class_count_electric_bike"] = class_counts["Electric bike"]
    row["class_count_curb"] = class_counts["Curb"]
    row["class_count_parking_lane"] = class_counts["parking lane"]
    row["class_count_tactile"] = class_counts["Tactile paving"]
    row["class_count_green_belt"] = class_counts["Green belt"]

    # ── 几何 overlap ──
    geo = {
        "main_vehicle_detected": main_bike_mask is not None,
        "overlap_with_parking_lane": 0.0,
        "iou_with_parking_lane": 0.0,
        "overlap_with_tactile_paving": 0.0,
        "overlap_with_green_belt": 0.0,
        "status_inference": "unknown",
    }
    if main_bike_mask is not None:
        parking_mask = combine_masks(objects, "parking lane")
        if parking_mask is not None:
            iou, overlap = calculate_iou_and_overlap(main_bike_mask, parking_mask)
            geo["iou_with_parking_lane"] = iou
            geo["overlap_with_parking_lane"] = overlap
        tactile_mask = combine_masks(objects, "Tactile paving")
        if tactile_mask is not None:
            _, overlap_t = calculate_iou_and_overlap(main_bike_mask, tactile_mask)
            geo["overlap_with_tactile_paving"] = overlap_t
        green_belt_mask = combine_masks(objects, "Green belt")
        if green_belt_mask is not None:
            _, overlap_g = calculate_iou_and_overlap(main_bike_mask, green_belt_mask)
            geo["overlap_with_green_belt"] = overlap_g
        if geo["overlap_with_parking_lane"] > 0.8:
            geo["status_inference"] = "Likely Compliant (High Overlap)"
        elif geo["overlap_with_parking_lane"] < 0.1:
            geo["status_inference"] = "Likely Out of Bounds"

    row["iou_parking"] = geo["iou_with_parking_lane"]
    row["overlap_parking"] = geo["overlap_with_parking_lane"]
    row["overlap_tactile"] = geo["overlap_with_tactile_paving"]
    row["overlap_green_belt"] = geo["overlap_with_green_belt"]

    # 绿化带 override 决策（在 VLM 之前算好，VLM 后应用）
    green_belt_overlap = geo.get("overlap_with_green_belt", 0.0)
    green_belt_override = green_belt_overlap >= 0.01

    # ── CV 角度分析 ──
    cv_angle_result = {
        "cv_judgment": "[N/A]", "curb_fallback": False, "angle_to_bike": 0.0,
        "disambiguation": "", "n_line_types": 0, "summary": "",
    }
    if main_bike_mask is not None:
        try:
            cv_angle_result = analyze_angle(objects, main_bike_mask, W, H)
        except Exception as e:
            pass  # 保留默认
    row["cv_angle_judgment"] = cv_angle_result.get("cv_judgment", "[N/A]")
    row["cv_angle_to_bike"] = cv_angle_result.get("angle_to_bike", 0.0)
    row["cv_curb_fallback"] = cv_angle_result.get("curb_fallback", False)
    row["cv_n_line_types"] = cv_angle_result.get("n_line_types", 0)
    row["cv_disambiguation"] = cv_angle_result.get("disambiguation", "")
    full["cv_summary"] = cv_angle_result.get("summary", "")

    # 双解（per-cluster）
    line_results = _extract_line_results(objects, main_bike_mask, W, H)
    full["cv_line_results"] = line_results
    if len(line_results) >= 1:
        row["cv_judgment_1"] = line_results[0]["cv_judgment"]
        row["cv_angle_1"] = line_results[0]["angle_to_bike"]
    if len(line_results) >= 2:
        row["cv_judgment_2"] = line_results[1]["cv_judgment"]
        row["cv_angle_2"] = line_results[1]["angle_to_bike"]

    # ── VLM + 评分 + override ──
    is_valid_parking = None
    confidence = 0.0
    message = ""
    vlm_raw = ""
    blind_result = {"iou": 0.0, "depth_diff": None, "override": False, "reason": ""}

    try:
        vis_img = draw_wireframe_visual(raw_img, objects)
        b64_raw = encode_image_to_base64(raw_img)
        b64_vis = encode_image_to_base64(vis_img)

        detection_info = {
            "image_size": [H, W],
            "detected_objects": cv_detections,
            "class_summary": class_counts,
            "geometry_analysis": {
                **geo,
                "cv_angle_judgment": cv_angle_result.get("cv_judgment", "[N/A]"),
                "cv_disambiguation": cv_angle_result.get("disambiguation", ""),
                "cv_curb_fallback": cv_angle_result.get("curb_fallback", False),
            },
        }
        structured_info = json.dumps(detection_info, ensure_ascii=False, indent=2)
        full_prompt = (
            _prompt_text
            + "\n\n# YOLOv8-Seg Detection & Geometry Analysis\n```json\n"
            + structured_info
            + "\n```"
        )

        vlm_resp = chat_completion_with_retry(
            _vlm_client,
            model=_vlm_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_raw}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_vis}"}},
                ],
            }],
            max_tokens=VLM_MAX_TOKENS,
        )
        vlm_raw = vlm_resp.choices[0].message.content.strip()
        vlm_result = parse_vlm_response(vlm_raw)

        # 盲道 override（VLM 后、scoring 前）
        if main_bike_mask is not None:
            try:
                pil_image_for_depth = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                blind_result = check_blind_lane(main_bike_mask, objects, pil_image_for_depth)
            except Exception as e:
                blind_result = {
                    "iou": 0.0, "depth_diff": None, "override": False,
                    "reason": f"check_blind_lane异常: {e}",
                }
        if blind_result["override"]:
            vlm_result.medium = "[不合规-盲道]"

        # 绿化带 override（覆盖 blind_result，与 app.py 一致）
        if green_belt_override:
            vlm_result.medium = "[不合规-绿化]"
            blind_result["override"] = True
            blind_result["reason"] = f"车辆压绿化带 overlap_ratio={green_belt_overlap:.3f}"

        if vlm_result.is_valid:
            score_result = _scoring.score(*vlm_result.statuses)
            is_valid_parking = score_result.is_compliant
            confidence = score_result.final_score
            if is_valid_parking:
                message = f"合规停车（综合评分 {confidence:.2f}）"
            else:
                dims_fail = [k for k, v in score_result.dimension_scores.items() if v < 0.5]
                message = f"停车违规：{', '.join(dims_fail) if dims_fail else '综合评分不足'}"
            row["vlm_position"] = vlm_result.position
            row["vlm_medium"] = vlm_result.medium
            row["vlm_angle"] = vlm_result.angle
            row["vlm_state"] = vlm_result.state
            row["vlm_position_conf"] = vlm_result.position_confidence
            row["vlm_medium_conf"] = vlm_result.medium_confidence
            row["vlm_angle_conf"] = vlm_result.angle_confidence
            row["vlm_state_conf"] = vlm_result.state_confidence
            for dim in ("position", "medium", "angle", "state"):
                row[f"dim_score_{dim}"] = score_result.dimension_scores.get(dim, "")
        else:
            is_valid_parking, confidence, message = _rule_based_judgment(
                parking_lane_found, curb_found, tactile_found
            )
            row["error"] = f"vlm_parse_fail: {vlm_result.parse_error}"
    except Exception as e:
        is_valid_parking, confidence, message = _rule_based_judgment(
            parking_lane_found, curb_found, tactile_found
        )
        row["error"] = (row.get("error", "") + f" | vlm_fail: {e}").strip(" |")

    # seg_result 为空等的兜底（与 app.py 一致）
    if is_valid_parking is None:
        is_valid_parking = True
        confidence = 0.5
        message = "停车位置确认（AI 引擎不可用，仅凭车牌判断）"

    row["pred_is_valid"] = bool(is_valid_parking)
    row["pred_final_score"] = confidence
    row["pred_message"] = message
    row["blind_override"] = blind_result["override"]
    row["blind_iou"] = blind_result["iou"]
    row["blind_depth_diff"] = blind_result["depth_diff"] if blind_result["depth_diff"] is not None else ""
    row["blind_reason"] = blind_result["reason"]
    row["green_belt_override"] = green_belt_override
    row["green_belt_overlap"] = green_belt_overlap
    row["green_belt_reason"] = (
        f"车辆压绿化带 overlap_ratio={green_belt_overlap:.3f}" if green_belt_override else ""
    )
    row["vlm_raw_response"] = vlm_raw[:200] if vlm_raw else ""

    full.update({k: row[k] for k in row})
    full["vlm_raw_response_full"] = vlm_raw
    full["cv_detections"] = cv_detections
    return row, full


# ──────────────────────────── 输出字段顺序 ────────────────────────────

CSV_FIELDS = [
    "id", "src",
    "gt_position", "gt_medium", "gt_angle", "gt_state", "gt_is_valid",
    "pred_is_valid", "pred_final_score", "pred_message",
    "vlm_position", "vlm_medium", "vlm_angle", "vlm_state",
    "vlm_position_conf", "vlm_medium_conf", "vlm_angle_conf", "vlm_state_conf",
    "dim_score_position", "dim_score_medium", "dim_score_angle", "dim_score_state",
    "det_electric_bike", "det_curb", "det_parking_lane", "det_tactile", "det_green_belt",
    "class_count_electric_bike", "class_count_curb", "class_count_parking_lane",
    "class_count_tactile", "class_count_green_belt",
    "iou_parking", "overlap_parking", "overlap_tactile", "overlap_green_belt",
    "cv_angle_judgment", "cv_angle_to_bike", "cv_curb_fallback", "cv_n_line_types",
    "cv_disambiguation", "cv_judgment_1", "cv_judgment_2", "cv_angle_1", "cv_angle_2",
    "blind_override", "blind_iou", "blind_depth_diff", "blind_reason",
    "green_belt_override", "green_belt_overlap", "green_belt_reason",
    "vlm_raw_response", "error",
]


def _write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            if r is None:
                continue  # 未完成的占位行跳过
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def _write_json(fulls, path):
    out = [f for f in fulls if f is not None]  # 跳过未完成的占位行
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ──────────────────────────── summary 统计 ────────────────────────────


def _build_summary(rows, fulls, elapsed, n_total):
    n = len(rows)
    gt_yes = sum(1 for r in rows if r.get("gt_is_valid"))
    gt_no = n - gt_yes

    # 整体二分类（pred_is_valid is None 视为无效，排除）
    tp = tn = fp = fn = invalid = 0
    for r in rows:
        if r.get("pred_is_valid") is None:
            invalid += 1
            continue
        gt_yes_i = bool(r.get("gt_is_valid"))
        pred_yes = bool(r["pred_is_valid"])
        if gt_yes_i:
            if pred_yes:
                tp += 1
            else:
                fn += 1
        else:
            if pred_yes:
                fp += 1
            else:
                tn += 1
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    pre = tp / (tp + fp) if (tp + fp) else 0.0
    rec_comp = tp / (tp + fn) if (tp + fn) else 0.0  # 合规召回 TPR
    rec_viol = tn / (tn + fp) if (tn + fp) else 0.0  # 违规召回 TNR
    bal_acc = (rec_comp + rec_viol) / 2
    f1 = 2 * pre * rec_comp / (pre + rec_comp) if (pre + rec_comp) else 0.0

    # 四维 exact-match acc
    dim_stats = {}
    for dim, gt_key, pred_key in [
        ("position", "gt_position", "vlm_position"),
        ("medium", "gt_medium", "vlm_medium"),
        ("angle", "gt_angle", "vlm_angle"),
        ("state", "gt_state", "vlm_state"),
    ]:
        match = miss = skip = 0
        for r in rows:
            gt_v = r.get(gt_key, "")
            pred_v = r.get(pred_key, "")
            if not gt_v or not pred_v:
                skip += 1
                continue
            if gt_v == pred_v:
                match += 1
            else:
                miss += 1
        denom = match + miss
        dim_stats[dim] = (match / denom if denom else 0.0, match, miss, denom, skip)

    # 检出率
    det_stats = {}
    for label, key in [
        ("Electric bike", "det_electric_bike"),
        ("Curb", "det_curb"),
        ("parking lane", "det_parking_lane"),
        ("Tactile paving", "det_tactile"),
        ("Green belt", "det_green_belt"),
    ]:
        cnt = sum(1 for r in rows if r.get(key))
        det_stats[label] = (cnt, n)

    # override / 降级统计
    blind_override_cnt = sum(1 for r in rows if r.get("blind_override"))
    green_belt_override_cnt = sum(1 for r in rows if r.get("green_belt_override"))
    curb_fallback_cnt = sum(1 for r in rows if r.get("cv_curb_fallback"))
    disambig_ok = sum(
        1 for r in rows
        if ("垂直车位" in (r.get("cv_disambiguation") or "")
            or "平行车位" in (r.get("cv_disambiguation") or ""))
    )
    disambig_fail = sum(
        1 for r in rows
        if ("无法消歧" in (r.get("cv_disambiguation") or "")
            or "无路缘" in (r.get("cv_disambiguation") or "")
            or "无标线" in (r.get("cv_disambiguation") or "")
            or "路缘降级" in (r.get("cv_disambiguation") or ""))
    )

    L = []
    L.append("# Benchmark v4 当前上线管线实验汇总\n")
    L.append(f"- 总样本: {n}（GT 总数 {n_total}）")
    L.append(f"- GT 分布: yes={gt_yes}, no={gt_no}")
    L.append(f"- 无效预测（YOLO/读图失败等）: {invalid}")
    L.append(f"- 耗时: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    L.append("")
    L.append("## 整体二分类指标")
    L.append(f"- Accuracy: {acc:.4f} ({tp + tn}/{total})")
    L.append(f"- Balanced Accuracy: {bal_acc:.4f}")
    L.append(f"- Precision: {pre:.4f}")
    L.append(f"- 合规召回 CompRec (TPR): {rec_comp:.4f} ({tp}/{tp + fn})")
    L.append(f"- 违规召回 ViolRec (TNR): {rec_viol:.4f} ({tn}/{tn + fp})")
    L.append(f"- F1: {f1:.4f}")
    L.append(f"- 混淆矩阵: TP={tp} TN={tn} FP={fp} FN={fn}")
    L.append("")
    L.append("## 四维 Exact-Match Accuracy")
    L.append("| 维度 | Acc | match | miss | skip(空预测) |")
    L.append("|---|---|---|---|---|")
    for dim in ("position", "medium", "angle", "state"):
        a, m, ms, d, sk = dim_stats[dim]
        L.append(f"| {dim} | {a:.4f} | {m} | {ms} | {sk} |")
    L.append("")
    L.append("## 检测类别检出率")
    L.append("| 类别 | found | / | total | rate |")
    L.append("|---|---|---|---|---|")
    for label in ("Electric bike", "Curb", "parking lane", "Tactile paving", "Green belt"):
        c, t = det_stats[label]
        rate = c / t if t else 0.0
        L.append(f"| {label} | {c} | / | {t} | {rate:.4f} |")
    L.append("")
    L.append("## Override 与降级统计")
    L.append(f"- 盲道 override 触发（blind_override）: {blind_override_cnt}")
    L.append(f"- 绿化带 override 触发（green_belt_override）: {green_belt_override_cnt}")
    L.append(f"- 路缘降级（cv_curb_fallback）: {curb_fallback_cnt}")
    L.append(f"- 消歧成功（垂直/平行车位）: {disambig_ok}")
    L.append(f"- 消歧失败/降级（无法消歧/无路缘/无标线/路缘降级）: {disambig_fail}")
    L.append("")
    L.append("## 复刻说明")
    L.append("- 忠实复刻 app.py check_parking (399-720)，含下方区域裁剪（center_y=h*0.7, box_h=w/3）+ compress_image 预处理。")
    L.append("- YOLO: predict(visual=False, retina_masks=True, max_input_size=1280)，conf=0.35, iou=0.7。")
    L.append("- VLM: 两图（原图+线框图）+ prompt(cv_enhanced_v2_newdim_v2)，max_tokens=1024。")
    L.append("- Scoring: scoring_new4d_gs_best.yaml (weights pos=0.15/med=0.45/ang=0.3/state=0.1, threshold=0.6)。")
    L.append("- check_blind_lane 用原图做 depth 估计，但 main_bike_mask/blind_mask 在裁剪后空间 → 非宽图场景 depth 索引会异常 → blind depth override 实际很少触发（与上线管线一致）。")
    L.append("- green_belt override 覆盖 blind_result['override']/['reason']，与 app.py 行为一致。")
    L.append("- 双解字段（cv_judgment_1/2, cv_angle_1/2）由 _extract_line_results 复算 per-cluster 得到。")

    return "\n".join(L)


# ──────────────────────────── 主流程 ────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Benchmark v4 当前上线管线实验")
    ap.add_argument("--smoke", type=int, default=0, help="只跑前 N 张（0=全量）")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="VLM 并发线程数")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    ap.add_argument("--gt", default=GT_PATH, help="GT json 路径")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _load_models()

    with open(args.gt, "r", encoding="utf-8") as f:
        gt_list = json.load(f)
    n_total = len(gt_list)
    if args.smoke > 0:
        gt_list = gt_list[:args.smoke]
    print(f"[run] 样本: {len(gt_list)} (GT 总 {n_total}), workers={args.workers}", flush=True)

    rows = [None] * len(gt_list)
    fulls = [None] * len(gt_list)
    done = [0]
    done_lock = threading.Lock()
    t0 = time.time()

    def _snapshot(elapsed):
        """周期性写出已完成的中间结果，防止进程被杀丢全部"""
        try:
            _write_csv(rows, os.path.join(args.out_dir, "v4_current_results.csv"))
            _write_json(fulls, os.path.join(args.out_dir, "v4_current_results.json"))
            with _print_lock:
                print(f"[snapshot] {done[0]}/{len(gt_list)} elapsed={elapsed:.0f}s 已写盘", flush=True)
        except Exception as e:
            print(f"[snapshot] 写盘失败: {e}", flush=True)

    def _task(i, item):
        try:
            r, ff = _process_one(item)
        except Exception as e:
            tb = traceback.format_exc()
            r = _empty_row(item["id"], item["src"], item.get("gt", "").lower() == "yes")
            r["error"] = f"task_fail: {e}"
            r["pred_is_valid"] = None
            ff = dict(r)
            ff["cv_detections"] = []
            ff["vlm_raw_response_full"] = ""
            ff["cv_summary"] = ""
            ff["cv_line_results"] = []
            ff["_traceback"] = tb
        rows[i] = r
        fulls[i] = ff
        with done_lock:
            done[0] += 1
            d = done[0]
            do_snap = (d % 100 == 0)
            if d % 20 == 0 or d <= 5 or d == len(gt_list):
                with _print_lock:
                    print(
                        f"[progress] {d}/{len(gt_list)} "
                        f"elapsed={time.time() - t0:.0f}s "
                        f"last_id={r['id'][:40]} err={'Y' if r.get('error') else 'N'}",
                        flush=True,
                    )
            if do_snap:
                _snapshot(time.time() - t0)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_task, i, item) for i, item in enumerate(gt_list)]
            for f in as_completed(futures):
                f.result()
    finally:
        # 即使中断也写出已完成的
        elapsed = time.time() - t0
        csv_path = os.path.join(args.out_dir, "v4_current_results.csv")
        json_path = os.path.join(args.out_dir, "v4_current_results.json")
        summary_path = os.path.join(args.out_dir, "summary.md")
        _write_csv(rows, csv_path)
        _write_json(fulls, json_path)
        summary = _build_summary(rows, fulls, elapsed, n_total)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\n[done] 耗时 {elapsed:.1f}s ({elapsed / 60:.1f}min)", flush=True)
        print(f"[out] CSV:    {csv_path}", flush=True)
        print(f"[out] JSON:   {json_path}", flush=True)
        print(f"[out] SUMMARY:{summary_path}", flush=True)
        print("\n" + summary, flush=True)


if __name__ == "__main__":
    main()
