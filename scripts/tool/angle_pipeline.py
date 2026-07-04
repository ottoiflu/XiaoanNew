#!/usr/bin/env python3
"""CV+VLM 联合角度判定全流程管线 — benchmark v4 全覆盖

对 benchmark_v4 每张图片执行：
  YOLO 检测 → CV PCA 分析 → 标线聚类 → 消歧 → CV 判定 → 绘制掩模图
  → VLM 调用 → 后处理 → Scoring → CSV 输出 → 统计

用法:
  .venv/bin/python3 scripts/tool/angle_pipeline.py
  .venv/bin/python3 scripts/tool/angle_pipeline.py --workers 4 --smoke 10
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, "/root/otto/XiaoanNew")

from modules.config.settings import settings
from modules.cv.yolov8_inference import load_yolov8_seg
from modules.cv.image_utils import combine_masks, encode_image_to_base64, calculate_iou_and_overlap
from modules.experiment.scoring import ScoringEngine

# -- Depth Anything (lazy init) --
_DEPTH_PIPE = None
def _get_depth_map(pil_img):
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        from transformers import pipeline
        _DEPTH_PIPE = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    depth = _DEPTH_PIPE(pil_img)["predicted_depth"]
    return depth.cpu().numpy()
from modules.prompt.manager import PromptManager
from modules.vlm.parser import parse_vlm_response
from modules.vlm.retry import chat_completion_with_retry
from openai import OpenAI

# ──────────────────────────── 路径常量 ────────────────────────────

GT_PATH = "/root/otto/XiaoanNew/data/benchmark/benchmark_v4/fourdim_gt_v4.json"
DATA_ROOT = "/root/otto/XiaoanNew/data"
SCORING_YAML = "/root/otto/XiaoanNew/assets/configs/scoring_new4d_gs_best.yaml"
DEFAULT_OUTPUT = "/root/otto/XiaoanNew/outputs/benchmark_output/v4/angle_pipeline_results.csv"

# ──────────────────────────── PCA 工具 ────────────────────────────


def pca_2d_image(mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """对二值 mask 做 PCA，返回 (中心点, 主轴方向单位向量)"""
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return None, None
    uv = np.column_stack([xs, ys]).astype(float)
    c = uv.mean(axis=0)
    _, _, Vt = np.linalg.svd(uv - c, full_matrices=False)
    return c, Vt[0]


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """两向量夹角（锐角，度）"""
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    a = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    return min(a, 180.0 - a)


def get_centermost_bike(objects: list[dict], W: float, H: float) -> dict | None:
    """取距画面中心最近的 Electric bike"""
    cx, cy = W / 2.0, H / 2.0
    best, bd = None, float("inf")
    for obj in objects:
        if obj["label"] != "Electric bike":
            continue
        bx1, by1, bx2, by2 = obj["bbox"]
        d = ((bx1 + bx2) / 2 - cx) ** 2 + ((by1 + by2) / 2 - cy) ** 2
        if d < bd:
            bd, best = d, obj
    return best


# ──────────────────────────── 标线聚类 ────────────────────────────


def cluster_lines(
    line_pcas: list[tuple[np.ndarray, np.ndarray, dict]],
) -> list[dict]:
    """将多条标线按主轴夹角聚类

    Args:
        line_pcas: [(center, direction, obj), ...] 各标线的 PCA 结果

    Returns:
        [{
            "direction": np.ndarray (均值方向),
            "angle_to_bike": float,
            "count": int,
        }, ...]
    """
    if not line_pcas:
        return []

    # 先两两分组：夹角 < 30° 归为同类
    groups: list[list[int]] = []
    used = set()
    for i in range(len(line_pcas)):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, len(line_pcas)):
            if j in used:
                continue
            _, d_i, _ = line_pcas[i]
            _, d_j, _ = line_pcas[j]
            ang = angle_between(d_i, d_j)
            if ang < 30.0:
                group.append(j)
                used.add(j)
        groups.append(group)

    results = []
    for group in groups:
        dirs = [line_pcas[i][1] for i in group]
        # 均值方向：对单位向量求和后归一化
        mean_dir = np.mean(dirs, axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-8)
        results.append({
            "direction": mean_dir,
            "count": len(group),
        })
    return results


# ──────────────────────────── CV 消歧 ────────────────────────────


def cv_disambiguate(
    line_angle_to_bike: float,
    curb_angle_to_line: float | None,
    has_curb: bool,
) -> tuple[str, str, str]:
    """CV 消歧：确定车位类型和合规标准

    Returns:
        (compliance_std, disambiguation, std_summary)
        - compliance_std: "60-90°" or "0-30°"
        - disambiguation: "垂直车位(线⊥路缘, 夹角XX°)" 等
        - std_summary: "垂直车位" or "平行车位" or "无法消歧"
    """
    if has_curb and curb_angle_to_line is not None:
        if curb_angle_to_line > 60.0:
            summary = "垂直车位"
            std = "60-90°"
            disambig = f"垂直车位(线路缘夹角{curb_angle_to_line:.1f}°)"
        elif curb_angle_to_line < 30.0:
            summary = "平行车位"
            std = "0-30°"
            disambig = f"平行车位(线路缘夹角{curb_angle_to_line:.1f}°)"
        else:
            summary = "无法消歧"
            std = "60-90°"
            disambig = f"无法消歧(线路缘夹角{curb_angle_to_line:.1f}°, 默认垂直)"
    else:
        summary = "无法消歧"
        std = "60-90°"
        disambig = "无路缘(默认垂直车位)"

    return std, disambig, summary


def cv_judge_angle(angle: float, std: str) -> tuple[str, str]:
    """按合规标准判定角度合规性

    Returns:
        (judgment, label)
        - judgment: "[合规]" or "[不合规-斜停]"
        - label: "合规" or "不合规"
    """
    if std == "60-90°":
        compliant = 60.0 <= angle <= 90.0
    elif std == "0-30°":
        compliant = 0.0 <= angle <= 30.0
    else:
        compliant = False

    if compliant:
        return "[合规]", "合规"
    return "[不合规-斜停]", "不合规"


# ──────────────────────────── 绘图 ────────────────────────────


def draw_pca_annotation(
    img: np.ndarray,
    bike_mask: np.ndarray | None,
    line_objects: list[dict],
    curb_mask: np.ndarray | None,
    bike_center: np.ndarray | None,
    bike_dir: np.ndarray | None,
    line_pcas: list[tuple[np.ndarray, np.ndarray, dict]],
    curb_pca: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """在原图上绘制半透明掩模和 PCA 主轴箭头"""
    H, W = img.shape[:2]
    overlay = np.zeros((H, W, 4), dtype=np.uint8)  # RGBA

    # 车 mask (绿)
    if bike_mask is not None:
        overlay[bike_mask, 0] = 0
        overlay[bike_mask, 1] = 255
        overlay[bike_mask, 2] = 0
        overlay[bike_mask, 3] = 80

    # 标线 mask (黄)
    for obj in line_objects:
        m = obj.get("mask")
        if m is not None:
            overlay[m, 0] = 255
            overlay[m, 1] = 255
            overlay[m, 2] = 0
            overlay[m, 3] = 80

    # 路缘 mask (橙)
    if curb_mask is not None:
        overlay[curb_mask, 0] = 0
        overlay[curb_mask, 1] = 165
        overlay[curb_mask, 2] = 255
        overlay[curb_mask, 3] = 80

    # 融合 overlay
    vis = img.copy().astype(float)
    alpha = overlay[:, :, 3:4] / 255.0
    vis = vis * (1 - alpha) + overlay[:, :, :3].astype(float) * alpha
    vis = vis.astype(np.uint8)

    # 转 PIL 画箭头
    pil_img = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil_img)

    def _draw_arrow(center, direction, color, length=150):
        if center is None or direction is None:
            return
        cx, cy = float(center[0]), float(center[1])
        dx, dy = direction[0], direction[1]
        norm = np.linalg.norm([dx, dy])
        if norm < 1e-8:
            return
        dx, dy = dx / norm * length, dy / norm * length
        # 起点终点
        x1, y1 = cx - dx, cy - dy
        x2, y2 = cx + dx, cy + dy
        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
        # 箭头头部 (两个小线段)
        head_len = 20
        angle_rad = np.arctan2(dy, dx)
        for sign in [1, -1]:
            hx = x2 - head_len * np.cos(angle_rad + sign * 0.45)
            hy = y2 - head_len * np.sin(angle_rad + sign * 0.45)
            draw.line([(x2, y2), (hx, hy)], fill=color, width=3)

    # 车 PCA 箭头 (绿)
    if bike_center is not None and bike_dir is not None:
        _draw_arrow(bike_center, bike_dir, (0, 200, 0), length=150)

    # 标线 PCA 箭头 (红) — 每条独立画
    for center, direction, _ in line_pcas:
        _draw_arrow(center, direction, (200, 0, 0), length=150)

    # 路缘 PCA 箭头 (橙)
    if curb_pca is not None and curb_pca[0] is not None and curb_pca[1] is not None:
        _draw_arrow(curb_pca[0], curb_pca[1], (0, 130, 200), length=150)

    return np.array(pil_img)


def resize_max(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """缩放至最长边不超过 max_size"""
    H, W = img.shape[:2]
    if max(H, W) <= max_size:
        return img
    scale = max_size / max(H, W)
    nh, nw = int(H * scale), int(W * scale)
    pil = Image.fromarray(img).resize((nw, nh), Image.LANCZOS)
    return np.array(pil)


def encode_to_base64_jpeg(img: np.ndarray, max_size: int = 1024, quality: int = 85) -> str:
    """编码为 base64 JPEG"""
    img_resized = resize_max(img, max_size)
    pil = Image.fromarray(img_resized)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ──────────────────────────── VLM ────────────────────────────


def build_cv_text(
    line_results: list[dict],
    disambiguation: str,
    curb_angle_to_line: float | None,
    has_curb: bool,
) -> str:
    """构造 CV 数据文本"""
    n_types = len(line_results)
    lines = ["# CV Angle Analysis"]
    lines.append(f"- 车线类型数: {n_types}")

    for i, lr in enumerate(line_results, 1):
        ang = lr["angle_to_bike"]
        std = lr.get("std", "")
        disambig = lr.get("disambiguation", "")
        cvj = lr.get("cv_judgment", "")
        lines.append(f"- 标线类型{i}夹角: {ang:.1f}°")
        if disambig:
            lines.append(f"- 标线类型{i}消歧: {disambig}")
        if std:
            lines.append(f"- 标线类型{i}标准: {std}")
        if cvj:
            lines.append(f"- 标线类型{i}CV判定: {cvj}")

    lines.append(f"- 消歧总结: {disambiguation}")
    return "\n".join(lines)


def call_vlm(
    client: OpenAI,
    model: str,
    prompt_text: str,
    cv_text: str,
    raw_img: np.ndarray,
    mask_img: np.ndarray,
) -> str:
    """调用 VLM 获取四维判定"""
    raw_b64 = encode_to_base64_jpeg(raw_img, max_size=1024)
    mask_b64 = encode_to_base64_jpeg(mask_img, max_size=1024)

    combined_text = f"{prompt_text}\n\n{CV_TEXT_SEP}\n{cv_text}\n{CV_TEXT_SEP}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": combined_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{raw_b64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{mask_b64}"},
                },
            ],
        }
    ]

    resp = chat_completion_with_retry(
        client,
        model=model,
        messages=messages,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.95,
    )
    return resp.choices[0].message.content


CV_TEXT_SEP = "---" * 20


# ──────────────────────────── 单张处理 ────────────────────────────


def process_one(
    item: tuple,
    seg,
    prompt_text: str,
    vlm_client: OpenAI,
    vlm_model: str,
    scoring: ScoringEngine,
) -> dict:
    """处理单张图片，返回结果字典"""
    entry, img_path = item
    t_start = time.time()
    result = {
        "id": entry["id"],
        "gt": entry.get("gt", ""),
        "vlm_position": "",
        "vlm_medium": "",
        "vlm_angle": "",
        "vlm_state": "",
        "final_position": "",
        "final_angle": "",
        "final_score": "",
        "final_pred": "",
        "cv_n_line_types": "",
        "cv_angle_1": "",
        "cv_std_1": "",
        "cv_judgment_1": "",
        "cv_disambiguation_1": "",
        "cv_angle_2": "",
        "cv_std_2": "",
        "cv_judgment_2": "",
        "cv_disambiguation_2": "",
        "cv_disambiguation_summary": "",
        "cv_line_curb_angle": "",
        "vlm_reason": "",
        "latency": "",
        "error": "",
    }

    try:
        # ── Step 1: YOLO 检测 ──
        img_bytes = open(img_path, "rb").read()
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        # 图片压缩和统一尺寸（长边 <= 1024，减少 VLM 传输量）
        MAX_SIZE = 1024
        if max(W, H) > MAX_SIZE:
            scale = MAX_SIZE / max(W, H)
            pil = pil.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
            W, H = pil.size
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=85)
        img_bytes = buf.getvalue()
        yolo_result = seg.predict(img_bytes, visual=False, retina_masks=True)
        objects = yolo_result["objects"]

        # 最居中车
        bike_obj = get_centermost_bike(objects, W, H)
        if bike_obj is None:
            raise ValueError("未检测到 Electric bike")

        bike_mask_obj = bike_obj.get("mask")
        if bike_mask_obj is None:
            raise ValueError("Electric bike 无 mask")

        # 各停车标线独立实例 & 路缘
        line_objects = [o for o in objects if o["label"] == "parking lane"]
        curb_mask = combine_masks(objects, "Curb")
        if curb_mask is not None and curb_mask.sum() < 50:
            curb_mask = None

        # ── Step 2: PCA ──
        bike_center, bike_dir = pca_2d_image(bike_mask_obj)
        if bike_center is None or bike_dir is None:
            raise ValueError("车辆 PCA 失败")

        line_pcas = []  # [(center, dir, obj), ...]
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

        # ── Step 3: 标线聚类 ──
        clusters = cluster_lines(line_pcas)

        # ── Steps 4-5: 消歧 & CV 判定（每类标线分别消歧） ──
        has_curb = curb_pca is not None
        curb_angle_to_line = None

        line_results = []
        for cluster in clusters:
            ang_to_bike = angle_between(cluster["direction"], bike_dir)

            # 每类标线分别和路缘算夹角消歧
            curb_angle_to_line = None
            if has_curb:
                curb_angle_to_line = angle_between(cluster["direction"], curb_pca[1])

            if has_curb and curb_angle_to_line is not None:
                std, disambig, summary = cv_disambiguate(
                    ang_to_bike, curb_angle_to_line, has_curb
                )
            else:
                std, disambig, summary = cv_disambiguate(
                    ang_to_bike, None, False
                )
            cvj, _ = cv_judge_angle(ang_to_bike, std)
            line_results.append({
                "angle_to_bike": round(ang_to_bike, 1),
                "std": std,
                "disambiguation": disambig,
                "cv_judgment": cvj,
                "summary": summary,
            })

        # 消歧总结（统一用第一类）
        disambig_summary = line_results[0]["summary"] if line_results else "无标线"
        # ── 路缘降级：无标线但有路缘时用路缘作基准轴 ──
        if not line_results and curb_pca is not None:
            ang_to_curb = angle_between(curb_pca[1], bike_dir)
            curb_cvj = "[合规]" if ang_to_curb >= 60 else "[不合规-斜停]"
            line_results.append({
                "angle_to_bike": round(ang_to_curb, 1),
                "std": "60-90\u00b0",
                "disambiguation": "无标线降级路缘(默认垂直车位)",
                "cv_judgment": curb_cvj,
                "summary": "路缘降级",
            })
            disambig_summary = "路缘降级(无标线, 用路缘作基准轴)"

        # ── Step 6: 绘制掩模图 ──
        raw_img = np.array(pil)
        mask_img = draw_pca_annotation(
            raw_img,
            bike_mask=bike_mask_obj,
            line_objects=line_objects,
            curb_mask=curb_mask,
            bike_center=bike_center,
            bike_dir=bike_dir,
            line_pcas=line_pcas,
            curb_pca=curb_pca,
        )

        # ── Step 7: VLM ──
        cv_text = build_cv_text(line_results, disambig_summary, curb_angle_to_line, has_curb)
        vlm_raw = call_vlm(vlm_client, vlm_model, prompt_text, cv_text, raw_img, mask_img)
        parsed = parse_vlm_response(vlm_raw)

        vlm_pos = parsed.position
        vlm_med = parsed.medium
        vlm_ang = parsed.angle
        vlm_st = parsed.state
        vlm_reason = parsed.reason[:300] if parsed.reason else ""

        # -- blind lane detection (2D IoU + Depth Anything) --
        blind_override = False
        blind_iou_val = ""
        blind_depth_diff = ""
        try:
            blind_mask_ = combine_masks(objects, "Tactile paving")
            if bike_mask_obj is not None and blind_mask_ is not None:
                iou_v, _ = calculate_iou_and_overlap(bike_mask_obj, blind_mask_)
                blind_iou_val = "%.6f" % iou_v
                if iou_v >= 0.01:
                    depth_map = _get_depth_map(pil)
                    car_region_ = np.zeros_like(bike_mask_obj, dtype=bool)
                    rows_ = np.any(bike_mask_obj, axis=1)
                    if rows_.any():
                        top_ = max(0, int(bike_mask_obj.shape[0] * 0.5))
                        car_region_[top_:, :] = bike_mask_obj[top_:, :]
                    car_depth = float(np.median(depth_map[car_region_ & bike_mask_obj]))
                    blind_depth = float(np.median(depth_map[blind_mask_]))
                    blind_depth_diff = "%.6f" % abs(car_depth - blind_depth)
                    if abs(car_depth - blind_depth) < 0.1:
                        blind_override = True
        except Exception:
            pass
        if blind_override:
            vlm_med = "[incompliant-blind]"
            prefix_ = "[blind] IoU=" + blind_iou_val + " depth_diff=" + blind_depth_diff + " | "
            vlm_reason = prefix_ + vlm_reason

        # ── Step 8: 后处理 ──
        if vlm_med.startswith("[不合规") or vlm_med.startswith("不合规"):
            final_pos = "[无参照]"
            final_ang = "[N/A]"
        else:
            final_pos = vlm_pos
            final_ang = vlm_ang

        # ── Scoring ──
        sr = scoring.score(
            position=final_pos,
            medium=vlm_med,
            angle=final_ang,
            state=vlm_st,
        )
        final_pred = "yes" if sr.is_compliant else "no"

        # ── 填充结果 ──
        result.update({
            "vlm_position": vlm_pos,
            "vlm_medium": vlm_med,
            "vlm_angle": vlm_ang,
            "vlm_state": vlm_st,
            "final_position": final_pos,
            "final_angle": final_ang,
            "final_score": f"{sr.final_score:.4f}",
            "final_pred": final_pred,
            "cv_n_line_types": str(len(line_results)),
            "cv_disambiguation_summary": disambig_summary,
            "cv_line_curb_angle": f"{curb_angle_to_line:.1f}" if curb_angle_to_line is not None else "",
            "vlm_reason": vlm_reason,
            "blind_iou": blind_iou_val,
            "blind_depth_diff": blind_depth_diff,
            "blind_override": "1" if blind_override else "",
            "latency": f"{time.time() - t_start:.2f}",
        })

        # 标线具体信息
        if len(line_results) >= 1:
            lr1 = line_results[0]
            result["cv_angle_1"] = str(lr1["angle_to_bike"])
            result["cv_std_1"] = lr1["std"]
            result["cv_judgment_1"] = lr1["cv_judgment"]
            result["cv_disambiguation_1"] = lr1["disambiguation"]
        if len(line_results) >= 2:
            lr2 = line_results[1]
            result["cv_angle_2"] = str(lr2["angle_to_bike"])
            result["cv_std_2"] = lr2["std"]
            result["cv_judgment_2"] = lr2["cv_judgment"]
            result["cv_disambiguation_2"] = lr2["disambiguation"]

    except Exception as e:
        result["error"] = str(e)[:200]
        result["latency"] = f"{time.time() - t_start:.2f}"

    return result


# ──────────────────────────── 统计 ────────────────────────────


def calc_metrics(rows: list[dict]) -> dict:
    """计算 Acc / ViolRec / 混淆矩阵"""
    tp = tn = fp = fn = 0
    for r in rows:
        g = r.get("gt", "").strip().lower()
        p = r.get("final_pred", "").strip().lower()
        if g == "yes":
            if p == "yes":
                tp += 1
            else:
                fn += 1
        else:
            if p == "no":
                tn += 1
            else:
                fp += 1
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    viol_rec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "total": total,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": acc, "viol_rec": viol_rec,
    }


def dim_accuracy(
    rows: list[dict],
    gt_entries: dict,
    dim: str,
    pred_key: str,
) -> dict:
    """计算某维度的准确率"""
    correct = total = 0
    for r in rows:
        rid = r.get("id", "")
        g = gt_entries.get(rid, {})
        gt_val = g.get(dim, "")
        pred_val = r.get(pred_key, "")
        if not gt_val or not pred_val:
            continue
        if pred_val in ("err", "parse_fail") or pred_val.startswith("0."):
            continue
        total += 1
        if gt_val == pred_val:
            correct += 1
    return {"correct": correct, "total": total, "acc": correct / total if total else 0.0}


def print_stats(metrics: dict, label: str):
    print(f"\n{'=' * 40}")
    print(f" {label}")
    print(f"{'=' * 40}")
    print(f"  Acc: {metrics['acc']:.3f}  ViolRec: {metrics['viol_rec']:.3f}")
    print(f"  TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"  总数: {metrics['total']}")


# ──────────────────────────── Main ────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="CV+VLM 联合角度判定管线")
    parser.add_argument("--workers", type=int, default=32, help="并发线程数")
    parser.add_argument("--smoke", type=int, default=0, help="冒烟测试张数 (0=全部)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="输出 CSV 路径")
    args = parser.parse_args()

    # ── 加载数据 ──
    print("[Load] 加载 benchmark gt ...")
    gt_entries = {x["id"]: x for x in json.load(open(GT_PATH, "r", encoding="utf-8"))}
    print(f"  gt 条目: {len(gt_entries)}")

    # ── 构建图片路径列表 ──
    items = []
    missing = 0
    for eid, entry in gt_entries.items():
        src = entry.get("src", "")
        if not src:
            missing += 1
            continue
        img_path = os.path.join(DATA_ROOT, src)
        if not os.path.exists(img_path):
            # 尝试扩展名
            found = False
            for ext in (".JPEG", ".jpeg", ".png", ".jpg"):
                alt = os.path.splitext(img_path)[0] + ext
                if os.path.exists(alt):
                    img_path = alt
                    found = True
                    break
            if not found:
                missing += 1
                continue
        items.append((entry, img_path))
    print(f"  有效图片: {len(items)}, 缺失: {missing}")

    if args.smoke > 0:
        items = items[:args.smoke]
        print(f"  冒烟测试: {args.smoke} 张")

    # ── 加载模型 ──
    print("[Load] 加载 YOLOv8-Seg ...")
    seg = load_yolov8_seg(settings.YOLO_WEIGHTS, device=settings.INFERENCE_DEVICE)

    print("[Load] 加载 VLM prompt ...")
    pm = PromptManager()
    prompt_text = pm.get_content("cv_enhanced_v2_newdim_v2")

    print("[Load] 加载 ScoringEngine ...")
    scoring = ScoringEngine.from_yaml(SCORING_YAML)

    print("[Load] 创建 VLM 客户端 ...")
    vlm_client = OpenAI(base_url=settings.API_BASE_URL, api_key=settings.VLM_API_KEY)
    vlm_model = settings.VLM_MODEL

    # ── 并发处理 ──
    print(f"\n[Run] 并发 {args.workers} 线程处理 {len(items)} 张图 ...")
    t0 = time.time()
    results = []
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_one,
                item,
                seg,
                prompt_text,
                vlm_client,
                vlm_model,
                scoring,
            ): item
            for item in items
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            r = f.result()
            if r.get("error"):
                errors += 1
            results.append(r)

    elapsed = time.time() - t0
    print(f"\n[完成] 耗时: {elapsed:.0f}s, 异常: {errors}")

    # ── 输出 CSV ──
    fieldnames = [
        "id", "gt",
        "vlm_position", "vlm_medium", "vlm_angle", "vlm_state",
        "final_position", "final_angle",
        "final_score", "final_pred",
        "cv_n_line_types",
        "cv_angle_1", "cv_std_1", "cv_judgment_1", "cv_disambiguation_1",
        "cv_angle_2", "cv_std_2", "cv_judgment_2", "cv_disambiguation_2",
        "cv_disambiguation_summary", "cv_line_curb_angle",
        "blind_iou", "blind_depth_diff", "blind_override",
        "vlm_reason", "latency", "error",
    ]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"[CSV] 已保存: {args.output} ({len(results)} 行)")

    # ── 统计 ──
    print("\n" + "=" * 50)
    print(" 管线结果统计")
    print("=" * 50)

    # Final 指标
    final_metrics = calc_metrics(results)
    print_stats(final_metrics, "Final (Scoring)")

    # VLM angle 逐维准确率
    print("\n-- VLM 逐维准确率 --")
    for dim, pk in [("position", "vlm_position"), ("medium", "vlm_medium"),
                     ("angle", "vlm_angle"), ("state", "vlm_state")]:
        da = dim_accuracy(results, gt_entries, dim, pk)
        print(f"  {dim:>10}: {da['correct']:>3}/{da['total']:<4} = {da['acc']:.3f}")

    # CV angle 逐维准确率
    print("\n-- CV Angle 逐维准确率 --")
    for dim, pk in [("angle", "cv_judgment_1")]:
        # CV 判定映射到 [合规]/[不合规-斜停]，gt angle 也是同样的标签
        da = dim_accuracy(results, gt_entries, dim, pk)
        print(f"  {dim:>10}: {da['correct']:>3}/{da['total']:<4} = {da['acc']:.3f}")

    # Final angle 逐维准确率 (从 scoring 的 raw_statuses 看，但我们有 final_angle)
    print("\n-- Final Angle 逐维准确率 --")
    da = dim_accuracy(results, gt_entries, "angle", "final_angle")
    print(f"  {'angle':>10}: {da['correct']:>3}/{da['total']:<4} = {da['acc']:.3f}")

    # 排除 N/A 后的 angle 准确率
    non_na_rows = [
        r for r in results
        if gt_entries.get(r["id"], {}).get("angle", "") != "[N/A]"
    ]
    c = sum(
        1 for r in non_na_rows
        if r.get("final_angle", "") == gt_entries.get(r["id"], {}).get("angle", "")
        and r.get("final_angle", "")
    )
    t = sum(1 for r in non_na_rows if r.get("final_angle", ""))
    print(f"\n  排除 N/A 后 angle: {c}/{t} = {c / t if t else 0:.3f}")

    print(f"\n  总耗时: {elapsed:.0f}s  均耗时: {elapsed / len(results):.2f}s/张" if results else "")


if __name__ == "__main__":
    main()