"""角度推断模块

封装 PCA 工具、标线聚类、消歧、CV 判定等纯函数，
以及一个入口函数 analyze_angle 串联整条链路。
"""

from __future__ import annotations

import numpy as np

from modules.cv.image_utils import combine_masks

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


# ──────────────────────────── CV 文本 ────────────────────────────


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


# ──────────────────────────── 入口函数 ────────────────────────────


def analyze_angle(
    objects: list[dict],
    main_bike_mask: np.ndarray | None,
    W: int,
    H: int,
) -> dict:
    """封装 PCA + 标线聚类 + 消歧 + 路缘降级整条链路

    Args:
        objects: [{label, mask, bbox, ...}, ...] YOLO 检测结果
        main_bike_mask: 主车二值 mask (H, W) bool
        W: 图像宽度
        H: 图像高度

    Returns:
        {
            "angle_to_bike": float,       # 车身-标线夹角（度），无标线路缘降级时为车身-路缘夹角
            "cv_judgment": str,            # "[合规]" / "[不合规-斜停]" / "[N/A]"
            "disambiguation": str,         # 消歧结论
            "curb_fallback": bool,         # 是否走了路缘降级
            "n_line_types": int,           # 标线类别数
            "summary": str,                # 总结
        }
    """
    if main_bike_mask is None:
        return {
            "angle_to_bike": 0.0,
            "cv_judgment": "[N/A]",
            "disambiguation": "",
            "curb_fallback": False,
            "n_line_types": 0,
            "summary": "",
        }

    # PCA on bike
    bike_center, bike_dir = pca_2d_image(main_bike_mask)
    if bike_center is None or bike_dir is None:
        return {
            "angle_to_bike": 0.0,
            "cv_judgment": "[N/A]",
            "disambiguation": "",
            "curb_fallback": False,
            "n_line_types": 0,
            "summary": "",
        }

    # 从 objects 提取标线和路缘
    line_objects = [o for o in objects if o["label"] == "parking lane"]
    curb_mask = combine_masks(objects, "Curb")
    if curb_mask is not None and curb_mask.sum() < 50:
        curb_mask = None

    # 标线 PCA
    line_pcas = []
    for lo in line_objects:
        lm = lo.get("mask")
        if lm is None:
            continue
        c, d = pca_2d_image(lm)
        if c is not None and d is not None:
            line_pcas.append((c, d, lo))

    # 路缘 PCA
    curb_pca = None
    if curb_mask is not None:
        cc, cd = pca_2d_image(curb_mask)
        if cc is not None and cd is not None:
            curb_pca = (cc, cd)

    # 标线聚类
    clusters = cluster_lines(line_pcas)
    has_curb = curb_pca is not None

    # 消歧 & CV 判定（每类标线分别消歧）
    line_results = []
    for cluster in clusters:
        ang_to_bike = angle_between(cluster["direction"], bike_dir)

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

    disambig_summary = line_results[0]["summary"] if line_results else "无标线"
    curb_fallback = False

    # 路缘降级：无标线但有路缘时用路缘作基准轴
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
        disambig_summary = "路缘降级(无标线, 用路缘作基准轴)"
        curb_fallback = True

    if not line_results:
        return {
            "angle_to_bike": 0.0,
            "cv_judgment": "[N/A]",
            "disambiguation": "无标线无路缘",
            "curb_fallback": False,
            "n_line_types": 0,
            "summary": disambig_summary,
        }

    return {
        "angle_to_bike": line_results[0]["angle_to_bike"],
        "cv_judgment": line_results[0]["cv_judgment"],
        "disambiguation": disambig_summary,
        "curb_fallback": curb_fallback,
        "n_line_types": len(clusters),
        "summary": build_cv_text(line_results, disambig_summary, None, has_curb),
    }
