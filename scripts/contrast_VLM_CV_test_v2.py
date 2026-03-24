"""VLM + CV 联合测试脚本

使用 YOLOv8-Seg 实例分割 + VLM 进行停车合规判定。
输入给 VLM 的是线框轮廓图，Python 端预计算 IoU/重叠率。
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from datetime import datetime

from PIL import Image
from tqdm import tqdm

# 项目根目录
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
from modules.experiment.config import load_config, save_config
from modules.experiment.io import (
    ResultWriter,
    append_summary,
    collect_image_tasks,
    load_all_labels,
)
from modules.experiment.metrics import calculate_metrics, print_metrics_report, update_leaderboard
from modules.experiment.scoring import ScoringEngine
from modules.prompt.manager import load_prompt
from modules.vlm.client import create_client_pool, distribute_tasks
from modules.vlm.parser import normalize_label, parse_vlm_response

# ================= 默认配置 =================

TEST_OUTPUT_ROOT = os.path.join(_PROJECT_ROOT, "outputs/test_outputs")

CONFIG = {
    "exp_name": "qwen3-vl-30b-a3b_contours_iou_fix",
    "model": "qwen/qwen3-vl-30b-a3b-instruct",
    "max_size": (768, 768),
    "quality": 80,
    "prompt_id": "cv_enhanced_p4",
    "scoring_config": None,
}

DATA_FOLDERS = [
    os.path.join(_PROJECT_ROOT, "data/Compliance_test_data/no_val"),
    os.path.join(_PROJECT_ROOT, "data/Compliance_test_data/yes_val"),
]

SEGMENTOR_CONFIG = {
    "weights": os.path.join(_PROJECT_ROOT, "assets/weights/best.pt"),
    "device": "cuda:0",
    "conf_threshold": 0.6,
}

# ================= 全局模型加载 =================

print("=" * 60)
print(">>> 正在加载 YOLOv8-Seg 模型...")
segmentor = load_yolov8_seg(
    weights_path=SEGMENTOR_CONFIG["weights"],
    device=SEGMENTOR_CONFIG["device"],
)
segmentor.conf_threshold = SEGMENTOR_CONFIG["conf_threshold"]
print(">>> YOLOv8-Seg 模型加载完成")
print("=" * 60)

# ================= 评判引擎 =================

_scoring_engine = None
if CONFIG.get("scoring_config"):
    cfg_path = os.path.join(_PROJECT_ROOT, CONFIG["scoring_config"])
    _scoring_engine = ScoringEngine.from_yaml(cfg_path)


def _judge(vlm_result):
    """根据 VLM 解析结果进行合规判定"""
    comp, ang, dist, cont = vlm_result.statuses
    if _scoring_engine:
        sr = _scoring_engine.score(comp, ang, dist, cont)
        return ("yes" if sr.is_compliant else "no"), sr.final_score
    # 一票否决
    return ScoringEngine.veto_judge(comp, ang, dist, cont), 0.0


# ================= 核心处理 =================

CSV_HEADERS = [
    "image",
    "folder",
    "pred",
    "gt",
    "composition",
    "angle",
    "distance",
    "context",
    "num_detections",
    "electric_bike",
    "curb",
    "parking_lane",
    "tactile_paving",
    "reason",
    "latency",
]


def process_single_image(args):
    """处理单张图片：YOLOv8-Seg + VLM"""
    image_name, folder_path, client, labels_dict, config = args
    image_path = os.path.join(folder_path, image_name)
    gt = labels_dict.get((image_name, folder_path), "N/A")
    start_t = time.time()

    try:
        # 1. YOLOv8-Seg 实例分割
        seg_result = segmentor.predict(image_path)
        raw_img = seg_result["image_raw"]
        objects = seg_result["objects"]
        H, W = seg_result["image_size"]

        # 2. 生成线框轮廓图
        vis_img = draw_wireframe_visual(raw_img, objects)
        vis_path = os.path.join(config["_vis_dir"], image_name)
        Image.fromarray(vis_img).save(vis_path)

        # 3. 编码图像
        b64_raw = encode_image_to_base64(raw_img, config["max_size"], config["quality"])
        b64_vis = encode_image_to_base64(vis_img, config["max_size"], config["quality"])

        # 4. 构造 CV 结构化信息
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

        # 几何关系计算
        geo = {
            "main_vehicle_detected": False,
            "overlap_with_parking_lane": 0.0,
            "iou_with_parking_lane": 0.0,
            "overlap_with_tactile_paving": 0.0,
            "status_inference": "unknown",
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
            if geo["overlap_with_parking_lane"] > 0.8:
                geo["status_inference"] = "Likely Compliant (High Overlap)"
            elif geo["overlap_with_parking_lane"] < 0.1:
                geo["status_inference"] = "Likely Out of Bounds"

        detection_info["geometry_analysis"] = geo
        structured_info = json.dumps(detection_info, ensure_ascii=False, indent=2)

        # 5. 组装 Prompt + 调用 VLM
        full_prompt = (
            load_prompt(config["prompt_id"])
            + "\n\n# YOLOv8-Seg Detection & Geometry Analysis\n```json\n"
            + structured_info
            + "\n```"
        )
        res = client.chat.completions.create(
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
            max_tokens=1000,
            temperature=0.1,
            top_p=0.9,
        )
        vlm_out = res.choices[0].message.content
        vlm_result = parse_vlm_response(vlm_out)

        if not vlm_result.is_valid:
            return [
                image_name,
                os.path.basename(folder_path),
                "error",
                gt,
                "fail",
                "fail",
                "fail",
                "fail",
                0,
                0,
                0,
                0,
                0,
                vlm_result.parse_error,
                0,
            ]

        pred, _ = _judge(vlm_result)
        return [
            image_name,
            os.path.basename(folder_path),
            pred,
            gt,
            vlm_result.composition,
            vlm_result.angle,
            vlm_result.distance,
            vlm_result.context,
            len(objects),
            class_counts.get("Electric bike", 0),
            class_counts.get("Curb", 0),
            class_counts.get("parking lane", 0),
            class_counts.get("Tactile paving", 0),
            vlm_result.reason,
            round(time.time() - start_t, 3),
        ]
    except Exception as e:
        import traceback

        traceback.print_exc()
        return [
            image_name,
            os.path.basename(folder_path),
            "error",
            gt,
            "err",
            "err",
            "err",
            "err",
            0,
            0,
            0,
            0,
            0,
            str(e),
            0,
        ]


# ================= 实验运行 =================


def run_experiment():
    """执行一次完整实验"""
    settings = get_settings()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建实验目录
    exp_dir = os.path.join(TEST_OUTPUT_ROOT, f"exp_{timestamp}_{CONFIG['exp_name']}")
    vis_dir = os.path.join(exp_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    # 注入 vis_dir 到 config 供 process_single_image 使用
    config = dict(CONFIG)
    config["_vis_dir"] = vis_dir

    print("\n>>> 实验启动 (YOLOv8-Seg + VLM)")
    print(f">>> 模型: {config['model']}  提示词: {config['prompt_id']}")
    print(f">>> 实验目录: {exp_dir}")

    clients = create_client_pool()
    global_labels = load_all_labels(DATA_FOLDERS)
    image_tasks = collect_image_tasks(DATA_FOLDERS)
    all_tasks = distribute_tasks(image_tasks, clients, extra_args=(global_labels, config))
    print(f">>> 共计 {len(all_tasks)} 个图片请求")

    out_csv = os.path.join(exp_dir, f"{config['exp_name']}.csv")
    final_results = []

    with ResultWriter(out_csv, CSV_HEADERS) as writer:
        with concurrent.futures.ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as ex:
            for row in tqdm(ex.map(process_single_image, all_tasks), total=len(all_tasks), desc="推理中"):
                writer.write_row(row)
                final_results.append(row)

    # 指标计算
    predictions = [normalize_label(r[2]) for r in final_results]
    ground_truths = [normalize_label(r[3]) for r in final_results]
    latencies = [r[-1] for r in final_results if isinstance(r[-1], (int, float)) and r[-1] > 0]
    metrics_result = calculate_metrics(predictions, ground_truths, latencies)
    print_metrics_report(metrics_result)

    # 汇总
    summary_path = os.path.join(exp_dir, "all_experiments_summary.csv")
    summary = metrics_result.to_dict()
    summary.update(
        {
            "exp_name": config["exp_name"],
            "segmentor": "yolov8l-seg",
            "folders": len(DATA_FOLDERS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    append_summary(summary_path, summary)
    update_leaderboard(TEST_OUTPUT_ROOT)

    print(f"\n>>> 实验结束！详细结果: {out_csv}")


# ================= 命令行入口 =================


def parse_args():
    parser = argparse.ArgumentParser(description="VLM + CV 联合测试脚本")
    parser.add_argument("--config", "-c", type=str, default=None, help="实验配置文件路径")
    parser.add_argument("--list-configs", action="store_true", help="列出可用配置")
    return parser.parse_args()


def apply_config(config_path: str):
    """从 YAML 加载配置覆盖全局变量"""
    exp_config = load_config(config_path)
    CONFIG.update(
        {
            "exp_name": exp_config.exp_name,
            "model": exp_config.model,
            "max_size": tuple(exp_config.max_size),
            "quality": exp_config.quality,
            "prompt_id": exp_config.prompt_id,
        }
    )
    DATA_FOLDERS[:] = exp_config.data_folders
    SEGMENTOR_CONFIG.update(
        {
            "weights": exp_config.segmentor_weights,
            "device": exp_config.segmentor_device,
            "conf_threshold": exp_config.conf_threshold,
        }
    )
    # 重新配置分割器
    segmentor.conf_threshold = SEGMENTOR_CONFIG["conf_threshold"]
    save_config(exp_config, os.path.join(TEST_OUTPUT_ROOT, "last_config.yaml"))
    print(f">>> 配置已加载: {exp_config.exp_name}")


if __name__ == "__main__":
    args = parse_args()
    if args.list_configs:
        config_dir = os.path.join(_PROJECT_ROOT, "assets", "configs")
        if os.path.exists(config_dir):
            configs = [f for f in os.listdir(config_dir) if f.endswith(".yaml")]
            print("可用配置: " + ", ".join(configs))
    else:
        if args.config:
            apply_config(args.config)
        run_experiment()
