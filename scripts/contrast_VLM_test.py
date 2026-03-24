"""v1 纯 VLM 停车合规检测实验脚本

使用 VLM 模型对停车图片进行四维度合规判定。
支持加权评分和一票否决两种评判模式。
"""

import os
import sys
import time
import concurrent.futures

from tqdm import tqdm

# 项目根目录加入路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from config.settings import get_settings
from scripts.prompt_manager import load_prompt
from utils.vlm_client import create_client_pool, distribute_tasks
from utils.vlm_parser import parse_vlm_response, normalize_label
from utils.image_utils import encode_image_to_base64
from utils.experiment_io import (
    load_all_labels,
    collect_image_tasks,
    ResultWriter,
    append_summary,
)
from utils.metrics import calculate_metrics, print_metrics_report
from utils.scoring import ScoringEngine

# ================= 实验配置 =================

CONFIG = {
    "exp_name": "v1_standard_p6_weighted",
    "model": "qwen/qwen3-vl-30b-a3b-instruct",
    "max_size": (768, 768),
    "quality": 80,
    "prompt_id": "standard_p6",
    "scoring_config": "configs/scoring_default.yaml",  # None 使用一票否决
}

DATA_FOLDERS = [
    os.path.join(_PROJECT_ROOT, "Compliance_test_data/no_val"),
    os.path.join(_PROJECT_ROOT, "Compliance_test_data/yes_val"),
]

SAVE_DIR = os.path.join(_PROJECT_ROOT, "experiment_outputs")
os.makedirs(SAVE_DIR, exist_ok=True)

# ================= 初始化 =================

_scoring_engine = None
if CONFIG.get("scoring_config"):
    cfg_path = os.path.join(_PROJECT_ROOT, CONFIG["scoring_config"])
    _scoring_engine = ScoringEngine.from_yaml(cfg_path)
    print(f">>> 加权评判引擎已加载: {CONFIG['scoring_config']} (阈值={_scoring_engine.config.threshold})")
else:
    print(">>> 使用一票否决评判模式")


# ================= 核心推理 =================

def _judge(vlm_result):
    """根据 VLM 解析结果进行合规判定"""
    comp, ang, dist, cont = vlm_result.statuses
    if _scoring_engine:
        sr = _scoring_engine.score(comp, ang, dist, cont)
        return ("yes" if sr.is_compliant else "no"), sr.final_score
    else:
        return ScoringEngine.veto_judge(comp, ang, dist, cont), 0.0


def process_single_image(args):
    """处理单张图片：编码 -> VLM 推理 -> 解析 -> 判定"""
    image_name, folder_path, client, labels_dict, config = args
    image_path = os.path.join(folder_path, image_name)
    gt = labels_dict.get((image_name, folder_path), "N/A")

    start_t = time.time()
    try:
        b64_img = encode_image_to_base64(
            image_path,
            max_size=config["max_size"],
            quality=config["quality"],
        )

        prompt_text = load_prompt(config["prompt_id"])

        res = client.chat.completions.create(
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
            max_tokens=600,
            temperature=0.1,
        )
        vlm_out = res.choices[0].message.content
        vlm_result = parse_vlm_response(vlm_out)

        if not vlm_result.is_valid:
            return [image_name, os.path.basename(folder_path), "error", gt,
                    "fail", "fail", "fail", "fail",
                    vlm_result.parse_error, 0, 0.0]

        pred, w_score = _judge(vlm_result)
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
        return [image_name, os.path.basename(folder_path), "error", gt,
                "err", "err", "err", "err", str(e), 0, 0.0]


# ================= 主程序 =================

CSV_HEADERS = [
    "image_name", "folder", "result", "ground_truth",
    "composition", "angle", "distance", "context",
    "reason", "latency_sec", "weighted_score",
]


def main():
    settings = get_settings()
    print(f">>> 实验启动！模型: {CONFIG['model']}")

    clients = create_client_pool()
    print(f">>> 初始化 {len(clients)} 个 API 客户端")

    global_labels = load_all_labels(DATA_FOLDERS)
    image_tasks = collect_image_tasks(DATA_FOLDERS)
    all_tasks = distribute_tasks(image_tasks, clients, extra_args=(global_labels, CONFIG))
    print(f">>> 任务分发完毕，共计 {len(all_tasks)} 个图片请求。")

    out_csv = os.path.join(SAVE_DIR, f"results_{CONFIG['exp_name'].replace('/', '_')}_detailed.csv")
    final_results = []

    with ResultWriter(out_csv, CSV_HEADERS) as writer:
        with concurrent.futures.ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as executor:
            for row in tqdm(
                executor.map(process_single_image, all_tasks),
                total=len(all_tasks),
                desc="VLM 推理",
            ):
                writer.write_row(row)
                final_results.append(row)

    # 指标计算
    predictions = [normalize_label(r[2]) for r in final_results]
    ground_truths = [normalize_label(r[3]) for r in final_results]
    latencies = [r[9] for r in final_results if isinstance(r[9], (int, float)) and r[9] > 0]
    metrics_result = calculate_metrics(predictions, ground_truths, latencies)
    print_metrics_report(metrics_result)

    # 汇总
    summary_path = os.path.join(SAVE_DIR, "all_experiments_summary.csv")
    summary = metrics_result.to_dict()
    summary.update({
        "exp_name": CONFIG["exp_name"],
        "folders": len(DATA_FOLDERS),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    append_summary(summary_path, summary)

    print(f"\n>>> 实验结束！详细结果: {out_csv}")


if __name__ == "__main__":
    main()
