"""
VLM + CV 联合测试脚本 (修复版)

功能：
1. 使用 YOLOv8-Seg 进行实例分割
2. 【修改点】输入给 VLM 的是“线框轮廓图”而非掩膜图
3. 【修改点】Python 端预计算 IoU/重叠率，一并写入 Prompt 的 JSON 中
4. 保持原始 Prompt 逻辑与 CSV 输出格式不变

作者: Auto-generated & Fixed
日期: 2026-01-20
"""

import os
import sys
import base64
import csv
import re
import io
import json
import time
import argparse
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2  # 需要引入 opencv 画轮廓

# 添加脚本目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 添加项目根目录到路径以导入 config 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

# 导入 YOLOv8-Seg 推理模块
from experiment_config import load_config, save_config, ExperimentConfig
from yolov8_seg_inference import YOLOv8SegInference, load_yolov8_seg
from prompt_manager import load_prompt, list_prompts


# ================= 配置区域 =================

# 1. 数据文件夹
DATA_FOLDERS = [
    r"/root/XiaoanNew/Compliance_test_data/no_val",
    r"/root/XiaoanNew/Compliance_test_data/yes_val",
]

# 2. 输出根目录
TEST_OUTPUT_ROOT = "/root/XiaoanNew/test_outputs"

# 3. 生成带时间戳的实验目录
from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# 4. 创建本次实验的独立目录（将在配置读取后创建）
def create_experiment_dir(exp_name):
    """创建本次实验的输出目录，包含 CSV 和可视化子目录"""
    exp_dir = os.path.join(TEST_OUTPUT_ROOT, f"exp_{TIMESTAMP}_{exp_name}")
    vis_dir = os.path.join(exp_dir, "visuals")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    return exp_dir, vis_dir

# 占位变量，将在 main() 中初始化
SAVE_DIR = None
SEG_VIS_DIR = None

# 4. 实验配置
CONFIG = {
    "exp_name": "qwen3-vl-30b-a3b_contours_iou_fix", # 修改一下实验名以示区别
    "model": "qwen/qwen3-vl-30b-a3b-instruct",
    "max_size": (768, 768),
    "quality": 80,
    "prompt_id": "cv_enhanced_p4" # 保持原始 Prompt
}

# 5. YOLOv8-Seg 配置
SEGMENTOR_CONFIG = {
    "weights": "/root/XiaoanNew/weights/best.pt",
    "device": "cuda:0",
    "conf_threshold": 0.6
}

# 6. 轮廓颜色配置 (BGR格式)
CONTOUR_COLORS = {
    "Electric bike": (0, 255, 0),      # 绿
    "Curb": (255, 0, 255),             # 紫
    "parking lane": (0, 255, 255),     # 黄
    "Tactile paving": (0, 128, 255),   # 橙
    "default": (255, 255, 255)
}

# 7. 提示词管理 (从外部 YAML 文件加载)
# 可用提示词: scripts/prompts/*.yaml
# 使用方式: 修改 CONFIG['prompt_id'] 指定提示词名称

def get_prompt(prompt_id: str) -> str:
    """加载指定的提示词内容"""
    try:
        return load_prompt(prompt_id)
    except FileNotFoundError as e:
        print(f"⚠️ 提示词加载失败: {e}")
        print(f"可用的提示词: {list_prompts()}")
        raise

# ================= API 配置 =================

# API 配置从环境变量加载
BASE_URL = settings.API_BASE_URL
API_KEYS = settings.VLM_API_KEYS
MAX_WORKERS = settings.MAX_WORKERS


# ================= 新增：几何计算工具函数 =================

def calculate_iou_and_overlap(mask1, mask2):
    """计算两个 Mask 的 IoU 和包含率"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    area1 = mask1.sum()
    
    iou = intersection / union if union > 0 else 0
    overlap_ratio = intersection / area1 if area1 > 0 else 0 # mask1 被 mask2 覆盖的比例
    
    return round(iou, 4), round(overlap_ratio, 4)

def combine_masks(objects, label_filter):
    """将特定类别的所有 mask 合并为一个"""
    combined = None
    for obj in objects:
        if obj["label"] == label_filter and obj.get("mask") is not None:
            if combined is None:
                combined = obj["mask"].copy()
            else:
                combined = np.logical_or(combined, obj["mask"])
    return combined

def draw_wireframe_visual(image_raw, objects):
    """绘制线框轮廓图 (Wireframe) 替代掩膜填充"""
    vis_img = image_raw.copy()
    # 转 BGR 以便 cv2 处理
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    for obj in objects:
        mask = obj.get("mask")
        label = obj["label"]
        if mask is None: continue
        
        # 获取颜色
        color = CONTOUR_COLORS.get(label, CONTOUR_COLORS["default"])
        
        # 提取轮廓
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制轮廓 (线宽 2)
        cv2.drawContours(vis_img, contours, -1, color, 2)
        
        # 可选：绘制 BBox 辅助识别
        # x1, y1, x2, y2 = map(int, obj["bbox"])
        # cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 1)

    # 转回 RGB
    return cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)


# ================= 工具函数 (保持原样) =================

def norm_yesno(x: str) -> str:
    """标准化是/否标签"""
    if not x: return ""
    s = str(x).strip().lower()
    if any(k in s for k in ["yes", "true", "1", "合规"]): return "yes"
    if any(k in s for k in ["no", "false", "0", "不合规"]): return "no"
    return ""


def parse_vlm_response(response_text):
    """解析 VLM 返回的 JSON 响应"""
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return "error", "JSON not found", "fail", "fail", "fail", "fail"

        data = json.loads(json_match.group())
        scores = data.get("scores", {})

        comp = scores.get("composition_status", "")
        ang = scores.get("angle_status", "")
        dist = scores.get("distance_status", "")
        cont = scores.get("context_status", "")

        # 判断逻辑
        res = "yes"
        if "不合规" in comp or "不合规" in ang or "不合规" in cont or "超界" in dist:
            res = "no"

        reason = json.dumps(data.get("step_by_step_analysis", {}), ensure_ascii=False)
        return res, reason, comp, ang, dist, cont

    except Exception as e:
        return "error", str(e), "fail", "fail", "fail", "fail"


def encode_image(img_array, max_size=(768, 768), quality=80):
    """将图像数组编码为 Base64"""
    pil_img = Image.fromarray(img_array)
    pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ================= 全局模型加载 =================

print("=" * 60)
print("🚀 正在加载 YOLOv8-Seg 模型...")
print("=" * 60)

segmentor = load_yolov8_seg(
    weights_path=SEGMENTOR_CONFIG["weights"],
    device=SEGMENTOR_CONFIG["device"]
)
segmentor.conf_threshold = SEGMENTOR_CONFIG["conf_threshold"]

print("✅ YOLOv8-Seg 模型加载完成")
print("=" * 60)


# ================= 核心处理函数 =================

def process_single_image(args):
    """处理单张图片：YOLOv8-Seg + VLM"""
    image_name, folder_path, client, labels_dict, config = args
    image_path = os.path.join(folder_path, image_name)
    gt = labels_dict.get((image_name, folder_path), "N/A")

    start_t = time.time()

    try:
        # ===== 1. YOLOv8-Seg 实例分割 =====
        seg_result = segmentor.predict(image_path)
        
        raw_img = seg_result["image_raw"]       # 原图 (H, W, 3) RGB
        objects = seg_result["objects"]         # 检测对象列表
        H, W = seg_result["image_size"]

        # 【修改点 1】生成线框轮廓图 (Contours) 替代默认的 Image Visual
        vis_img = draw_wireframe_visual(raw_img, objects)

        # 保存可视化结果（调试）
        vis_path = os.path.join(SEG_VIS_DIR, image_name)
        Image.fromarray(vis_img).save(vis_path)

        # ===== 2. 编码两张图为 Base64 =====
        b64_raw = encode_image(raw_img, config['max_size'], config['quality'])
        b64_vis = encode_image(vis_img, config['max_size'], config['quality'])

        # ===== 3. 【修改点 2】构造增强的结构化信息（加入数学计算） =====
        detection_info = {
            "image_size": [H, W],
            "detected_objects": [],
            "geometry_analysis": {}  # 新增分析字段
        }
        
        class_counts = {"Electric bike": 0, "Curb": 0, "parking lane": 0, "Tactile paving": 0}
        
        # 3.1 基础信息填充
        main_bike_mask = None
        main_bike_conf = -1

        for obj in objects:
            detection_info["detected_objects"].append({
                "id": obj["id"],
                "label": obj["label"],
                "confidence": obj["confidence"],
                "bbox": obj["bbox"],
                # "area_ratio": obj["area_ratio"] 
            })
            if obj["label"] in class_counts:
                class_counts[obj["label"]] += 1
            
            # 寻找置信度最高的电动车作为主体
            if obj["label"] == "Electric bike" and obj["confidence"] > main_bike_conf:
                main_bike_conf = obj["confidence"]
                main_bike_mask = obj.get("mask")

        detection_info["class_summary"] = class_counts

        # 3.2 几何关系预计算 (Python计算IoU等)
        geo_analysis = {
            "main_vehicle_detected": False,
            "overlap_with_parking_lane": 0.0,
            "iou_with_parking_lane": 0.0,
            "overlap_with_tactile_paving": 0.0,
            "status_inference": "unknown"
        }

        if main_bike_mask is not None:
            geo_analysis["main_vehicle_detected"] = True
            
            # 合并停车区 Mask
            parking_mask = combine_masks(objects, "parking lane")
            if parking_mask is not None:
                iou, overlap = calculate_iou_and_overlap(main_bike_mask, parking_mask)
                geo_analysis["iou_with_parking_lane"] = iou
                geo_analysis["overlap_with_parking_lane"] = overlap # 车辆有多少比例在停车框内
            
            # 合并盲道 Mask
            tactile_mask = combine_masks(objects, "Tactile paving")
            if tactile_mask is not None:
                _, overlap_tactile = calculate_iou_and_overlap(main_bike_mask, tactile_mask)
                geo_analysis["overlap_with_tactile_paving"] = overlap_tactile

            # 简单的 Python 推断辅助
            if geo_analysis["overlap_with_parking_lane"] > 0.8:
                geo_analysis["status_inference"] = "Likely Compliant (High Overlap)"
            elif geo_analysis["overlap_with_parking_lane"] < 0.1:
                geo_analysis["status_inference"] = "Likely Out of Bounds"

        detection_info["geometry_analysis"] = geo_analysis
        
        # 序列化为 JSON
        structured_info = json.dumps(detection_info, ensure_ascii=False, indent=2)

        # ===== 4. 构造 Prompt (保持原始格式，注入新的 structured_info) =====
        full_prompt = get_prompt(config['prompt_id']) + \
            "\n\n# YOLOv8-Seg Detection & Geometry Analysis\n```json\n" + \
            structured_info + "\n```"

        # ===== 5. 调用 VLM（发送原图 + 线框轮廓图） =====
        res = client.chat.completions.create(
            model=config['model'],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_raw}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_vis}"}}
                ]
            }],
            max_tokens=1000,
            temperature=0.1,
            top_p=0.9,
        )

        vlm_out = res.choices[0].message.content
        pred, reason, comp, ang, dist, cont = parse_vlm_response(vlm_out)

        return [
            image_name,
            os.path.basename(folder_path),
            pred, gt,
            comp, ang, dist, cont,
            len(objects),  # 检测对象数
            class_counts.get("Electric bike", 0),
            class_counts.get("Curb", 0),
            class_counts.get("parking lane", 0),
            class_counts.get("Tactile paving", 0),
            reason,
            round(time.time() - start_t, 3)
        ]

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [image_name, os.path.basename(folder_path), "error", gt,
                "err", "err", "err", "err", 0, 0, 0, 0, 0, str(e), 0]


# ================= 评估函数 (保持原样) =================

def calculate_and_report(results):
    """计算并输出评估报告"""
    tp, tn, fp, fn, inv = 0, 0, 0, 0, 0
    lats = []
    
    for r in results:
        pred, gt = norm_yesno(r[2]), norm_yesno(r[3])
        if r[2] == "error": 
            inv += 1
            continue
        if gt == 'yes':
            if pred == 'yes': tp += 1
            else: fn += 1
        elif gt == 'no':
            if pred == 'no': tn += 1
            else: fp += 1
        if r[-1] > 0: 
            lats.append(r[-1])
    
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    avg_lat = round(sum(lats)/len(lats), 3) if lats else 0

    print(f"\n{'='*20} 评估报告 (YOLOv8-Seg + VLM) {'='*20}")
    print(f"总样本数 (Total Samples): {total}")
    print(f"无效/错误预测 (Invalid): {inv}")
    print("-" * 60)
    print(f"准确率 (Accuracy) : {acc:.2%}")
    print(f"精确率 (Precision): {pre:.2%}")
    print(f"召回率 (Recall)   : {rec:.2%}")
    print(f"F1分数 (F1-Score) : {f1:.2f}")
    print("-" * 60)
    print("混淆矩阵详情:")
    print(f"  [TP] 预测正确(合规): {tp}")
    print(f"  [TN] 预测正确(违规): {tn}")
    print(f"  [FP] 误判为合规 (实际违规): {fp}")
    print(f"  [FN] 误判为违规 (实际合规): {fn}")
    print(f"平均单样本耗时: {avg_lat}s")
    print("=" * 60)
    
    return {
        "acc": acc, "f1": f1, "pre": pre, "rec": rec,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": total, "invalid": inv, "avg_lat": avg_lat
    }


# ================= 主函数 (保持原样) =================

def main():
    # 初始化实验输出目录
    global SAVE_DIR, SEG_VIS_DIR
    SAVE_DIR, SEG_VIS_DIR = create_experiment_dir(CONFIG['exp_name'])
    print(f"\n>>> 实验目录: {SAVE_DIR}")
    
    print(f"\n>>> 实验启动（YOLOv8-Seg + VLM + Contours + IoU Calc）")
    print(f">>> 模型: {CONFIG['model']}")
    print(f">>> 分割模式: 轮廓图 (Contours)")

    # 初始化 API 客户端池
    clients = [OpenAI(base_url=BASE_URL, api_key=k) for k in API_KEYS]
    
    # 加载标签
    global_labels = {}
    all_tasks = []

    for folder in DATA_FOLDERS:
        if not os.path.exists(folder):
            print(f"⚠️ 文件夹不存在: {folder}")
            continue
            
        l_path = os.path.join(folder, "labels.txt")
        if os.path.exists(l_path):
            with open(l_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",", 1)
                    if len(parts) == 2:
                        name, lab = parts[0].strip(), parts[1].strip()
                        global_labels[(name, folder)] = norm_yesno(lab)

        imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for i, img in enumerate(imgs):
            all_tasks.append((img, folder, clients[i % len(clients)], global_labels, CONFIG))

    print(f">>> 任务分发完毕，共计 {len(all_tasks)} 个图片请求。")

    # 输出 CSV
    out_csv = os.path.join(SAVE_DIR, f"{CONFIG['exp_name']}.csv")
    results = []

    with open(out_csv, "w", newline='', encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "folder", "pred", "gt",
            "composition", "angle", "distance", "context",
            "num_detections", "electric_bike", "curb", "parking_lane", "tactile_paving",
            "reason", "latency"
        ])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for row in tqdm(ex.map(process_single_image, all_tasks), 
                          total=len(all_tasks), desc="推理中"):
                writer.writerow(row)
                f.flush()
                results.append(row)

    # 计算评估指标
    metrics = calculate_and_report(results)
    
    # 保存汇总
    summary_path = os.path.join(SAVE_DIR, "all_experiments_summary.csv")
    metrics.update({
        "exp_name": CONFIG['exp_name'],
        "segmentor": "yolov8l-seg",
        "folders": len(DATA_FOLDERS),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    file_exists = os.path.exists(summary_path)
    with open(summary_path, 'a', newline='', encoding='utf-8-sig') as f:
        dict_writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            dict_writer.writeheader()
        dict_writer.writerow(metrics)

    print(f"\n>>> 实验结束！详细结果已保存: {out_csv}")




def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VLM + CV 联合测试脚本')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='实验配置文件路径 (YAML格式)')
    parser.add_argument('--list-configs', action='store_true',
                       help='列出可用的配置文件')
    return parser.parse_args()


def run_with_config(config_path: str = None):
    """使用配置文件运行实验"""
    global CONFIG, DATA_FOLDERS, SEGMENTOR_CONFIG, MAX_WORKERS, SAVE_DIR, SEG_VIS_DIR
    
    if config_path:
        print(f">>> 加载配置文件: {config_path}")
        exp_config = load_config(config_path)
        
        # 更新全局配置
        CONFIG = {
            "exp_name": exp_config.exp_name,
            "model": exp_config.model,
            "max_size": tuple(exp_config.max_size),
            "quality": exp_config.quality,
            "prompt_id": exp_config.prompt_id
        }
        DATA_FOLDERS = exp_config.data_folders
        SEGMENTOR_CONFIG = {
            "weights": exp_config.segmentor_weights,
            "device": exp_config.segmentor_device,
            "conf_threshold": exp_config.conf_threshold
        }
        MAX_WORKERS = exp_config.max_workers
        
        print(f">>> 配置已加载:")
        print(f"    - 实验名称: {exp_config.exp_name}")
        print(f"    - 模型: {exp_config.model}")
        print(f"    - 提示词: {exp_config.prompt_id}")
        print(f"    - 数据目录: {len(DATA_FOLDERS)} 个")
    
    # 创建实验目录
    SAVE_DIR, SEG_VIS_DIR = create_experiment_dir(CONFIG['exp_name'])
    print(f"\n>>> 实验目录: {SAVE_DIR}")
    
    # 备份配置文件到实验目录
    if config_path:
        backup_path = os.path.join(SAVE_DIR, "experiment_config.yaml")
        save_config(exp_config, backup_path)
        print(f">>> 配置已备份: {backup_path}")
    
    # 调用原始 main 函数的核心逻辑
    _run_experiment()


def _run_experiment():
    """执行实验核心逻辑"""
    global SAVE_DIR, SEG_VIS_DIR
    
    print(f"\n>>> 实验启动（YOLOv8-Seg + VLM + Contours + IoU Calc）")
    print(f">>> 模型: {CONFIG['model']}")
    print(f">>> 分割模式: 轮廓图 (Contours)")

    # 初始化 API 客户端池
    clients = [OpenAI(base_url=BASE_URL, api_key=k) for k in API_KEYS]
    
    # 加载标签
    global_labels = {}
    all_tasks = []

    for folder in DATA_FOLDERS:
        if not os.path.exists(folder):
            print(f"⚠️ 文件夹不存在: {folder}")
            continue
            
        l_path = os.path.join(folder, "labels.txt")
        if os.path.exists(l_path):
            with open(l_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",", 1)
                    if len(parts) == 2:
                        name, lab = parts[0].strip(), parts[1].strip()
                        global_labels[(name, folder)] = norm_yesno(lab)

        imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for i, img in enumerate(imgs):
            all_tasks.append((img, folder, clients[i % len(clients)], global_labels, CONFIG))

    print(f">>> 任务分发完毕，共计 {len(all_tasks)} 个图片请求。")

    # 输出 CSV
    out_csv = os.path.join(SAVE_DIR, f"{CONFIG['exp_name']}.csv")
    results = []

    with open(out_csv, "w", newline='', encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "folder", "pred", "gt",
            "composition", "angle", "distance", "context",
            "num_detections", "electric_bike", "curb", "parking_lane", "tactile_paving",
            "reason", "latency"
        ])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for row in tqdm(ex.map(process_single_image, all_tasks), 
                          total=len(all_tasks), desc="推理中"):
                writer.writerow(row)
                f.flush()
                results.append(row)

    # 计算评估指标
    metrics = calculate_and_report(results)
    
    # 保存汇总
    summary_path = os.path.join(SAVE_DIR, "all_experiments_summary.csv")
    metrics.update({
        "exp_name": CONFIG['exp_name'],
        "segmentor": "yolov8l-seg",
        "folders": len(DATA_FOLDERS),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    file_exists = os.path.exists(summary_path)
    with open(summary_path, 'a', newline='', encoding='utf-8-sig') as f:
        dict_writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            dict_writer.writeheader()
        dict_writer.writerow(metrics)

    print(f"\n>>> 实验结束！详细结果已保存: {out_csv}")


if __name__ == "__main__":
    args = parse_args()
    
    if args.list_configs:
        # 列出可用配置
        config_dir = os.path.join(os.path.dirname(__file__), "configs")
        if os.path.exists(config_dir):
            configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
            print("可用的实验配置文件:")
            for c in configs:
                print(f"  - {c}")
        else:
            print("配置目录不存在")
    elif args.config:
        # 使用配置文件运行
        run_with_config(args.config)
    else:
        # 保持向后兼容：无参数时使用硬编码配置
        main()
