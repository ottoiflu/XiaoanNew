"""
VLM + CV 联合测试脚本

功能：
1. 使用 YOLOv8-Seg 进行实例分割，检测电动车、停车线、马路牙子、盲道
2. 将原图 + 分割可视化图 + 结构化检测信息 发送给 VLM
3. VLM 基于增强信息进行停车合规性判断

作者: Auto-generated
日期: 2026-01-20
"""

import base64
import concurrent.futures
import csv
import io
import json
import os
import re
import sys
import time

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# 添加脚本目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 YOLOv8-Seg 推理模块
from modules.cv.yolov8_inference import load_yolov8_seg

# ================= 配置区域 =================

# 1. 数据文件夹
DATA_FOLDERS = [
    r"/root/XiaoanNew/Compliance_test_data/no_val",
    r"/root/XiaoanNew/Compliance_test_data/yes_val",
]

# 2. 输出目录
# 2. 输出根目录
TEST_OUTPUT_ROOT = "/root/XiaoanNew/test_outputs"

# 3. 生成带时间戳的实验目录
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# 4. 创建本次实验的独立目录
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
    "exp_name": "qwen3-vl-30b-a3b-instruct_yolov8seg_cv_enhanced_p3_test_copy",
    "model": "qwen/qwen3-vl-30b-a3b-instruct",
    "max_size": (768, 768),
    "quality": 80,
    "prompt_id": "cv_enhanced_p3_copy",
}

# 5. YOLOv8-Seg 配置
SEGMENTOR_CONFIG = {"weights": "/root/XiaoanNew/weights/best.pt", "device": "cuda:0", "conf_threshold": 0.6}

# 6. 提示词库（CV增强版本）
PROMPT_LIB = {
    "cv_enhanced_p3_copy": """
# Role
你是一位专业的共享单车运维质检员，负责通过照片判定车辆停放是否符合城市管理严苛标准。

# Input Description
1. **原始图片**：原生视觉参考。
2. **分割可视化图**：辅助参考（绿色：电动车、黄色：停车线、紫色：马路牙子、橙色：盲道）。
3. **结构化检测数据**：JSON 格式的目标位置参考。

# General Principles (核心原则)
- **视觉验证优先**：AI 分割掩码仅作为“区域搜索建议”。**若掩码边缘与原图中肉眼可见的物理边界（如路沿、线框、地砖颜色分界点）不一致，必须以原图特征为准。**
- **自主推断边界**：若 AI 未检测到停车线，请根据原图中的马路牙子、绿化带边缘或地砖缝隙判断隐含边界。

# Task
锁定画面中【最显著】的一辆共享单车/电动车为主体，执行以下判定流程。

# Guidelines & Criteria

## 1. 图像构图合规性 (Image Quality & Composition)
- **状态选项**：
  - `[合规]`：画面清晰，车辆大部分在框内，有明确参照物。
  - `[基本合规]`：关键部位虽部分截断，但仍可判定空间关系。
  - `[不合规-构图]`：过暗/过曝、拍摄距离过远。
  - `[不合规-无参照]`：AI 未检出且原图也无法识别任何物理边界线。

## 2. 摆放角度合规性 (Angle Compliance)
- **基准线**：停车线长边或马路牙子边缘。
- **向量**：后轮中点 -> 把手中心。
- **准则**：
  - 必须与基准线保持【垂直】，偏差不得超过 ±30°（即夹角 > 60°）。
  - 若车辆贴边顺向摆放（平行），直接判定为不合规。
- **状态选项**：`[合规]`、`[不合规-角度]`。

## 3. 停放距离合规性 (Distance Compliance)
- **判定基准**：以车座中心点与后轮中点的【连线中点】为界限。
- **状态选项**：
  - `[完全合规]`：车辆主体完全在线内。
  - `[基本合规-压线]`：界限中点处于停车线/边界**内侧**，仅后轮局部压线或向**后轮方向**超出。
  - `[不合规-超界]`：界限中点已跨越边界线向**车头/车座方向**偏移（即车身 1/2 以上出线）。

## 4. 路面环境合规性 (Contextual Compliance)
- **核心判定：盲道占用**。
- **逻辑优先级**：
  - **P1 (看接地点)**：若可见轮迹，任一车轮接地点完全压在盲道纹路上 → `[不合规-环境]`。
  - **P2 (看投影)**：若接地点被遮挡，车座投影面积 > 50% 覆盖盲道纹路 → `[不合规-环境]`。
- **状态选项**：`[合规]`、`[不合规-环境]`。

# 极严输出约束 (Mandatory Constraints)
1. **标签匹配**：status 字段必须严格从给定选项中精确选择，严禁自创。
2. **纯净输出**：status 字段内严禁出现任何解释性文字或括号内容。
3. **逻辑一致**：`step_by_step_analysis` 的描述必须支持最后的 `scores` 结论。

# Output Format (JSON)
必须严格按以下格式回答：
```json
{
  "step_by_step_analysis": {
    "ai_detection_summary": "总结 AI 检测到的对象（电动车、停车线、马路牙子、盲道的数量）",
    "composition_check": "描述主体可见性、识别到的最强参照物具体名称及画质评估",
    "angle_analysis": "识别车辆向量长轴，明确说明是相对于哪类参考线，估算偏离夹角",
    "distance_analysis": "描述轮迹点与边界位置，根据【车座-后轮连线中点】说明车身超出比例",
    "context_analysis": "判断是否有盲道检测，以及车辆接地点或车座是否压在盲道上"
  },
  "scores": {
    "composition_status": "[合规] / [基本合规] / [不合规-构图] / [不合规-无参照]",
    "angle_status": "[合规] / [不合规-角度]",
    "distance_status": "[完全合规] / [基本合规-压线] / [不合规-超界]",
    "context_status": "[合规] / [不合规-环境]"
  }
}
```


# Role
你是一位专业的共享单车运维质检员，负责通过照片判定车辆停放是否符合城市管理严苛标准。

# Input Description
1. **原始图片**：原生视觉参考。
2. **分割可视化图**：辅助参考（绿色：电动车、黄色：停车线、紫色：马路牙子、橙色：盲道）。
3. **结构化检测数据**：JSON 格式的目标位置参考。

# General Principles (核心原则)
- **视觉验证优先**：AI 分割掩码仅作为“区域搜索建议”。**若掩码边缘与原图中肉眼可见的物理边界（如路沿、线框、地砖颜色分界点）不一致，必须以原图特征为准。**
- **自主推断边界**：若 AI 未检测到停车线，请根据原图中的马路牙子、绿化带边缘或地砖缝隙判断隐含边界。

# Task
锁定画面中【最显著】的一辆共享单车/电动车为主体，执行以下判定流程。

# Guidelines & Criteria

## 1. 图像构图合规性 (Image Quality & Composition)
- **状态选项**：
  - `[合规]`：画面清晰，车辆大部分在框内，有明确参照物。
  - `[基本合规]`：关键部位虽部分截断，但仍可判定空间关系。
  - `[不合规-构图]`：过暗/过曝、拍摄距离过远。
  - `[不合规-无参照]`：AI 未检出且原图也无法识别任何物理边界线。

## 2. 摆放角度合规性 (Angle Compliance)
- **基准线**：停车线长边或马路牙子边缘。
- **向量**：后轮中点 -> 把手中心。
- **准则**：
  - 必须与基准线保持【垂直】，偏差不得超过 ±30°（即夹角 > 60°）。
  - 若车辆贴边顺向摆放（平行），直接判定为不合规。
- **状态选项**：`[合规]`、`[不合规-角度]`。

## 3. 停放距离合规性 (Distance Compliance)
- **判定基准**：以车座中心点与后轮中点的【连线中点】为界限。
- **状态选项**：
  - `[完全合规]`：车辆主体完全在线内。
  - `[基本合规-压线]`：界限中点处于停车线/边界**内侧**，仅后轮局部压线或向**后轮方向**超出。
  - `[不合规-超界]`：界限中点已跨越边界线向**车头/车座方向**偏移（即车身 1/2 以上出线）。

## 4. 路面环境合规性 (Contextual Compliance)
- **核心判定：盲道占用**。
- **逻辑优先级**：
  - **P1 (看接地点)**：若可见轮迹，任一车轮接地点完全压在盲道纹路上 → `[不合规-环境]`。
  - **P2 (看投影)**：若接地点被遮挡，车座投影面积 > 50% 覆盖盲道纹路 → `[不合规-环境]`。
- **状态选项**：`[合规]`、`[不合规-环境]`。

# 极严输出约束 (Mandatory Constraints)
1. **标签匹配**：status 字段必须严格从给定选项中精确选择，严禁自创。
2. **纯净输出**：status 字段内严禁出现任何解释性文字或括号内容。
3. **逻辑一致**：`step_by_step_analysis` 的描述必须支持最后的 `scores` 结论。

# Output Format (JSON)
必须严格按以下格式回答：
```json
{
  "step_by_step_analysis": {
    "ai_detection_summary": "总结 AI 检测到的对象（电动车、停车线、马路牙子、盲道的数量）",
    "composition_check": "描述主体可见性、识别到的最强参照物具体名称及画质评估",
    "angle_analysis": "识别车辆向量长轴，明确说明是相对于哪类参考线，估算偏离夹角",
    "distance_analysis": "描述轮迹点与边界位置，根据【车座-后轮连线中点】说明车身超出比例",
    "context_analysis": "判断是否有盲道检测，以及车辆接地点或车座是否压在盲道上"
  },
  "scores": {
    "composition_status": "[合规] / [基本合规] / [不合规-构图] / [不合规-无参照]",
    "angle_status": "[合规] / [不合规-角度]",
    "distance_status": "[完全合规] / [基本合规-压线] / [不合规-超界]",
    "context_status": "[合规] / [不合规-环境]"
  }
}
```
"""
}

# ================= API 配置 =================

BASE_URL = "https://api.ppinfra.com/openai"
API_KEYS = [
    "REDACTED_API_KEY_3",
    # "REDACTED_API_KEY_2",
    "REDACTED_API_KEY_1",
    "REDACTED_API_KEY_5",
]
MAX_WORKERS = 15


# ================= 工具函数 =================


def norm_yesno(x: str) -> str:
    """标准化是/否标签"""
    if not x:
        return ""
    s = str(x).strip().lower()
    if any(k in s for k in ["yes", "true", "1", "合规"]):
        return "yes"
    if any(k in s for k in ["no", "false", "0", "不合规"]):
        return "no"
    return ""


def parse_vlm_response(response_text):
    """解析 VLM 返回的 JSON 响应"""
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
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
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


# ================= 全局模型加载 =================

print("=" * 60)
print("正在加载 YOLOv8-Seg 模型...")
print("=" * 60)

segmentor = load_yolov8_seg(weights_path=SEGMENTOR_CONFIG["weights"], device=SEGMENTOR_CONFIG["device"])
segmentor.conf_threshold = SEGMENTOR_CONFIG["conf_threshold"]

print("YOLOv8-Seg 模型加载完成")
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

        raw_img = seg_result["image_raw"]  # 原图 (H, W, 3) RGB
        vis_img = seg_result["image_visual"]  # 可视化图 (H, W, 3) RGB
        objects = seg_result["objects"]  # 检测对象列表
        H, W = seg_result["image_size"]

        # 保存可视化结果（调试）
        vis_path = os.path.join(SEG_VIS_DIR, image_name)
        Image.fromarray(vis_img).save(vis_path)

        # ===== 2. 编码两张图为 Base64 =====
        b64_raw = encode_image(raw_img, config["max_size"], config["quality"])
        b64_vis = encode_image(vis_img, config["max_size"], config["quality"])

        # ===== 3. 构造结构化检测信息 =====
        detection_info = {"image_size": [H, W], "detected_objects": []}

        # 统计各类别数量
        class_counts = {"Electric bike": 0, "Curb": 0, "parking lane": 0, "Tactile paving": 0}

        for obj in objects:
            detection_info["detected_objects"].append(
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "confidence": obj["confidence"],
                    "bbox": obj["bbox"],
                    "area_ratio": obj["area_ratio"],
                }
            )
            if obj["label"] in class_counts:
                class_counts[obj["label"]] += 1

        detection_info["class_summary"] = class_counts

        structured_info = json.dumps(detection_info, ensure_ascii=False, indent=2)

        # ===== 4. 构造增强 Prompt =====
        full_prompt = (
            PROMPT_LIB[config["prompt_id"]]
            + "\n\n# YOLOv8-Seg Detection Results (Reference Data)\n```json\n"
            + structured_info
            + "\n```"
        )

        # ===== 5. 调用 VLM（发送原图 + 分割可视化图） =====
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
        pred, reason, comp, ang, dist, cont = parse_vlm_response(vlm_out)

        return [
            image_name,
            os.path.basename(folder_path),
            pred,
            gt,
            comp,
            ang,
            dist,
            cont,
            len(objects),  # 检测对象数
            class_counts.get("Electric bike", 0),
            class_counts.get("Curb", 0),
            class_counts.get("parking lane", 0),
            class_counts.get("Tactile paving", 0),
            reason,
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


# ================= 评估函数 =================


def calculate_and_report(results):
    """计算并输出评估报告"""
    tp, tn, fp, fn, inv = 0, 0, 0, 0, 0
    lats = []

    for r in results:
        pred, gt = norm_yesno(r[2]), norm_yesno(r[3])
        if r[2] == "error":
            inv += 1
            continue
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
        if r[-1] > 0:
            lats.append(r[-1])

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    avg_lat = round(sum(lats) / len(lats), 3) if lats else 0

    print(f"\n{'=' * 20} 评估报告 (YOLOv8-Seg + VLM) {'=' * 20}")
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
        "acc": acc,
        "f1": f1,
        "pre": pre,
        "rec": rec,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "invalid": inv,
        "avg_lat": avg_lat,
    }


# ================= 主函数 =================


def main():
    # 初始化实验输出目录
    global SAVE_DIR, SEG_VIS_DIR
    SAVE_DIR, SEG_VIS_DIR = create_experiment_dir(CONFIG["exp_name"])

    print("\n>>> 实验启动（YOLOv8-Seg + VLM）")
    print(f">>> 实验名称: {CONFIG['exp_name']}")
    print(f">>> 实验目录: {SAVE_DIR}")
    print(f">>> 数据文件夹: {DATA_FOLDERS}")
    print(f">>> 模型: {CONFIG['model']}")
    print(">>> 分割模型: YOLOv8l-Seg")

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

        imgs = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        for i, img in enumerate(imgs):
            all_tasks.append((img, folder, clients[i % len(clients)], global_labels, CONFIG))

    print(f">>> 任务分发完毕，共计 {len(all_tasks)} 个图片请求。")

    # 输出 CSV
    out_csv = os.path.join(SAVE_DIR, f"{CONFIG['exp_name']}.csv")
    results = []

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for row in tqdm(
                ex.map(process_single_image, all_tasks), total=len(all_tasks), desc="YOLOv8-Seg + VLM 推理"
            ):
                writer.writerow(row)
                f.flush()
                results.append(row)

    # 计算评估指标
    metrics = calculate_and_report(results)

    # 保存汇总
    summary_path = os.path.join(SAVE_DIR, "all_experiments_summary.csv")
    metrics.update(
        {
            "exp_name": CONFIG["exp_name"],
            "segmentor": "yolov8l-seg",
            "folders": len(DATA_FOLDERS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    file_exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8-sig") as f:
        dict_writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            dict_writer.writeheader()
        dict_writer.writerow(metrics)

    print(f"\n>>> 实验结束！详细结果已保存: {out_csv}")


if __name__ == "__main__":
    main()
