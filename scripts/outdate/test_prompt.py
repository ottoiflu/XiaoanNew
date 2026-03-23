import os
import base64
import csv
import re
import io
import json
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from PIL import Image

# ================= 配置区域 =================

BASE_URL = "https://api.ppinfra.com/openai"
API_KEY = "REDACTED_API_KEY_3"
MODEL = "qwen/qwen3-vl-235b-a22b-instruct"

# 路径配置
IMAGE_DIR = r"/root/XiaoanNew/App_collected_dataset/test/2025-12-31"
LABELS_PATH = os.path.join(IMAGE_DIR, "labels.txt")
VLM_OUTPUT_CSV = r"/root/XiaoanNew/App_collected_dataset/test/qwen3_structured_test_results.csv"

# 并发数量
MAX_WORKERS = 50

# 图片压缩配置 (建议略微提升分辨率以辅助识别标线)
MAX_IMAGE_SIZE = (768, 768) 
JPEG_QUALITY = 80

PROMPT = """
<|im_start|>system
- Role: 具有常识的共享单车运维主管。
- Task: 判断照片中【最明显】的一辆电动车是否停放得“可以接受”。
- Principle: 【疑罪从无】。除非车辆明显阻碍交通、倒地或严重乱停，否则请倾向于判定为合格（yes）。

【三级判定准则】:

1. 空间位置（第一重要）:
   - 只要车辆靠近 [标线内]、[马路牙子边]、[墙边] 或 [树边] 停放，且没有横在路中央，即可判为合格。
   - 即使车辆没有完全进入白线框，只要车轮或脚撑在线附近，且方向正确，也视为合格。

2. 停放角度（放宽标准）:
   - 垂直标准：只要车身不是与路缘“平行”摆放，看起来有“垂直对齐”的意图，角度在 45°-135° 之间均视为合格。
   - 平行标准：如果是在窄路靠边，车身与墙面平行也是合格的。
   - 只有当车辆“横七竖八”地斜在路中间，明显比周围车辆突兀时，才判为不合格。

3. 绝对违规（一票否决）:
   - 车辆压在盲道（黄色凸起带）上。
   - 车辆倒在地上。
   - 车辆停在机动车道或完全堵死了人行道出口。

【输出要求】
必须严格按 JSON 回答，分析理由要体现对“不挡路”的考量：
{
  "reference_frame": "描述看到的参考物(线/路缘/墙/无)",
  "compliance_logic": "简述为什么觉得它挡路或不挡路",
  "is_compliant": boolean,
  "reason": "简述理由"
}
<|im_end|>
"""

# ================= 辅助工具函数 =================

def norm_yesno(x: str) -> str:
    if x is None: return ""
    s = str(x).strip().lower()
    if s in ("yes", "y", "true", "1", "合格", "合规"): return "yes"
    if s in ("no", "n", "false", "0", "不合格", "违规"): return "no"
    return ""

def parse_vlm_response(response_text):
    """增强版解析逻辑：优先解析JSON，失败则回退到正则"""
    response_text = response_text.strip()
    try:
        # 尝试寻找并提取 JSON 块
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            # 将布尔值转换为 yes/no
            res = "yes" if data.get("is_compliant") is True else "no"
            reason = data.get("reason", "")
            return res, reason
    except:
        pass

    # 兜底正则
    result_match = re.search(r'"is_compliant":\s*(true|false)', response_text, re.I)
    if result_match:
        return "yes" if result_match.group(1).lower() == "true" else "no", "Regex Fallback"
    
    return "unknown", response_text[:100]

def compress_and_encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA': img = img.convert('RGB')
            img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            in_mem_file = io.BytesIO()
            img.save(in_mem_file, format='JPEG', quality=JPEG_QUALITY)
            return base64.b64encode(in_mem_file.getvalue()).decode('utf-8')
    except:
        return None

# ================= 核心处理逻辑 =================

def process_single_image(args):
    image_name, client, labels_dict = args
    image_path = os.path.join(IMAGE_DIR, image_name)
    ground_truth = labels_dict.get(image_name, "N/A")
    
    base64_image = compress_and_encode_image(image_path)
    if not base64_image:
        return [image_name, 'error', ground_truth, 'Image Error']

    try:
        chat_completion_res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.2, # 极低温度保证稳定性
            top_p=0.1
        )
        vlm_output = chat_completion_res.choices[0].message.content
        result, reason = parse_vlm_response(vlm_output)
        return [image_name, result, ground_truth, reason]

    except Exception as e:
        return [image_name, 'error', ground_truth, str(e)]

def run_inference():
    print(f"\n>>> 阶段一：开始 VLM 结构化推理...")
    # 读取标签
    labels_dict = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2: labels_dict[parts[0].strip()] = norm_yesno(parts[1])

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with open(VLM_OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_name', 'result', 'ground_truth', 'reason'])

        tasks = [(f, client, labels_dict) for f in image_files]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for row in tqdm(executor.map(process_single_image, tasks), total=len(tasks)):
                csv_writer.writerow(row)
                csvfile.flush()

# ================= 指标评估 =================

def calculate_metrics():
    print(f"\n>>> 阶段二：评估准确率...")
    tp, tn, fp, fn = 0, 0, 0, 0
    with open(VLM_OUTPUT_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt, pred = norm_yesno(row['ground_truth']), norm_yesno(row['result'])
            if gt == 'yes':
                if pred == 'yes': tp += 1
                elif pred == 'no': fn += 1
            elif gt == 'no':
                if pred == 'no': tn += 1
                elif pred == 'yes': fp += 1

    total = tp + tn + fp + fn
    if total == 0: return
    
    acc = (tp + tn) / total
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0

    print(f"\n{'='*15} 最终报告 {'='*15}")
    print(f"准确率: {acc:.2%} | F1-Score: {f1:.2f}")
    print(f"漏抓违停(FP): {fp} | 过于严苛(FN): {fn}")
    print(f"{'='*40}")

if __name__ == "__main__":
    run_inference()
    calculate_metrics()