import os
import base64
import csv
import re
import io
import torch
import torchvision
import numpy as np
import cv2
import threading
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ================= 1. 配置区域 =================

DEBUG_MODE = True  # 调试开关，设为 True 会保存画线图
DEBUG_SAVE_DIR = r"/root/XiaoanNew/App_collected_dataset/test/debug_visuals"

if DEBUG_MODE and not os.path.exists(DEBUG_SAVE_DIR):
    os.makedirs(DEBUG_SAVE_DIR)
# VLM 配置
BASE_URL = "https://api.ppinfra.com/openai"
API_KEY = "REDACTED_API_KEY_3"
VLM_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"

# Mask R-CNN 配置
MASK_RCNN_WEIGHTS = "/root/yk/maskrcnn_simple/MaskRCNN_Xiaoan_4class_v2.pth" # 请修改为你的真实权重路径
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径配置
IMAGE_DIR = r"/root/XiaoanNew/App_collected_dataset/test/2025-12-31"
LABELS_PATH = os.path.join(IMAGE_DIR, "labels.txt")
OUTPUT_CSV = r"/root/XiaoanNew/App_collected_dataset/test/combined_test_results.csv"

# 并发与性能
MAX_WORKERS = 20  # 由于包含本地模型推理，建议下调至 20 左右避免显存波动
MAX_IMAGE_SIZE = (640, 640) # 发送给 VLM 的分辨率

# 推理锁（防止多线程同时挤爆显存）
model_lock = threading.Lock()

# --- Version 15: 几何增强对齐版 Prompt ---
PROMPT = """你是一位交通停放审核专家。当前图中已由AI预标注了几何辅助线：
1. 绿色直线：代表【电动车的中轴线】。
2. 红色直线：代表【停车位边界、马路牙子或参考线】。

【判定标准】
- 合格 (yes)：绿色线与红色线呈“垂直”趋势（夹角在 70-110 度之间，呈 T 或 L 字型）。
- 合格 (yes)：如果图中没有红色线，但绿色线与其他已停放车辆的方向保持平行。
- 不合格 (no)：绿色线与红色线呈“平行”趋势（像二字型），或绿色线与红色线夹角约为 45 度（斜切占道）。

【一票否决】
- 压盲道（黄色纹路）、车辆倒地、严重阻碍通行。

请优先参考红绿线的夹角关系给出结论。
result: yes 或 no
reason: 简述夹角状态。示例："[几何分析] 绿线与红线呈85度垂直，主体在界内，合格。"
"""

# ================= 2. 感知引擎 (Mask R-CNN) =================

class MaskRCNNInference:
    def __init__(self, weights_path):
        self.kinds = ['_background_', 'electric_bike', 'parking_lane', 'curb']
        self.model = self._load_model(weights_path)
        self.box_conf = 0.5
        self.mask_conf = 0.5

    def _load_model(self, weights_path):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        num_classes = 4
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        return model

    def get_annotated_base64(self, image_path, save_path=None):
        """核心方法：识别并画线，返回Base64"""
        img_cv = cv2.imread(image_path)
        if img_cv is None: return None, False, 0, 0
        h, w = img_cv.shape[:2]
        
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(Image.fromarray(img_rgb)).to(DEVICE)
        
        with model_lock: # 确保模型推理是线程安全的
            with torch.no_grad():
                predictions = self.model([img_tensor])[0]

        scores = predictions['scores'].cpu().numpy()
        inds = np.where(scores >= self.box_conf)[0]
        
        has_lines = False
        if len(inds) > 0:
            masks = predictions['masks'].cpu().numpy()[inds]
            labels = predictions['labels'].cpu().numpy()[inds]

            for i in range(len(inds)):
                m = (masks[i] > self.mask_conf).squeeze().astype(np.uint8)
                label = labels[i]
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                cnt = max(contours, key=cv2.contourArea)

                if label == 1: # Bike -> 绿色中轴线
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect).astype(np.int32)
                    p1 = ((box[0][0] + box[1][0]) // 2, (box[0][1] + box[1][1]) // 2)
                    p2 = ((box[2][0] + box[3][0]) // 2, (box[2][1] + box[3][1]) // 2)
                    cv2.line(img_cv, p1, p2, (0, 255, 0), 10) # 绿色轴线
                    has_lines = True
                elif label in [2, 3]: # Lane/Curb -> 红色参考线
                    if cv2.contourArea(cnt) < 500: continue
                    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((w - x) * vy / vx) + y)
                    cv2.line(img_cv, (w - 1, righty), (0, lefty), (0, 0, 255), 10) # 红色参考线
                    has_lines = True

        # --- 新增：如果处于调试模式，保存画好线的原图 ---
        if save_path:
            # 可以在图上印一下是否检测到了线条，方便排查
            status_text = "Lines Detected" if has_lines else "No Lines"
            cv2.putText(img_cv, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(save_path, img_cv)

        # 之后再进行给 VLM 的压缩处理
        img_resized = cv2.resize(img_cv, MAX_IMAGE_SIZE)
        _, buffer = cv2.imencode('.jpg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return b64_str, has_lines, w, h

# 初始化引擎
mask_engine = MaskRCNNInference(MASK_RCNN_WEIGHTS)

# ================= 3. 辅助工具 =================

def norm_yesno(x: str) -> str:
    if not x: return ""
    s = str(x).strip().lower()
    if s in ("yes", "y", "true", "1"): return "yes"
    if s in ("no", "n", "false", "0"): return "no"
    return ""

def parse_vlm_response(text):
    result = 'unknown'
    reason = text
    res_m = re.search(r"result:\s*(yes|no)", text, re.I)
    rea_m = re.search(r"reason:(.*)", text, re.I | re.S)
    if res_m: result = res_m.group(1).lower()
    if rea_m: reason = rea_m.group(1).strip()
    return result, reason

def read_labels(path):
    labels = {}
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2: labels[parts[0].strip()] = norm_yesno(parts[1])
    return labels

# ================= 4. 处理逻辑 =================

def process_single_image(args):
    image_name, client, labels_dict = args
    image_path = os.path.join(IMAGE_DIR, image_name)
    gt = labels_dict.get(image_name, "N/A")
    
    save_path = os.path.join(DEBUG_SAVE_DIR, f"debug_{image_name}") if DEBUG_MODE else None
    # 步骤 1: Mask R-CNN 划线
    b64_img, has_lines, w, h = mask_engine.get_annotated_base64(image_path)
    if b64_img is None:
        return [image_name, 0, 0, 'error', gt, 'Read Error']
    b64_img, has_lines, w, h = mask_engine.get_annotated_base64(image_path, save_path=save_path)
    # 步骤 2: VLM 仲裁
    try:
        # 如果 Mask R-CNN 没画出线，给 VLM 一个小提示
        final_prompt = PROMPT if has_lines else PROMPT + "\n(注：AI未能自动勾勒线条，请根据原始图像特征判断)"
        
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": final_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                ]
            }],
            max_tokens=200,
            temperature=0.1
        )
        vlm_raw = response.choices[0].message.content
        pred, reason = parse_vlm_response(vlm_raw)
        return [image_name, w, h, pred, gt, reason]
    except Exception as e:
        return [image_name, w, h, 'error', gt, str(e)]

def main():
    print(f"\n>>> 启动集成推理系统 (模型: {VLM_MODEL})...")
    labels_dict = read_labels(LABELS_PATH)
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(image_files)} 张图片，开始处理...")

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'width', 'height', 'result', 'ground_truth', 'reason'])

        tasks = [(img, client, labels_dict) for img in image_files]
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # map 可以保持顺序，但 as_completed 在 tqdm 下更直观
            futures = [executor.submit(process_single_image, t) for t in tasks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
                writer.writerow(future.result())

    print(f"\n>>> 所有任务完成！结果保存在: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()