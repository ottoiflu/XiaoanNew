"""
共享单车停放检测后端 API

功能模块：
1. 数据采集 (/api/collect/upload)
2. 实时掩膜分割 (/api/segmentation/detect)
2.1 静态图片分析 (/api/segmentation/detect_static)
3. 停车检测 (/api/test/check_parking)

模型：
- YOLOv8-Seg (实例分割)
- 云端 VLM (车牌识别)

作者: Auto-generated
日期: 2026-01-20
"""

import base64
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request, send_file

# 引入 OpenAI 客户端用于调用云端 OCR
from openai import OpenAI
from PIL import Image, ImageDraw
from werkzeug.utils import secure_filename

# 导入配置模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.config.settings import settings
from modules.cv.angle_inference import angle_between, analyze_angle, pca_2d_image
from modules.cv.blind_lane_check import check_blind_lane
from modules.cv.image_utils import (
    calculate_iou_and_overlap,
    combine_masks,
    compress_image,
    draw_wireframe_visual,
    encode_image_to_base64,
)
from modules.experiment.scoring import ScoringEngine
from modules.prompt.manager import load_prompt
from modules.vlm.parser import parse_vlm_response
from modules.vlm.retry import chat_completion_with_retry

# 添加脚本目录到路径以导入推理模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

app = Flask(__name__)

# =========================================================
# 1. 配置区域
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "App_collected_dataset")
TEMP_PROCESS_DIR = os.path.join(BASE_DIR, "temp_processing")

print(f"存储根目录: {UPLOAD_ROOT}")

os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(TEMP_PROCESS_DIR, exist_ok=True)

# --- 云端 OCR 配置 ---
# OCR 配置从环境变量加载
OCR_API_KEY = settings.OCR_API_KEY
OCR_BASE_URL = settings.API_BASE_URL
OCR_MODEL = settings.OCR_MODEL

try:
    ocr_client = OpenAI(base_url=OCR_BASE_URL, api_key=OCR_API_KEY)
except Exception as e:
    print(f"❌ OCR 客户端初始化失败: {e}")
    ocr_client = None

# --- YOLOv8-Seg 模型配置 ---
YOLO_SEG_WEIGHTS = settings.YOLO_WEIGHTS

# 尝试加载 YOLOv8-Seg 模型
ai_engine = None
try:
    from modules.cv.yolov8_inference import load_yolov8_seg

    print(f"🚀 正在加载 YOLOv8-Seg 模型: {YOLO_SEG_WEIGHTS}")
    ai_engine = load_yolov8_seg(YOLO_SEG_WEIGHTS, device=settings.INFERENCE_DEVICE)
    print("✅ YOLOv8-Seg 模型加载成功!")

except ImportError as e:
    print(f"⚠️ 警告: 无法导入 YOLOv8-Seg 推理模块 ({e})")
    print("⚠️ 尝试回退到 MaskRCNN...")

    # 回退到 MaskRCNN
    try:
        from modules.cv.mask_inference import MaskRCNNInference

        MASKRCNN_WEIGHTS = "/root/yk/maskrcnn_simple/MaskRCNN_Xiaoan_4class_v2.pth"
        ai_engine = MaskRCNNInference(MASKRCNN_WEIGHTS)
        print("✅ MaskRCNN 模型加载成功 (回退模式)")
    except Exception as e2:
        print(f"❌ 警告: AI 模型加载失败 ({e2})。实时检测将无法使用。")
        ai_engine = None

except Exception as e:
    print(f"❌ 警告: YOLOv8-Seg 模型加载失败 ({e})")
    ai_engine = None

# --- VLM 合规分析配置 ---
VLM_MODEL = settings.VLM_MODEL
VLM_PROMPT_ID = "cv_enhanced_v2_newdim_v2"

vlm_client = None
_scoring_engine = None
try:
    if settings.VLM_API_KEY:
        vlm_client = OpenAI(base_url=settings.API_BASE_URL, api_key=settings.VLM_API_KEY)
        scoring_config_path = os.path.join(BASE_DIR, "assets", "configs", "scoring_new4d_gs_best.yaml")
        _scoring_engine = ScoringEngine.from_yaml(scoring_config_path)
        print(f"✅ VLM 合规分析客户端初始化成功，模型: {VLM_MODEL}，评分配置: scoring_new4d_gs_best")
    else:
        print("⚠️ VLM_API_KEYS 未配置，合规分析将回退到规则判断")
except Exception as _e:
    print(f"❌ VLM 合规分析客户端初始化失败: {_e}")


# --- 结果缓存（相同图片短时间内重复请求直接返回） ---
_RESULT_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 300  # 秒，缓存有效期


def _get_cached_result(cache_key: str):
    """查询缓存，未命中或过期返回 None"""
    with _CACHE_LOCK:
        entry = _RESULT_CACHE.get(cache_key)
        if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
            return entry["data"]
        if entry:
            del _RESULT_CACHE[cache_key]
    return None


def _put_cached_result(cache_key: str, data: dict):
    """写入缓存并清理过期条目"""
    with _CACHE_LOCK:
        _RESULT_CACHE[cache_key] = {"data": data, "ts": time.time()}
        # 惰性清理过期条目
        expired = [k for k, v in _RESULT_CACHE.items() if (time.time() - v["ts"]) >= _CACHE_TTL]
        for k in expired:
            del _RESULT_CACHE[k]


# --- 并行执行器（OCR 与 YOLO 共享） ---
_executor = ThreadPoolExecutor(max_workers=4)


# =========================================================
# 辅助函数: 调用云端 OCR
# =========================================================
_PROVINCE_CHARS = set("京津沪渝冀豫云辽黑湘皮鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁")


def _is_valid_plate(plate: str) -> bool:
    """
    检查字符串是否符合中国车牌基本格式。

    接受两种格式：
    - 完整格式：省份汉字(1) + 城市字母(1) + 5位字母/数字 = 7字符（新能源最多8位）
    - 不含省份格式：城市字母(1) + 5位字母/数字 = 6字符（MLKit 可能漏识省份汉字）
    """
    clean = plate.strip().upper().replace(" ", "")
    if len(clean) < 5 or len(clean) > 8:
        return False
    # 完整格式：首字符为省份汉字
    if clean[0] in _PROVINCE_CHARS:
        if not clean[1].isalpha():
            return False
        return all(c.isalnum() for c in clean[2:])
    # 不含省份格式：首字符为字母（城市代码），后跟 4-6 位字母数字
    if clean[0].isalpha() and 5 <= len(clean) <= 7:
        return all(c.isalnum() for c in clean[1:])
    # Demo 格式：城市名（2-4个汉字）+ 纯数字（4-8位），用于 Demo 预设 GT 车牌
    if re.match(r'^[\u4e00-\u9fa5]{2,4}\d{4,8}$', clean):
        return True
    return False


def recognize_license_plate(image_bytes):
    """
    将图片字节流转为 Base64，调用云端大模型识别车牌
    """
    if not ocr_client:
        return None

    try:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = chat_completion_with_retry(
            ocr_client,
            model=OCR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请识别图片中的电动车或机动车号牌。"
                                "中国号牌格式为：1个汉字省份缩写 + 1个字母城市代码 + 5位字母/数字，共 7 个字符，"
                                "例如「粤B12345」。"
                                "请直接输出车牌号字符串（省份汉字+6位字符），"
                                "不要包含任何标点、空格或解释性文字。"
                                "如果图片中没有清晰可见的号牌，请回答「无」。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=30,
        )

        result_text = response.choices[0].message.content.strip()
        print(f"📋 [云端OCR] 识别结果: {result_text}")

        if "无" in result_text or len(result_text) < 3:
            return None

        # 格式校验：过滤掉明显不是车牌的识别结果
        if not _is_valid_plate(result_text):
            print(f"[OCR] 识别结果不符合车牌格式，忽略: {result_text}")
            return None

        return result_text.strip().upper().replace(" ", "")

    except Exception as e:
        print(f"❌ OCR 调用失败: {e}")
        return None


def _plates_match(a: str, b: str) -> bool:
    """
    判断两个车牌字符串是否一致。

    匹配策略（任一通过即视为一致）：
    1. 完全相同
    2. 逐字符差异 ≤ 1（容忍单字符 OCR 误识，如 0/O、1/I）
    3. 后 5 位数字/字母部分完全相同（省份/城市字符误识时的 fallback）
    """
    a = a.strip().upper().replace(" ", "")
    b = b.strip().upper().replace(" ", "")
    if a == b:
        return True
    if abs(len(a) - len(b)) > 2:
        # 长度差大于2时，仍尝试后5位比对
        pass
    else:
        diffs = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        if diffs <= 1:
            return True
    # fallback：比对后5位（省份/城市字符常被误识，但序列号部分更可靠）
    if len(a) >= 5 and len(b) >= 5 and a[-5:] == b[-5:]:
        return True
    return False


def _rule_based_judgment(parking_lane: bool, curb: bool, tactile: bool):
    """基于 CV 检测结果的规则判断（VLM 不可用时的降级方案）"""
    if tactile:
        return False, 0.3, "停车违规：检测到盲道"
    if parking_lane:
        return True, 0.7, "规范停车（检测到停车线）"
    if curb:
        return True, 0.65, "停车位置确认（检测到马路牙子）"
    return True, 0.5, "停车位置确认（车牌清晰）"


def _run_yolo(engine, image_bytes):
    """YOLO 推理封装，供 ThreadPoolExecutor 调用。跳过内置可视化生成以节省 CPU。"""
    return engine.predict(image_bytes, visual=False, retina_masks=True, max_input_size=1280)


# =========================================================
# 功能 1: 数据采集
# =========================================================
@app.route("/api/collect/upload", methods=["POST"])
def collect_upload():
    """数据采集接口"""
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file"}), 400

        file = request.files["file"]
        label = request.form.get("label", "unknown")
        date_str = request.form.get("date", datetime.now().strftime("%Y-%m-%d"))
        custom_path = request.form.get("custom_path", "").strip()
        raw_gt = request.form.get("ground_truth")
        ground_truth = str(raw_gt).strip().lower() if raw_gt else ""

        if custom_path:
            save_dir = os.path.join(UPLOAD_ROOT, custom_path.replace("../", ""))
        else:
            save_dir = os.path.join(UPLOAD_ROOT, label, date_str)
        os.makedirs(save_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(save_dir, f"{timestamp}_{filename}")
        file.save(final_path)

        if ground_truth and ground_truth not in ["null", "none", "no data"]:
            with open(os.path.join(save_dir, "labels.txt"), "a", encoding="utf-8") as f:
                f.write(f"{timestamp}_{filename}, {ground_truth}\n")

        return jsonify({"status": "success", "path": final_path}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================================================
# 功能 2: 实时掩膜分割 (流式返回 PNG)
# =========================================================
@app.route("/api/segmentation/detect", methods=["POST"])
def detect_mask_realtime():
    """
    实时掩膜分割接口

    输入: 图片文件 (multipart/form-data)
    输出: PNG 格式的透明掩码叠加层

    客户端可直接将返回的 PNG 叠加在原图上显示
    """
    try:
        if "file" not in request.files:
            return "No file", 400

        file = request.files["file"]

        if ai_engine is None:
            return "Model not loaded", 500

        # 调用 predict_memory 返回 PNG 字节流
        img_bytes = file.read()
        png_buffer = ai_engine.predict_memory(img_bytes)

        return send_file(png_buffer, mimetype="image/png")

    except Exception as e:
        traceback.print_exc()
        return str(e), 500


# =========================================================
# 功能 2.1: 静态图片分析 (返回 JSON)
# =========================================================
@app.route("/api/segmentation/detect_static", methods=["POST"])
def detect_static():
    """
    静态图片分析接口

    输入: 图片文件 (multipart/form-data)
    输出: JSON 格式的检测结果，包含：
        - status: 状态
        - detections: 检测对象列表
        - mask_base64: Base64 编码的可视化掩码
    """
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file"}), 400

        file = request.files["file"]

        if ai_engine is None:
            return jsonify({"status": "error", "message": "Model not loaded"}), 500

        img_bytes = file.read()
        result = ai_engine.predict_static_json(img_bytes)
        result["bike_mask_base64"] = result.get("bike_mask_base64", "")

        return jsonify({"status": "success", "data": result}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================================================
# 功能 3: 停车检测 (集成云端 OCR + 实例分割)
# =========================================================
@app.route("/api/test/check_parking", methods=["POST"])
def check_parking():
    """
    停车检测接口

    流程（OCR 与 YOLO 并行执行）:
    0. 查询结果缓存，命中直接返回
    1. 裁剪图片下方区域
    2. 并行: OCR 识别车牌 + YOLOv8-Seg 实例分割
    3. 车牌一致性交叉验证
    4. CV+VLM 联合合规判断
    5. 保存证据 + 写入缓存
    """
    try:
        if "file" not in request.files:
            return jsonify({"code": 400, "message": "No file"}), 400

        file = request.files["file"]
        img_bytes = file.read()

        # --- 步骤 0: 结果缓存查询 ---
        client_plate_raw = request.form.get("plate_number", "").strip().upper().replace(" ", "")
        cache_key = hashlib.md5(img_bytes).hexdigest() + "|" + client_plate_raw
        cached = _get_cached_result(cache_key)
        if cached is not None:
            print(f"[缓存] 命中: {cache_key[:16]}...")
            return jsonify(cached), 200

        # -----------------------------------------------------
        # 预处理: 裁剪下方 30% 区域
        # -----------------------------------------------------
        try:
            pil_image = Image.open(io.BytesIO(img_bytes))
            w, h = pil_image.size
            if w > 0:
                box_h = w / 3.0
                center_y = h * 0.7
                y1 = max(0, center_y - box_h / 2)
                y2 = min(h, center_y + box_h / 2)

                cropped_img = pil_image  # 跳过裁剪，完整图给YOLO避免漏检（路缘/标线/绿化带常在边缘）

                buf = io.BytesIO()
                filt_format = pil_image.format if pil_image.format else "JPEG"
                cropped_img.save(buf, format=filt_format)
                processed_bytes = buf.getvalue()

                print(f"[预处理] 图片已裁剪: 原尺寸({w}x{h}) -> 裁剪区域 y={y1:.1f}~{y2:.1f}")
                if compress_image:
                    try:
                        processed_bytes = compress_image(processed_bytes)
                        print("[预处理] 已压缩")
                    except Exception as comp_err:
                        print(f"[预处理警告] 压缩失败: {comp_err}")
            else:
                processed_bytes = img_bytes
        except Exception as crop_err:
            print(f"[预处理警告] 裁剪失败，将使用原图: {crop_err}")
            processed_bytes = img_bytes

        # -----------------------------------------------------
        # 步骤 A+B 并行: OCR 识别车牌 + YOLO 实例分割
        # OCR 使用原图（远程 I/O），YOLO 使用裁剪图（本地 GPU），无资源争抢
        # -----------------------------------------------------
        ocr_future = _executor.submit(recognize_license_plate, img_bytes)
        yolo_future = _executor.submit(_run_yolo, ai_engine, processed_bytes) if ai_engine else None

        # 等待 YOLO（通常更快，50-200ms）
        seg_result = yolo_future.result() if yolo_future else None

        # 等待 OCR（远程 API，1-5s）
        try:
            server_plate = ocr_future.result(timeout=15)
        except Exception as ocr_err:
            print(f"[OCR] 并行调用超时或失败: {ocr_err}")
            server_plate = None

        # -----------------------------------------------------
        # 步骤 A-post: 车牌一致性交叉验证
        # -----------------------------------------------------
        raw_client_plate = request.form.get("plate_number", "").strip().upper().replace(" ", "")
        client_plate = raw_client_plate if _is_valid_plate(raw_client_plate) else ""
        if raw_client_plate and not client_plate:
            print(f"[过滤] 客户端车牌格式非法，忽略: {raw_client_plate}")

        if client_plate and server_plate:
            if not _plates_match(client_plate, server_plate):
                print(f"[安全] 车牌不符: 客户端={client_plate}, 服务端 OCR={server_plate}")
                result = {
                    "is_valid": False,
                    "message": "图像中的车牌与扫描车牌不符，请确认对准自己的车辆后重试",
                    "confidence": 0.0,
                    "plate_number": server_plate,
                }
                _put_cached_result(cache_key, result)
                return jsonify(result), 200
            plate_number = client_plate
            print(f"[验证] 车牌一致: {plate_number}")
        elif client_plate:
            plate_number = client_plate
            print(f"[警告] 服务端未识别到车牌，使用客户端车牌: {plate_number}")
        elif server_plate:
            plate_number = server_plate
            print(f"[OCR] 使用服务端识别车牌: {plate_number}")
        else:
            plate_number = None

        if not plate_number or len(plate_number) < 3:
            plate_number = "未识别"
            print("[车牌] 未检测到清晰车牌，使用占位符，继续 CV+VLM 合规判定")

        print(f"[业务逻辑] 确认车牌: {plate_number}")

        # -----------------------------------------------------
        # 步骤 B-post: YOLO 检测结果后处理 + 几何指标计算
        # -----------------------------------------------------
        parking_lane_found = False
        curb_found = False
        tactile_paving_found = False
        green_belt_found = False
        is_valid_parking = False
        confidence = 0.0
        message = ""
        vlm_analysis = None
        cv_detections = []
        cv_angle_result = {"cv_judgment": "[N/A]", "curb_fallback": False, "angle_to_bike": None, "disambiguation": "", "n_line_types": 0}
        blind_result = {"iou": 0.0, "depth_diff": None, "override": False, "reason": ""}
        # 几何重叠指标默认值（seg_result 缺失时仍可安全序列化到响应）
        geo = {
            "overlap_with_parking_lane": 0.0,
            "overlap_with_tactile_paving": 0.0,
            "overlap_with_green_belt": 0.0,
        }

        if seg_result:
            raw_img = seg_result["image_raw"]
            objects = seg_result["objects"]
            H, W = seg_result["image_size"]

            class_counts = {"Electric bike": 0, "Curb": 0, "parking lane": 0, "Tactile paving": 0, "Green belt": 0}
            # 主车选取：取「距画面中心最近」的电动车，而非置信度最高者。
            # 前端引导阶段已将目标车对中到画面中央，此处以中心距离接住该约束，
            # 避免多车场景下误选到旁边的邻车（与标注口径一致：以中心车辆为准）。
            main_bike_mask, main_bike_center_dist = None, float("inf")
            img_cx, img_cy = W / 2.0, H / 2.0

            for obj in objects:
                label = obj["label"]
                cv_detections.append({"id": obj["id"], "label": label, "confidence": obj["confidence"], "bbox": obj["bbox"]})
                if label in class_counts:
                    class_counts[label] += 1
                if label == "parking lane":
                    parking_lane_found = True
                elif label == "Curb":
                    curb_found = True
                elif label == "Tactile paving":
                    tactile_paving_found = True
                elif label == "Green belt":
                    green_belt_found = True
                if label == "Electric bike":
                    bx1, by1, bx2, by2 = obj["bbox"]
                    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    center_dist = (bcx - img_cx) ** 2 + (bcy - img_cy) ** 2
                    if center_dist < main_bike_center_dist:
                        main_bike_center_dist = center_dist
                        main_bike_mask = obj.get("mask")

            print(f"[AI检测] 停车线:{parking_lane_found}, 马路牙子:{curb_found}, 盲道:{tactile_paving_found}, 绿化带:{green_belt_found}")

            # 几何关系计算
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

            # -------------------------------------------------
            # 步骤 B-2: CV 角度分析（独立于 VLM，纯几何）
            # -------------------------------------------------
            if main_bike_mask is not None and analyze_angle:
                try:
                    cv_angle_result = analyze_angle(objects, main_bike_mask, W, H)
                except Exception as e:
                    print(f"[CV角度] 异常: {e}")

            # -------------------------------------------------
            # 步骤 C: CV+VLM 联合合规判断
            # -------------------------------------------------
            if vlm_client and _scoring_engine:
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
                        load_prompt(VLM_PROMPT_ID)
                        + "\n\n# YOLOv8-Seg Detection & Geometry Analysis\n```json\n"
                        + structured_info
                        + "\n```"
                    )

                    print(f"[VLM] 调用合规分析，模型: {VLM_MODEL}")
                    vlm_resp = chat_completion_with_retry(
                        vlm_client,
                        model=VLM_MODEL,
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
                    )
                    vlm_text = vlm_resp.choices[0].message.content.strip()
                    print(f"[VLM] 原始响应: {vlm_text}")

                    vlm_result = parse_vlm_response(vlm_text)

                    # -------------------------------------------------
                    # 盲道检测 override（VLM 响应后、scoring 前）
                    # -------------------------------------------------
                    if main_bike_mask is not None and check_blind_lane:
                        try:
                            pil_image_for_depth = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            blind_result = check_blind_lane(main_bike_mask, objects, pil_image_for_depth)
                        except Exception as e:
                            print(f"[盲道检测] 异常: {e}")
                    if blind_result["override"]:
                        vlm_result.medium = "[不合规-盲道]"

                    # 绿化带 override：车压绿化带面积比 ≥0.01 判违规
                    if geo.get("overlap_with_green_belt", 0.0) >= 0.01:
                        vlm_result.medium = "[不合规-绿化]"
                        blind_result["override"] = True
                        blind_result["reason"] = f"车辆压绿化带 overlap_ratio={geo['overlap_with_green_belt']:.3f}"

                    if vlm_result.is_valid:
                        score_result = _scoring_engine.score(*vlm_result.statuses)
                        is_valid_parking = score_result.is_compliant
                        confidence = score_result.final_score
                        vlm_analysis = {
                            "position": vlm_result.position,
                            "medium": vlm_result.medium,
                            "angle": vlm_result.angle,
                            "state": vlm_result.state,
                            "position_confidence": vlm_result.position_confidence,
                            "medium_confidence": vlm_result.medium_confidence,
                            "angle_confidence": vlm_result.angle_confidence,
                            "state_confidence": vlm_result.state_confidence,
                            "final_score": score_result.final_score,
                            "dimension_scores": score_result.dimension_scores,
                            "reason": str(vlm_result.reason)[:500],
                            "adjustment_suggestion": vlm_result.adjustment_suggestion,
                        }
                        if is_valid_parking:
                            message = f"合规停车（综合评分 {confidence:.2f}）"
                        else:
                            dims_fail = [k for k, v in score_result.dimension_scores.items() if v < 0.5]
                            message = f"停车违规：{', '.join(dims_fail) if dims_fail else '综合评分不足'}"
                        print(f"[VLM] 合规判定: {'合规' if is_valid_parking else '违规'}, 评分: {confidence:.4f}")
                    else:
                        print(f"[VLM] 响应解析失败: {vlm_result.parse_error}，回退到规则判断")
                        is_valid_parking, confidence, message = _rule_based_judgment(
                            parking_lane_found, curb_found, tactile_paving_found
                        )

                except Exception as vlm_err:
                    print(f"[VLM] 调用异常: {vlm_err}，回退到规则判断")
                    traceback.print_exc()
                    is_valid_parking, confidence, message = _rule_based_judgment(
                        parking_lane_found, curb_found, tactile_paving_found
                    )
            else:
                is_valid_parking, confidence, message = _rule_based_judgment(
                    parking_lane_found, curb_found, tactile_paving_found
                )
        else:
            is_valid_parking = True
            confidence = 0.5
            message = "停车位置确认（AI 引擎不可用，仅凭车牌判断）"

        # -----------------------------------------------------
        # 步骤 D: 保存证据 + 写入缓存
        # -----------------------------------------------------
        result_data = {
            "is_valid": is_valid_parking,
            "plate_number": plate_number,
            "confidence": confidence,
            "message": message,
            "detections": {
                "parking_lane": parking_lane_found,
                "curb": curb_found,
                "tactile_paving": tactile_paving_found,
                "objects": cv_detections,
            },
        }
        if vlm_analysis:
            result_data["vlm_analysis"] = vlm_analysis
        result_data["cv_analysis"] = {
            "angle_judgment": cv_angle_result.get("cv_judgment", "[N/A]"),
            "angle_to_bike": cv_angle_result.get("angle_to_bike"),
            "disambiguation": cv_angle_result.get("disambiguation", ""),
            "curb_fallback": cv_angle_result.get("curb_fallback", False),
            "n_line_types": cv_angle_result.get("n_line_types", 0),
            "overlap_parking": round(float(geo.get("overlap_with_parking_lane", 0.0)), 4),
            "overlap_tactile": round(float(geo.get("overlap_with_tactile_paving", 0.0)), 4),
            "overlap_green_belt": round(float(geo.get("overlap_with_green_belt", 0.0)), 4),
            "green_belt_override": bool(geo.get("overlap_with_green_belt", 0.0) >= 0.01),
            "blind_lane": {
                "iou": blind_result.get("iou", 0.0),
                "depth_diff": blind_result.get("depth_diff"),
                "override": blind_result.get("override", False),
                "reason": blind_result.get("reason", ""),
            },
        }
        result_data["image_compressed"] = True

        _put_cached_result(cache_key, result_data)

        status_dir = "parking_success" if is_valid_parking else "parking_violation"
        evidence_dir = os.path.join(UPLOAD_ROOT, "evidence", status_dir)
        os.makedirs(evidence_dir, exist_ok=True)

        safe_plate = secure_filename(plate_number) or "unknown"
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_plate}.jpg"

        with open(os.path.join(evidence_dir, filename), "wb") as f:
            f.write(img_bytes)

        return jsonify(result_data), 200

    except Exception as e:
        print(f"Check Parking Error: {e}")
        traceback.print_exc()
        return jsonify({"code": 500, "message": str(e)}), 500


def _generate_mask_pca_image(img_bytes: bytes, save_path: str):
    """生成 YOLO 掩模图带 PCA 主轴叠加

    复用全局 ai_engine (new_best.pt) 跑分割 (imgsz=1024, conf=0.35)，
    各类 mask 半透明叠加 (Electric bike绿/Curb紫/parking lane黄/Tactile paving橙/Green belt天蓝)，
    对主车 mask 与停车线/路缘 mask 算 2D PCA 主轴并画线，标注夹角度数。

    Args:
        img_bytes: 原图字节流
        save_path: 输出 jpg 路径

    Returns:
        (mask_pca_image 文件名 or None, yolo_detections 列表)
    """
    if ai_engine is None:
        print("[GT掩模] ai_engine 未加载，跳过掩模图生成")
        return None, []

    try:
        seg = ai_engine.predict(img_bytes, conf=0.35, imgsz=1024, retina_masks=True, visual=False)
        objects = seg["objects"]
        img_array = seg["image_raw"]
        H, W = seg["image_size"]

        colors = {
            "Electric bike": (0, 255, 0),
            "Curb": (180, 80, 200),
            "parking lane": (255, 255, 0),
            "Tactile paving": (255, 165, 0),
            "Green belt": (0, 200, 255),
        }

        # 主车选取：距画面中心最近的 Electric bike
        main_bike_mask = None
        img_cx, img_cy = W / 2.0, H / 2.0
        best_dist = float("inf")
        for obj in objects:
            if obj["label"] != "Electric bike":
                continue
            bx1, by1, bx2, by2 = obj["bbox"]
            d = ((bx1 + bx2) / 2 - img_cx) ** 2 + ((by1 + by2) / 2 - img_cy) ** 2
            if d < best_dist:
                best_dist = d
                main_bike_mask = obj.get("mask")

        # 标线 mask：优先停车线，缺失或过小则降级到路缘
        parking_mask = combine_masks(objects, "parking lane")
        curb_mask = combine_masks(objects, "Curb")
        line_mask = parking_mask
        if (line_mask is None or line_mask.sum() < 50) and curb_mask is not None:
            line_mask = curb_mask

        # 半透明叠加各类 mask
        arr = img_array.astype(np.float32)
        for obj in objects:
            label = obj["label"]
            color = colors.get(label, (128, 128, 128))
            mask = obj.get("mask")
            if isinstance(mask, np.ndarray) and mask.sum() > 0:
                m = mask.astype(bool)
                if m.shape == arr.shape[:2]:
                    for c in range(3):
                        arr[:, :, c][m] = arr[:, :, c][m] * 0.5 + color[c] * 0.5
        overlay = Image.fromarray(arr.astype(np.uint8))
        draw = ImageDraw.Draw(overlay)

        def draw_axis(center, direction, color, length=160, width=4):
            """画 PCA 主轴双向线段"""
            c = np.array(center, dtype=float)
            d = np.array(direction, dtype=float)
            n = np.linalg.norm(d)
            if n < 1e-8:
                return
            d = d / n
            end = c + d * length
            start = c - d * length
            draw.line([float(start[0]), float(start[1]), float(end[0]), float(end[1])], fill=color, width=width)

        # PCA 主轴 + 夹角
        bike_c, bike_d = (None, None)
        if main_bike_mask is not None:
            bike_c, bike_d = pca_2d_image(main_bike_mask)
            if bike_c is not None:
                draw_axis(bike_c, bike_d, (0, 255, 0), 180, 4)  # 绿=车
        line_c, line_d = (None, None)
        if line_mask is not None and line_mask.sum() >= 50:
            line_c, line_d = pca_2d_image(line_mask)
            if line_c is not None:
                draw_axis(line_c, line_d, (255, 255, 0), 150, 3)  # 黄=标线/路缘
        angle_text = None
        if bike_d is not None and line_d is not None:
            angle_text = f"angle={angle_between(bike_d, line_d):.1f} deg"

        # 左上角标注
        y = 8
        if angle_text:
            draw.text((8, y), angle_text, fill=(255, 255, 255))
            y += 16
        draw.text((8, y), f"objects={len(objects)}", fill=(255, 255, 255))

        overlay.save(save_path, quality=85)

        detections = [
            {"label": obj["label"], "confidence": obj["confidence"], "bbox": obj["bbox"]}
            for obj in objects
        ]
        return os.path.basename(save_path), detections
    except Exception as e:
        print(f"[GT掩模] 生成失败: {e}")
        traceback.print_exc()
        return None, []


# =========================================================
# 功能 4: GT 收集 (App 测试模式上传样本+真值)
# =========================================================
@app.route("/api/test/submit_gt", methods=["POST"])
def submit_gt():
    """
    GT 收集接口 (App 测试模式)

    接收 App 测试模式上传的图片样本 + 模型判断结果 + 用户手动真值,
    保存到 data/gt_collected/{folder_name}/ 下, 用于后续模型评测.

    存储:
      - 图: data/gt_collected/{folder_name}/{timestamp}_{plate}.jpg
      - GT: data/gt_collected/{folder_name}/{timestamp}_{plate}.json
    """
    try:
        if "file" not in request.files:
            return jsonify({"ok": False, "message": "No file"}), 400

        file = request.files["file"]

        # --- 字段解析 ---
        plate_number_raw = request.form.get("plate_number", "").strip()
        model_is_valid_raw = request.form.get("model_is_valid", "").strip().lower()
        model_score_raw = request.form.get("model_score", "").strip()
        model_message = request.form.get("model_message", "").strip()
        user_gt = request.form.get("user_gt", "").strip().lower()
        folder_name_raw = request.form.get("folder_name", "").strip()
        timestamp_raw = request.form.get("timestamp", "").strip()

        # --- 字段解析（增强：四维状态 + CV 几何 + 时延，由 app 端 check_parking 结果回传） ---
        def _form_str(key: str) -> str:
            return request.form.get(key, "").strip()

        def _form_float(key: str):
            v = request.form.get(key, "").strip()
            if not v:
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def _form_bool(key: str) -> bool:
            return request.form.get(key, "").strip().lower() == "true"

        def _form_int(key: str):
            v = request.form.get(key, "").strip()
            if not v:
                return None
            try:
                return int(v)
            except ValueError:
                return None

        vlm_position = _form_str("vlm_position")
        vlm_medium = _form_str("vlm_medium")
        vlm_angle = _form_str("vlm_angle")
        vlm_state = _form_str("vlm_state")
        vlm_position_conf = _form_float("vlm_position_conf")
        vlm_medium_conf = _form_float("vlm_medium_conf")
        vlm_angle_conf = _form_float("vlm_angle_conf")
        vlm_state_conf = _form_float("vlm_state_conf")
        dim_score_position = _form_float("dim_score_position")
        dim_score_medium = _form_float("dim_score_medium")
        dim_score_angle = _form_float("dim_score_angle")
        dim_score_state = _form_float("dim_score_state")
        cv_angle_judgment = _form_str("cv_angle_judgment")
        cv_angle_to_bike = _form_float("cv_angle_to_bike")
        cv_disambiguation = _form_str("cv_disambiguation")
        cv_curb_fallback = _form_bool("cv_curb_fallback")
        overlap_parking = _form_float("overlap_parking")
        overlap_tactile = _form_float("overlap_tactile")
        overlap_green_belt = _form_float("overlap_green_belt")
        blind_override = _form_bool("blind_override")
        green_belt_override = _form_bool("green_belt_override")
        latency_ms = _form_int("latency_ms")

        # --- folder_name 安全校验: 只允许字母数字下划线横杠, 防路径穿越 ---
        if not folder_name_raw:
            return jsonify({"ok": False, "message": "folder_name is required"}), 400
        if not re.fullmatch(r"[A-Za-z0-9_-]+", folder_name_raw):
            return jsonify({"ok": False, "message": "folder_name contains invalid characters"}), 400
        folder_name = folder_name_raw

        # --- timestamp: 没传或非法则服务端生成; 只允许字母数字下划线横杠 ---
        if not timestamp_raw or not re.fullmatch(r"[A-Za-z0-9_-]+", timestamp_raw):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = timestamp_raw

        # --- plate_number: 空则 unknown, 清洗特殊字符 (保留字母数字与中文) ---
        if not plate_number_raw:
            plate_number_raw = "unknown"
        plate_clean = re.sub(r"[^A-Za-z0-9\u4e00-\u9fa5]", "", plate_number_raw)
        if not plate_clean:
            plate_clean = "unknown"

        # --- 创建目录 ---
        gt_root = os.path.join(BASE_DIR, "data", "gt_collected")
        save_dir = os.path.join(gt_root, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        # --- 文件名与路径 ---
        base_name = f"{timestamp}_{plate_clean}"
        image_path = os.path.join(save_dir, f"{base_name}.jpg")
        mask_pca_path = os.path.join(save_dir, f"{base_name}_mask_pca.jpg")
        gt_path = os.path.join(save_dir, f"{base_name}.json")

        # --- 保存原图（先读字节，便于复用跑分割） ---
        img_bytes = file.read()
        with open(image_path, "wb") as f:
            f.write(img_bytes)

        # --- 生成 YOLO 掩模图带 PCA 主轴叠加 ---
        mask_pca_image, yolo_detections = _generate_mask_pca_image(img_bytes, mask_pca_path)

        # --- 解析模型字段类型 ---
        model_is_valid = model_is_valid_raw == "true"
        try:
            model_score = float(model_score_raw) if model_score_raw else None
        except ValueError:
            model_score = None

        # --- 写 GT JSON ---
        gt_data = {
            "plate_number": plate_clean,
            "model_is_valid": model_is_valid,
            "model_score": model_score,
            "model_message": model_message,
            "user_gt": user_gt,
            "timestamp": timestamp,
            "folder_name": folder_name,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vlm_position": vlm_position,
            "vlm_medium": vlm_medium,
            "vlm_angle": vlm_angle,
            "vlm_state": vlm_state,
            "vlm_position_conf": vlm_position_conf,
            "vlm_medium_conf": vlm_medium_conf,
            "vlm_angle_conf": vlm_angle_conf,
            "vlm_state_conf": vlm_state_conf,
            "dim_score_position": dim_score_position,
            "dim_score_medium": dim_score_medium,
            "dim_score_angle": dim_score_angle,
            "dim_score_state": dim_score_state,
            "cv_angle_judgment": cv_angle_judgment,
            "cv_angle_to_bike": cv_angle_to_bike,
            "cv_disambiguation": cv_disambiguation,
            "cv_curb_fallback": cv_curb_fallback,
            "overlap_parking": overlap_parking,
            "overlap_tactile": overlap_tactile,
            "overlap_green_belt": overlap_green_belt,
            "blind_override": blind_override,
            "green_belt_override": green_belt_override,
            "latency_ms": latency_ms,
            "mask_pca_image": mask_pca_image,
            "yolo_detections": yolo_detections,
        }
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=2)

        return jsonify({"ok": True, "image_path": image_path, "gt_path": gt_path, "mask_pca_image": mask_pca_image}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "message": str(e)}), 500



# =========================================================
# 健康检查
# =========================================================
@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return jsonify(
        {
            "status": "ok",
            "model_loaded": ai_engine is not None,
            "model_type": type(ai_engine).__name__ if ai_engine else None,
            "ocr_available": ocr_client is not None,
        }
    ), 200


# =========================================================
# 启动服务
# =========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动共享单车停放检测后端服务")
    print("=" * 60)
    print(f"📁 存储根目录: {UPLOAD_ROOT}")
    print(f"🤖 AI引擎: {type(ai_engine).__name__ if ai_engine else 'None'}")
    print(f"📡 OCR服务: {'可用' if ocr_client else '不可用'}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=settings.FLASK_PORT, debug=False)
