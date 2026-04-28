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
import io
import json
import os
import sys
import traceback
from datetime import datetime

from flask import Flask, jsonify, request, send_file

# 引入 OpenAI 客户端用于调用云端 OCR
from openai import OpenAI
from PIL import Image
from werkzeug.utils import secure_filename

# 导入配置模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.config.settings import settings
from modules.cv.image_utils import calculate_iou_and_overlap, combine_masks, draw_wireframe_visual, encode_image_to_base64
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
YOLO_SEG_WEIGHTS = "/root/XiaoanNew/assets/weights/best.pt"

# 尝试加载 YOLOv8-Seg 模型
ai_engine = None
try:
    from modules.cv.yolov8_inference import load_yolov8_seg

    print(f"🚀 正在加载 YOLOv8-Seg 模型: {YOLO_SEG_WEIGHTS}")
    ai_engine = load_yolov8_seg(YOLO_SEG_WEIGHTS, device="cuda:0")
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
VLM_PROMPT_ID = "cv_enhanced_p5"

vlm_client = None
_scoring_engine = None
try:
    if settings.VLM_API_KEY:
        vlm_client = OpenAI(base_url=settings.API_BASE_URL, api_key=settings.VLM_API_KEY)
        _scoring_engine = ScoringEngine()
        print(f"✅ VLM 合规分析客户端初始化成功，模型: {VLM_MODEL}")
    else:
        print("⚠️ VLM_API_KEYS 未配置，合规分析将回退到规则判断")
except Exception as _e:
    print(f"❌ VLM 合规分析客户端初始化失败: {_e}")


# =========================================================
# 辅助函数: 调用云端 OCR
# =========================================================
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
                            "text": "请识别图片中的电动车车牌号码。请直接输出车牌号字符串，不要包含任何标点符号或其他解释性文字。如果图片中没有车牌，请回答'无'。",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=50,
        )

        result_text = response.choices[0].message.content.strip()
        print(f"📋 [云端OCR] 识别结果: {result_text}")

        if "无" in result_text or len(result_text) < 3:
            return None

        return result_text

    except Exception as e:
        print(f"❌ OCR 调用失败: {e}")
        return None


def _rule_based_judgment(parking_lane: bool, curb: bool, tactile: bool):
    """基于 CV 检测结果的规则判断（VLM 不可用时的降级方案）"""
    if tactile:
        return False, 0.3, "停车违规：检测到盲道"
    if parking_lane:
        return True, 0.7, "规范停车（检测到停车线）"
    if curb:
        return True, 0.65, "停车位置确认（检测到马路牙子）"
    return True, 0.5, "停车位置确认（车牌清晰）"


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

    流程:
    1. 裁剪图片下方区域
    2. 云端 OCR 识别车牌
    3. AI 模型检测停车线/马路牙子/盲道
    4. 综合判断停车合规性
    """
    try:
        if "file" not in request.files:
            return jsonify({"code": 400, "message": "No file"}), 400

        file = request.files["file"]
        img_bytes = file.read()

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

                cropped_img = pil_image.crop((0, y1, w, y2))

                buf = io.BytesIO()
                filt_format = pil_image.format if pil_image.format else "JPEG"
                cropped_img.save(buf, format=filt_format)
                processed_bytes = buf.getvalue()

                print(f"[预处理] 图片已裁剪: 原尺寸({w}x{h}) -> 裁剪区域 y={y1:.1f}~{y2:.1f}")
            else:
                processed_bytes = img_bytes
        except Exception as crop_err:
            print(f"[预处理警告] 裁剪失败，将使用原图: {crop_err}")
            processed_bytes = img_bytes

        # -----------------------------------------------------
        # 步骤 A: 车牌识别（优先使用客户端传入，跳过云端 OCR）
        # -----------------------------------------------------
        client_plate = request.form.get("plate_number", "").strip()
        if client_plate:
            plate_number = client_plate
            print(f"[优化] 使用客户端 OCR 车牌: {plate_number}（跳过云端 OCR）")
        else:
            plate_number = recognize_license_plate(processed_bytes)

        if not plate_number or len(plate_number) < 3:
            return jsonify(
                {
                    "is_valid": False,
                    "message": "未检测到清晰车牌，请对准车牌重拍",
                    "confidence": 0.0,
                    "plate_number": "未识别",
                }
            ), 200

        print(f"[业务逻辑] 识别到车牌: {plate_number}")

        # -----------------------------------------------------
        # 步骤 B: YOLOv8-Seg 实例分割 + 端侧几何指标计算
        # -----------------------------------------------------
        parking_lane_found = False
        curb_found = False
        tactile_paving_found = False
        is_valid_parking = False
        confidence = 0.0
        message = ""
        vlm_analysis = None
        cv_detections = []

        if ai_engine:
            seg_result = ai_engine.predict(processed_bytes)
            raw_img = seg_result["image_raw"]
            objects = seg_result["objects"]
            H, W = seg_result["image_size"]

            class_counts = {"Electric bike": 0, "Curb": 0, "parking lane": 0, "Tactile paving": 0}
            main_bike_mask, main_bike_conf = None, -1.0

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
                if label == "Electric bike" and obj["confidence"] > main_bike_conf:
                    main_bike_conf = obj["confidence"]
                    main_bike_mask = obj.get("mask")

            print(f"[AI检测] 停车线:{parking_lane_found}, 马路牙子:{curb_found}, 盲道:{tactile_paving_found}")

            # 几何关系计算
            geo = {
                "main_vehicle_detected": main_bike_mask is not None,
                "overlap_with_parking_lane": 0.0,
                "iou_with_parking_lane": 0.0,
                "overlap_with_tactile_paving": 0.0,
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
                if geo["overlap_with_parking_lane"] > 0.8:
                    geo["status_inference"] = "Likely Compliant (High Overlap)"
                elif geo["overlap_with_parking_lane"] < 0.1:
                    geo["status_inference"] = "Likely Out of Bounds"

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
                        "geometry_analysis": geo,
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
                    print(f"[VLM] 原始响应（前200字符）: {vlm_text[:200]}")

                    vlm_result = parse_vlm_response(vlm_text)
                    if vlm_result.is_valid:
                        score_result = _scoring_engine.score(*vlm_result.statuses)
                        is_valid_parking = score_result.is_compliant
                        confidence = score_result.final_score
                        vlm_analysis = {
                            "composition": vlm_result.composition,
                            "angle": vlm_result.angle,
                            "distance": vlm_result.distance,
                            "context": vlm_result.context,
                            "final_score": score_result.final_score,
                            "dimension_scores": score_result.dimension_scores,
                            "reason": str(vlm_result.reason)[:500],
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
        # 步骤 D: 保存证据
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

        status_dir = "parking_success" if is_valid_parking else "parking_violation"
        evidence_dir = os.path.join(UPLOAD_ROOT, "evidence", status_dir)
        os.makedirs(evidence_dir, exist_ok=True)

        safe_plate = secure_filename(plate_number)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_plate}.jpg"

        with open(os.path.join(evidence_dir, filename), "wb") as f:
            f.write(img_bytes)

        return jsonify(result_data), 200

    except Exception as e:
        print(f"Check Parking Error: {e}")
        traceback.print_exc()
        return jsonify({"code": 500, "message": str(e)}), 500


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

    app.run(host="0.0.0.0", port=5000, debug=False)
