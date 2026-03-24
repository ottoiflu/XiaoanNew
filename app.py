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
from config.settings import settings

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
YOLO_SEG_WEIGHTS = "/root/XiaoanNew/weights/best.pt"

# 尝试加载 YOLOv8-Seg 模型
ai_engine = None
try:
    from yolov8_seg_inference import load_yolov8_seg

    print(f"🚀 正在加载 YOLOv8-Seg 模型: {YOLO_SEG_WEIGHTS}")
    ai_engine = load_yolov8_seg(YOLO_SEG_WEIGHTS, device="cuda:0")
    print("✅ YOLOv8-Seg 模型加载成功!")

except ImportError as e:
    print(f"⚠️ 警告: 无法导入 YOLOv8-Seg 推理模块 ({e})")
    print("⚠️ 尝试回退到 MaskRCNN...")

    # 回退到 MaskRCNN
    try:
        from mask_inference import MaskRCNNInference

        MASKRCNN_WEIGHTS = "/root/yk/maskrcnn_simple/MaskRCNN_Xiaoan_4class_v2.pth"
        ai_engine = MaskRCNNInference(MASKRCNN_WEIGHTS)
        print("✅ MaskRCNN 模型加载成功 (回退模式)")
    except Exception as e2:
        print(f"❌ 警告: AI 模型加载失败 ({e2})。实时检测将无法使用。")
        ai_engine = None

except Exception as e:
    print(f"❌ 警告: YOLOv8-Seg 模型加载失败 ({e})")
    ai_engine = None


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

        response = ocr_client.chat.completions.create(
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
        # 步骤 A: 调用云端 OCR 识别车牌
        # -----------------------------------------------------
        plate_number = recognize_license_plate(processed_bytes)

        has_plate = plate_number is not None
        if not has_plate:
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
        # 步骤 B: 调用 AI 引擎获取环境掩膜
        # -----------------------------------------------------
        parking_lane_found = False
        curb_found = False
        tactile_paving_found = False

        if ai_engine:
            ai_result = ai_engine.predict_static_json(img_bytes)
            detections = ai_result.get("detections", [])

            for det in detections:
                label = det.get("label", "")
                if label == "parking lane" or label == "parking_lane":
                    parking_lane_found = True
                elif label == "Curb" or label == "curb":
                    curb_found = True
                elif label == "Tactile paving" or label == "tactile_paving":
                    tactile_paving_found = True

            print(f"[AI检测] 停车线:{parking_lane_found}, 马路牙子:{curb_found}, 盲道:{tactile_paving_found}")

        # -----------------------------------------------------
        # 步骤 C: 综合判断
        # -----------------------------------------------------
        is_valid_parking = False
        message = ""

        # 如果检测到盲道，可能需要更严格的判断（这里暂时只记录）
        if tactile_paving_found:
            # 实际业务中可能需要判断车辆是否压在盲道上
            print("[警告] 检测到盲道，需注意停车位置")

        if parking_lane_found:
            is_valid_parking = True
            message = "规范停车 (检测到停车线)"
        elif curb_found:
            is_valid_parking = True
            message = "停车位置确认 (检测到马路牙子)"
        else:
            # 有车牌但无明确停车标识
            is_valid_parking = True
            message = "停车位置确认 (车牌清晰)"

        confidence = 0.95 if is_valid_parking else 0.45

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
            },
        }

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
