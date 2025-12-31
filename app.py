import os
import io
import random
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np


# 引入 AI 推理类 (注意文件名是否匹配)
try:
    from mask_inference import MaskRCNNInference
except ImportError:
    # 兼容处理，防止文件名不对报错
    from mask_inference import MaskRCNNInference

app = Flask(__name__)

# =========================================================
# 1. 配置路径
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(BASE_DIR, "App_collected_dataset")
TEMP_PROCESS_DIR = os.path.join(BASE_DIR, "temp_processing")

print(f"存储根目录: {UPLOAD_ROOT}")

os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(TEMP_PROCESS_DIR, exist_ok=True)

# =========================================================
# 2. 初始化 AI 引擎
# =========================================================
try:
    WEIGHTS_PATH = "/root/yk/maskrcnn_simple/MaskRCNN_Xiaoan_MultiClass.pth"
    # 如果本地测试，解除下面注释
    # WEIGHTS_PATH = os.path.join(BASE_DIR, "MaskRCNN_Xiaoan_MultiClass.pth")
    
    print(f"正在加载 AI 模型: {WEIGHTS_PATH} ...")
    ai_engine = MaskRCNNInference(WEIGHTS_PATH)
    print("AI 模型加载成功！")
except Exception as e:
    print(f"警告: AI 模型加载失败 ({e})。实时检测将无法使用。")
    ai_engine = None


# =========================================================
# 功能 1: 数据采集
# =========================================================
@app.route('/api/collect/upload', methods=['POST'])
def collect_upload():
    try:
        print(f"\n--- [收到上传请求] ---")
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
            
        file = request.files['file']
        label = request.form.get('label', 'unknown') 
        date_str = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        custom_path = request.form.get('custom_path', '').strip()
        
        raw_gt = request.form.get('ground_truth')
        ground_truth = str(raw_gt).strip().lower() if raw_gt else "no data"

        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        # 构建路径
        if custom_path:
            safe_path = custom_path.replace('../', '').replace('..\\', '')
            save_dir = os.path.join(UPLOAD_ROOT, safe_path)
            print(f"路径模式: 自定义 -> {save_dir}")
        else:
            save_dir = os.path.join(UPLOAD_ROOT, label, date_str)
            print(f"路径模式: 标准 -> {save_dir}")

        os.makedirs(save_dir, exist_ok=True)

        # 保存文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%H%M%S")
        final_name = f"{timestamp}_{filename}"
        final_path = os.path.join(save_dir, final_name)
        
        file.save(final_path)
        print(f"图片已保存: {final_name}")

        # 记录真值
        if ground_truth and ground_truth not in ['null', 'none', 'no data']:
            txt_path = os.path.join(save_dir, "labels.txt")
            line_content = f"{final_name}, {ground_truth}\n"
            try:
                with open(txt_path, "a", encoding="utf-8") as f:
                    f.write(line_content)
                print(f"[记录成功] {line_content.strip()}")
            except Exception as e:
                print(f"[记录失败] {e}")

        return jsonify({
            "status": "success", 
            "path": final_path,
            "ground_truth": ground_truth
        }), 200

    except Exception as e:
        print(f"Server Error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================================================
# 功能 2: 实时掩膜分割 (流式返回 PNG)
# =========================================================
@app.route('/api/segmentation/detect', methods=['POST'])
def detect_mask_realtime():
    try:
        if 'file' not in request.files: return "No file", 400
        file = request.files['file']
        
        if ai_engine is None: return "Model not loaded", 500

        img_bytes = file.read()
        
        # 调用 inference.py 里的 predict_memory
        # 这里返回的是 BytesIO (PNG 图片流)
        result_image_stream = ai_engine.predict_memory(img_bytes)

        return send_file(result_image_stream, mimetype='image/png')

    except Exception as e:
        print(f"Seg Error: {e}")
        return str(e), 500


# =========================================================
# 功能 2.1: [新增] 静态图片分析 (返回 JSON)
# 解决 404 问题的关键接口
# =========================================================
@app.route('/api/segmentation/detect_static', methods=['POST'])
def detect_static():
    try:
        # 1. 接收文件
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        file = request.files['file']
        
        if ai_engine is None:
            return jsonify({"status": "error", "message": "Model not loaded"}), 500

        # 2. 读取流
        img_bytes = file.read()
        
        # 3. 调用 AI 引擎的 JSON 处理方法 (需要你在 inference.py 里添加)
        # 如果 inference.py 还没更新，这一步会报错 'AttributeError'
        result_data = ai_engine.predict_static_json(img_bytes)

        # 4. 返回 JSON
        return jsonify({
            "status": "success",
            "data": result_data
        }), 200

    except Exception as e:
        print(f"Static Seg Error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# =========================================================
# 功能 3: 停车规范检测 (逻辑增强版)
# 流程: 1.读取图片 -> 2.OCR识别车牌 -> 3.AI分割线/车 -> 4.几何判断
# =========================================================
@app.route('/api/test/check_parking', methods=['POST'])
def check_parking():
    try:
        # 1. 基础校验
        if 'file' not in request.files:
            return jsonify({"code": 400, "message": "No file"}), 400
        
        file = request.files['file']
        img_bytes = file.read() # 读取二进制流

        # -----------------------------------------------------
        # 步骤 A: 图像格式转换 (Bytes -> OpenCV)
        # -----------------------------------------------------
        # 将二进制流转为 numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        # 解码为 OpenCV 图像 (H, W, 3) BGR格式
        cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if cv_image is None:
            return jsonify({"code": 400, "message": "Image decode failed"}), 400

        # -----------------------------------------------------
        # 步骤 B: 车牌检测与 OCR (这里目前是模拟，需接入真实OCR)
        # -----------------------------------------------------
        # TODO: 这里接入 PaddleOCR 或 EasyOCR
        # 真实代码示例: 
        # ocr_result = ocr_engine.ocr(cv_image, cls=True)
        # plate_text = ocr_result[0][1][0]
        # plate_box = ocr_result[0][0] # [[x1,y1], [x2,y2], ...]
        
        # [模拟逻辑] 假设我们检测到了车牌
        # 在真实场景中，如果 OCR 返回空，直接 return "未检测到车牌，请按引导框重拍"
        has_plate = True 
        plate_number = "京A·88888" # 模拟识别到的车牌
        
        # 模拟车牌在图中的中心点 (假设用户听话，拍在图片下半部分中心)
        h, w, _ = cv_image.shape
        plate_center_y = h * 0.8 

        if not has_plate:
             return jsonify({
                "is_valid": False,
                "message": "未检测到车牌，请将车牌对准引导框重拍",
                "confidence": 0.0
            }), 200

        # -----------------------------------------------------
        # 步骤 C: 调用 AI 引擎获取环境掩膜 (白线/路沿石)
        # -----------------------------------------------------
        if ai_engine:
            # 复用之前的 json 接口逻辑获取检测框和掩膜
            # 注意：predict_static_json 接收的是 bytes
            ai_result = ai_engine.predict_static_json(img_bytes)
            detections = ai_result.get('detections', [])
            
            # 寻找“停车线”或“路沿石”的检测结果
            parking_lane_found = False
            for det in detections:
                if det['label'] == 'parking_lane': # 假设你的模型里有这个类别
                    parking_lane_found = True
                    break
        else:
            # 如果模型没加载，就只能瞎猜了
            parking_lane_found = False

        # -----------------------------------------------------
        # 步骤 D: 综合业务逻辑判断 (核心算法)
        # -----------------------------------------------------
        # 逻辑 1: 是否在电子围栏内？(这通常是 App 端传经纬度判断，这里假设 App 已过)
        
        # 逻辑 2: 视觉判断
        # 如果找到了车牌，且找到了停车线，且车牌位置在停车线区域内(简化逻辑)
        is_valid_parking = False
        message = ""

        if parking_lane_found:
            # 这里可以写更复杂的几何判断：计算车牌框是否在停车线掩膜内部
            # 目前简化为：只要检测到线，且检测到车牌，就算停好了
            is_valid_parking = True
            message = "规范停车"
        else:
            # 没找到线，可能是马路牙子场景，或者违停
            # 暂时模拟：如果车牌清晰，我们姑且认为在路边停好了（或者由人工二次审核）
            # 为了演示，这里设为 True，实际要严一点
            is_valid_parking = True 
            message = "停车位置确认 (路沿石模式)"

        # 生成置信度
        confidence = 0.95 if is_valid_parking else 0.45

        # -----------------------------------------------------
        # 步骤 E: 保存证据与返回
        # -----------------------------------------------------
        result_data = {
            "is_valid": is_valid_parking,
            "plate_number": plate_number, # 返回识别的车牌给用户确认
            "confidence": confidence,
            "message": message
        }

        # 保存证据图
        status_dir = "parking_success" if is_valid_parking else "parking_violation"
        evidence_dir = os.path.join(UPLOAD_ROOT, "evidence", status_dir)
        os.makedirs(evidence_dir, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{plate_number}.jpg"
        
        # 这里的 cv_image 是解码后的，建议直接写回原 bytes 以免压缩损失，或者用 cv2.imwrite
        with open(os.path.join(evidence_dir, filename), "wb") as f:
            f.write(img_bytes)

        return jsonify(result_data), 200

    except Exception as e:
        print(f"Check Parking Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"code": 500, "message": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11470, debug=True, threaded=True)