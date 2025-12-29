import os
import io
import random
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image, ImageDraw

app = Flask(__name__)

# =========================================================
# 核心修改 1: 使用绝对路径 (解决找不到文件的问题)
# =========================================================
# 获取当前 app.py 文件所在的绝对目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 拼接出绝对存储路径
UPLOAD_ROOT = os.path.join(BASE_DIR, "App_collected_dataset")
TEMP_PROCESS_DIR = os.path.join(BASE_DIR, "temp_processing")

# 打印一下路径，启动时方便你在控制台确认位置
print(f"📂 存储根目录: {UPLOAD_ROOT}")

os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(TEMP_PROCESS_DIR, exist_ok=True)


# ---------------------------------------------------------
# 功能 1: 数据采集与打标 (增强版：带校验)
# ---------------------------------------------------------
@app.route('/api/collect/upload', methods=['POST'])
def collect_upload():
    try:
        # 1. 基础参数校验
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
            
        file = request.files['file']
        label = request.form.get('label', 'unknown') 
        date_str = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        # =========================================================
        # 核心修改 2: 检测上传的数据是否有效
        # =========================================================
        # A. 检查文件大小
        file.seek(0, os.SEEK_END) # 移动指针到末尾
        file_length = file.tell() # 获取大小
        file.seek(0) # ⚠️ 必须把指针移回开头，否则后面 save 会保存空文件！

        if file_length == 0:
            return jsonify({"status": "error", "message": "File is empty (0 bytes)"}), 400
        
        # B. 检查是否为有效图片 (防止上传坏文件)
        try:
            # 使用 PIL 尝试打开，verify() 仅校验头信息，速度快
            img = Image.open(file)
            img.verify() 
            file.seek(0) # ⚠️ PIL 读取后指针也会动，必须再次复位！
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid image file: {str(e)}"}), 400

        # 2. 构建保存路径
        save_dir = os.path.join(UPLOAD_ROOT, label, date_str)
        os.makedirs(save_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%H%M%S_")
        final_path = os.path.join(save_dir, timestamp + filename)
        
        # 3. 保存文件
        file.save(final_path)
        
        # =========================================================
        # 核心修改 3: 保存后双重确认 (Double Check)
        # =========================================================
        if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
            print(f"[成功] 图片已确实写入磁盘: {final_path}")
            print(f"大小: {file_length} bytes")
            return jsonify({
                "status": "success", 
                "path": final_path,
                "size": file_length
            }), 200
        else:
            print(f"[失败] 文件保存逻辑执行了，但在磁盘上找不到或大小为0: {final_path}")
            return jsonify({"status": "error", "message": "Save failed on disk check"}), 500

    except Exception as e:
        print(f" Server Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ... (功能 2 和 功能 3 保持不变，但建议也确认一下 random 库是否导入) ...

# ---------------------------------------------------------
# 功能 2: 掩膜分割检测
# ---------------------------------------------------------
@app.route('/api/segmentation/detect', methods=['POST'])
def detect_mask():
    try:
        if 'file' not in request.files:
            return "No file", 400
        file = request.files['file']
        
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        draw.rectangle([w*0.25, h*0.25, w*0.75, h*0.75], fill=(255, 0, 0, 100))
        result = Image.alpha_composite(img, overlay)
        
        output_buffer = io.BytesIO()
        result.save(output_buffer, format="JPEG", quality=70)
        output_buffer.seek(0)

        return send_file(output_buffer, mimetype='image/jpeg')
    except Exception as e:
        return str(e), 500

# ---------------------------------------------------------
# 功能 3: 拍照停车检测
# ---------------------------------------------------------
@app.route('/api/test/check_parking', methods=['POST'])
def check_parking():
    try:
        if 'file' not in request.files:
            return jsonify({"code": 400, "message": "No file"}), 400
        
        file = request.files['file']
        img_bytes = file.read()
        
        is_valid_parking = random.random() > 0.2 
        
        result_data = {
            "is_valid": is_valid_parking,
            "confidence": round(random.uniform(0.8, 0.99), 2),
            "message": "规范停车" if is_valid_parking else "检测到车辆未在白线内，请重新停放"
        }

        if is_valid_parking:
            evidence_dir = os.path.join(UPLOAD_ROOT, "parking_success")
        else:
            evidence_dir = os.path.join(UPLOAD_ROOT, "parking_violation")
            
        os.makedirs(evidence_dir, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_evidence.jpg"
        
        with open(os.path.join(evidence_dir, filename), "wb") as f:
            f.write(img_bytes)

        return jsonify(result_data), 200

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"code": 500, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=11470, debug=True)