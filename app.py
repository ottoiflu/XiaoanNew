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
# 功能 1: 数据采集 (带详细调试信息的修复版)
# ---------------------------------------------------------
@app.route('/api/collect/upload', methods=['POST'])
def collect_upload():
    try:
        # =========================================================
        # 🕵️‍♂️ [调试核心] 打印收到的所有表单参数
        # =========================================================
        print(f"\n--- [收到上传请求] ---")
        print(f"所有参数: {dict(request.form)}") # 打印 Android 传来的所有文本参数
        
        # 1. 基础参数校验
        if 'file' not in request.files:
            print("错误: 没有收到 file 参数")
            return jsonify({"status": "error", "message": "No file part"}), 400
            
        file = request.files['file']
        
        # --- 获取参数 ---
        label = request.form.get('label', 'unknown') 
        date_str = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        custom_path = request.form.get('custom_path', '').strip()
        
        # [修改] 获取并清理 ground_truth，防止空格干扰
        # 兼容性处理：把 None 转为空字符串，去空格，转小写
        raw_gt = request.form.get('ground_truth')
        ground_truth = str(raw_gt).strip().lower() if raw_gt else "no data"

        print(f"解析后的 ground_truth: '{ground_truth}' (原始值: {raw_gt})")

        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        # --- 2. 图片有效性与分辨率获取 ---
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)

        if file_length == 0:
            return jsonify({"status": "error", "message": "File is empty"}), 400
        
        try:
            img_info = Image.open(file)
            width, height = img_info.size
            resolution_tag = f"{width}x{height}"
            file.seek(0)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid image: {str(e)}"}), 400

        # --- 3. 构建保存路径 ---
        if custom_path:
            safe_path = custom_path.replace('../', '').replace('..\\', '')
            save_dir = os.path.join(UPLOAD_ROOT, safe_path)
            print(f"路径模式: 自定义 -> {save_dir}")
        else:
            save_dir = os.path.join(UPLOAD_ROOT, label, date_str)
            print(f"路径模式: 默认 -> {save_dir}")

        os.makedirs(save_dir, exist_ok=True)

        # --- 4. 保存图片 ---
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%H%M%S")
        final_name = f"{timestamp}_{resolution_tag}_{filename}"
        final_path = os.path.join(save_dir, final_name)
        
        file.save(final_path)
        print(f"图片已保存: {final_name}")

        # =========================================================
        # [修改] 写入 labels.txt 逻辑 (增强兼容性)
        # =========================================================
        # 只要 ground_truth 不为空，并且不是 "null" / "none"，我们就尝试记录
        # 这样即使你传了 "true" 或者其他值，也能看到 txt 生成
        if ground_truth and ground_truth not in ['null', 'none']:
            txt_path = os.path.join(save_dir, "labels.txt")
            
            # 格式: 图片文件名, 真值
            line_content = f"{final_name}, {ground_truth}\n"
            
            try:
                with open(txt_path, "a", encoding="utf-8") as f:
                    f.write(line_content)
                print(f"[真值记录成功] 写入: {line_content.strip()} 到 {txt_path}")
            except Exception as e:
                print(f" [真值记录失败] 无法写入 txt: {e}")
        else:
            print(f"[跳过真值记录] 原因: ground_truth 为空或无效 (收到: '{ground_truth}')")

        # --- 5. 返回结果 ---
        return jsonify({
            "status": "success", 
            "path": final_path,
            "resolution": resolution_tag,
            "ground_truth": ground_truth if ground_truth else "N/A"
        }), 200

    except Exception as e:
        print(f"Server Error: {e}")
        import traceback
        traceback.print_exc() # 打印完整报错堆栈
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