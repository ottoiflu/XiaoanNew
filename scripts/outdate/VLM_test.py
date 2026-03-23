import os
import base64
import csv
import re
from openai import OpenAI
from tqdm import tqdm
import time
from PIL import Image
import io # 导入io库，用于在内存中处理图片数据

BASE_URL = "https://api.ppinfra.com/openai"
API_KEY = "REDACTED_API_KEY_3"
MODEL = "qwen/qwen3-vl-235b-a22b-instruct"

# 文件夹和文件路径
IMAGE_DIR = r"/root/XiaoanNew/App_collected_dataset/test/2025-12-31"
OUTPUT_CSV = r"/root/XiaoanNew/App_collected_dataset/test/qwen3-vl-235b-a22b-instruct_vlm_test_results_compressed.csv"

# --- 新增：图片压缩配置 ---
MAX_IMAGE_SIZE = (512, 512) # 设置图片最大尺寸 (宽度, 高度)
JPEG_QUALITY = 60 # 设置JPEG图片的压缩质量 (1-95, 越高越清晰，文件越大)


# --- 优化后的 VLM 提示词 ---
PROMPT = """你是一位经验丰富的共享单车运营管理员。你的任务是审核用户还车照片，判断车辆停放是否“合格”。
请注意：我们要寻找的是“不可接受的违规”，对于轻微的瑕疵应保持宽容。

【核心判断逻辑】
请依次思考以下步骤，最后给出结论：

1. **识别环境**：是否有明显的停车框（白线）？如果是无框区域，是否靠路边或与其他车辆对齐？
2. **宽容判定边界**：
   - 合格：车身主体（脚撑、车轮接地点）在白线内。
   - 合格：车把手、车篮、车尾轻微超出白线（这是常见现象，允许）。
   - 合格：车轮压在白线上（只要没完全出去）。
   - 不合格：车身主体完全在框外，或车身横跨两个停车位导致他人无法停车。
3. **宽容判定角度**：
   - 合格：车辆方向基本垂直于路沿，或与周围其他车辆排列方向一致（随大流）。
   - 不合格：明显的斜停（如 45 度以上）导致占用过多空间，或完全逆向、横向阻断道路。
4. **一票否决项（严重违规）**：
   - 车辆倒地。
   - 停在盲道（黄色凸起纹路）上。
   - 停在机动车道或完全阻断人行道/消防通道。

【输出要求】
result: yes 或 no (yes 代表合格/可接受，no 代表严重违规/不可接受)
reason: 先描述车辆与线/路边的关系，再说明判定理由。
"""

def compress_and_encode_image(image_path):
    """
    压缩图片（如果需要），然后将其编码为Base64字符串。
    返回处理后的图片尺寸和Base64编码。
    """
    try:
        with Image.open(image_path) as img:
            # 保留原始尺寸信息
            original_width, original_height = img.size

            # 如果图片是RGBA模式（如某些PNG），转换为RGB以兼容JPEG保存
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # 等比例缩放图片
            img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # 将压缩后的图片保存在内存中
            in_mem_file = io.BytesIO()
            img.save(in_mem_file, format='JPEG', quality=JPEG_QUALITY)
            
            # 从内存中读取二进制数据并进行Base64编码
            in_mem_file.seek(0)
            base64_image = base64.b64encode(in_mem_file.read()).decode('utf-8')
            
            # 返回原始尺寸和编码后的字符串
            return (original_width, original_height), base64_image

    except IOError as e:
        print(f"错误：无法读取或处理图片文件 {image_path}: {e}")
        return (None, None), None


def parse_vlm_response(response_text):
    """
    使用正则表达式解析VLM的响应，更健壮地提取 'result' 和 'reason'。
    """
    response_text = response_text.strip()
    result_match = re.search(r"result:\s*(yes|no)", response_text, re.IGNORECASE)
    reason_match = re.search(r"reason:(.*)", response_text, re.IGNORECASE | re.DOTALL)
    
    result = result_match.group(1).lower() if result_match else 'unknown'
    
    if reason_match:
        reason = reason_match.group(1).strip()
    elif result != 'unknown':
        start_index = response_text.lower().find(result) + len(result)
        reason = response_text[start_index:].strip().lstrip(',').lstrip('，').strip()
    else:
        reason = response_text

    return result, reason

# --- 主程序 ---
def main():
    """主执行函数"""
    print("开始处理电动车停放规范性评估（带图片压缩）...")
    
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    except Exception as e:
        print(f"错误：无法初始化OpenAI客户端: {e}")
        return

    try:
        image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not image_files:
            print(f"警告：在目录 {IMAGE_DIR} 中未找到任何图片文件。")
            return
    except FileNotFoundError:
        print(f"错误：图片目录 {IMAGE_DIR} 不存在。")
        return

    print(f"在 {IMAGE_DIR} 中找到 {len(image_files)} 张图片。")
    print(f"评估结果将保存到 {OUTPUT_CSV}")

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_name', 'original_width', 'original_height', 'result', 'reason'])

        for image_name in tqdm(image_files, desc="评估进度"):
            image_path = os.path.join(IMAGE_DIR, image_name)
            
            # 使用新的函数进行压缩和编码
            (width, height), base64_image = compress_and_encode_image(image_path)
            
            if not base64_image:
                csv_writer.writerow([image_name, 'error', 'error', 'error', 'Could not process image'])
                continue

            try:
                # 提示中仍然可以使用原始尺寸，让模型了解原始比例
                image_size_text = f"请分析以下图片（原始尺寸为 {width}x{height} 像素），判断共享单车停放是否合规。"
                
                chat_completion_res = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "text", "text": image_size_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}" # 统一使用jpeg格式
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=500,
                )
                
                vlm_output = chat_completion_res.choices[0].message.content
                result, reason = parse_vlm_response(vlm_output)
                
                csv_writer.writerow([image_name, width, height, result, reason])
                csvfile.flush()

            except Exception as e:
                print(f"\n处理图片 {image_name} 时发生错误: {e}")
                csv_writer.writerow([image_name, width, height, 'error', str(e)])
                csvfile.flush()

            time.sleep(0.5)

    print("\n评估完成！结果已成功保存。")

if __name__ == "__main__":
    main()