import os

import gradio as gr
import pandas as pd
from PIL import Image

# ================= 配置区域 =================
DATA_DIR = r"/root/XiaoanNew/MMLab/output/1"  # 改成你的图片文件夹

# 支持的图片格式
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ================= 数据加载 =================


def load_all_images():
    if not os.path.exists(DATA_DIR):
        return pd.DataFrame(columns=["image_name"]), "图片目录不存在"

    images = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(IMG_EXTS)])

    df = pd.DataFrame({"image_name": images})
    return df, f"成功加载 {len(df)} 张图片"


global_df, _ = load_all_images()

# ================= UI 刷新逻辑 =================


def get_ui_update(index):
    total = len(global_df)

    if total == 0:
        return 0, None, "无图片", "进度: 0 / 0"

    # 循环索引
    index = index % total

    img_name = global_df.iloc[index]["image_name"]
    img_path = os.path.join(DATA_DIR, img_name)

    img_display = None
    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                img.thumbnail((512, 512))
                img_display = img.convert("RGB")
        except Exception:
            img_display = None

    progress = f"进度: {index + 1} / {total}"

    return index, img_display, img_name, progress


# ================= Gradio UI =================

with gr.Blocks(title="图片文件夹浏览器", theme=gr.themes.Soft()) as demo:
    current_index = gr.State(0)

    with gr.Row():
        with gr.Column(scale=3):
            image_display = gr.Image(type="pil", label="图片预览", height=500)

            with gr.Row():
                btn_prev = gr.Button("◀ 上一张")
                btn_next = gr.Button("下一张 ▶")

        with gr.Column(scale=1):
            filename_display = gr.Textbox(label="文件名", interactive=False)
            status_progress = gr.Label(value="准备就绪")

    ui_outputs = [current_index, image_display, filename_display, status_progress]

    # 翻页逻辑
    btn_next.click(lambda idx: get_ui_update(idx + 1), inputs=[current_index], outputs=ui_outputs)

    btn_prev.click(lambda idx: get_ui_update(idx - 1), inputs=[current_index], outputs=ui_outputs)

    # 初始化
    demo.load(lambda: get_ui_update(0), outputs=ui_outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
