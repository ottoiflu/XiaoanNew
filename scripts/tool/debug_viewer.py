import os

import gradio as gr
import pandas as pd
from PIL import Image

# ================= 配置区域 =================
BASE_DATA_DIR = r"/root/XiaoanNew/App_collected_dataset/Xiaoan_datasets"
CSV_PATH = r"/root/XiaoanNew/experiment_outputs/qwen3-vl-30b-a3b-instruct_yolov8seg_cv_enhanced_p3_test.csv"
MASK_DIR = r"/root/XiaoanNew/test_outputs/seg_visuals"

# ================= 1. 路径预扫描索引 =================
print("正在扫描图片路径并建立索引...")
path_index = {}
for root, dirs, files in os.walk(BASE_DATA_DIR):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            parent_folder = os.path.basename(root)
            path_index[(f, parent_folder)] = os.path.join(root, f)
print(f"索引完毕，共 {len(path_index)} 张图片。")

# ================= 2. 数据处理逻辑 (增强健壮性) =================


def load_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(), pd.DataFrame(), f"错误：未找到CSV文件 {CSV_PATH}"

    try:
        # 使用 utf-8-sig 自动处理 Excel 生成的 BOM 字符
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", engine="python")

        # 【关键修复】：强制清理所有列名的空格并转为小写
        # 这样不管 CSV 里写的是 "gt ", " GT", 还是 "gt" 都能匹配
        df.columns = [str(c).strip().lower() for c in df.columns]

        # 检查必要的列是否存在
        required_cols = ["image", "folder", "pred", "gt"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                f"CSV 缺少必要的列: {missing}\n当前识别到的列名为: {list(df.columns)}",
            )

        def normalize(s):
            return str(s).strip().lower() if pd.notna(s) else ""

        df["norm_gt"] = df["gt"].apply(normalize)
        df["norm_pred"] = df["pred"].apply(normalize)

        def get_status(row):
            if row["norm_gt"] == row["norm_pred"]:
                return "Correct (预测正确)"
            if "yes" in row["norm_gt"] and "no" in row["norm_pred"]:
                return "FN (漏报 - 实际合规但判违规)"
            if "no" in row["norm_gt"] and "yes" in row["norm_pred"]:
                return "FP (误报 - 实际违规但判合规)"
            return "Mismatch"

        df["status_type"] = df.apply(get_status, axis=1)

        # 筛选预测错误的样本
        error_df = df[df["norm_gt"] != df["norm_pred"]].copy()

        return df, error_df, f"加载成功！总计: {len(df)} 条 | 预测错误: {len(error_df)} 条"

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"读取CSV失败: {str(e)}"


full_df, error_df, init_msg = load_data()
# 优先显示错误样本
display_df = error_df if not error_df.empty else full_df

# ================= 3. 业务工具函数 =================


def get_details(df, index):
    if df.empty or index < 0 or index >= len(df):
        return None, None, "无数据", "None", "", "", "", "进度: 0/0"

    row = df.iloc[index]
    # 使用清理过后的列名
    image_name = str(row["image"])
    folder_name = str(row["folder"])

    image_path = path_index.get((image_name, folder_name))
    mask_path = os.path.join(MASK_DIR, image_name)

    def load_img(p):
        if p and os.path.exists(p):
            try:
                with Image.open(p) as img:
                    img.thumbnail((800, 800))
                    return img.copy()
            except Exception:
                return None
        return None

    img_display = load_img(image_path)
    mask_display = load_img(mask_path)

    return (
        img_display,
        mask_display,
        f"文件名: {image_name} | 目录: {folder_name}",
        row["status_type"],
        row["gt"],
        row["pred"],
        row.get("reason", "无理由"),
        f"进度: {index + 1} / {len(df)}",
    )


# ================= 4. Gradio UI =================

css = ".error-box textarea {font-weight:bold; color:#d32f2f; font-size:16px;}"

with gr.Blocks(css=css, title="VLM 结果分析工具") as demo:
    current_index = gr.State(0)

    gr.Markdown(f"## VLM + CV 联合测试结果分析\n{init_msg}")

    with gr.Row():
        with gr.Column(scale=6):
            with gr.Row():
                image_display = gr.Image(type="pil", label="原始图片")
                mask_display = gr.Image(type="pil", label="CV 分割掩膜图 (YOLOv8-Seg)")

            with gr.Row():
                btn_prev = gr.Button("上一个 (Prev)", variant="secondary")
                btn_next = gr.Button("下一个 (Next) ", variant="primary")

        with gr.Column(scale=3):
            with gr.Row():
                search_input = gr.Textbox(label="搜索文件名", placeholder="输入图片名回车...", scale=4)
                search_btn = gr.Button("查找", scale=1)

            image_info = gr.Textbox(label="数据来源", interactive=False)
            status_text = gr.Textbox(label="样本进度", interactive=False)
            error_type_display = gr.Textbox(label="判定状态", interactive=False, elem_classes="error-box")

            with gr.Row():
                gt_display = gr.Textbox(label="Ground Truth (真值)", interactive=False)
                pred_display = gr.Textbox(label="VLM Prediction (预测)", interactive=False)

            reason_display = gr.TextArea(label="VLM 判定推理过程", interactive=False, lines=15)

    output_components = [
        current_index,
        image_display,
        mask_display,
        image_info,
        error_type_display,
        gt_display,
        pred_display,
        reason_display,
        status_text,
    ]

    def handle_search(q):
        if not q:
            return 0, *get_details(display_df, 0)
        match = display_df[display_df["image"].astype(str).str.contains(q, case=False, na=False)]
        if not match.empty:
            new_idx = display_df.index.get_loc(match.index[0])
            return new_idx, *get_details(display_df, new_idx)
        return 0, *get_details(display_df, 0)

    def navigate(idx, step):
        if display_df.empty:
            return 0, *get_details(display_df, 0)
        new_idx = (idx + step) % len(display_df)
        return new_idx, *get_details(display_df, new_idx)

    # 绑定事件
    search_input.submit(handle_search, [search_input], output_components)
    search_btn.click(handle_search, [search_input], output_components)
    btn_next.click(lambda i: navigate(i, 1), [current_index], output_components)
    btn_prev.click(lambda i: navigate(i, -1), [current_index], output_components)

    # 页面加载时自动运行一次
    demo.load(lambda: (0, *get_details(display_df, 0)), outputs=output_components)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
