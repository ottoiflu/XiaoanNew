import gradio as gr
import pandas as pd
import os
import shutil
from PIL import Image
import io

# ================= 配置区域 =================
DATA_DIR = r"/root/XiaoanNew/App_collected_dataset/Campus_val"
LABEL_FILE_NAME = "labels.txt"
SPLIT_BASE_DIR = r"/root/XiaoanNew/App_collected_dataset/Xiaoan_datasets"

if not os.path.exists(SPLIT_BASE_DIR):
    os.makedirs(SPLIT_BASE_DIR)

# ================= 核心逻辑函数 =================

def get_existing_split_folders():
    if not os.path.exists(SPLIT_BASE_DIR): return []
    folders = [f for f in os.listdir(SPLIT_BASE_DIR) if os.path.isdir(os.path.join(SPLIT_BASE_DIR, f))]
    defaults = ["train", "val", "test", "trash", "yes_val", "no_val"]
    return sorted(list(set(folders + defaults)))

def save_all_labels(df):
    label_path = os.path.join(DATA_DIR, LABEL_FILE_NAME)
    try:
        with open(label_path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(f"{row['image_name']},{row['ground_truth']}\n")
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False

def load_all_samples():
    label_path = os.path.join(DATA_DIR, LABEL_FILE_NAME)
    if not os.path.exists(label_path):
        return pd.DataFrame(columns=["image_name", "ground_truth"]), "未找到标签文件"
    
    samples = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "," not in line: continue
                parts = line.split(",", 1)
                samples.append({"image_name": parts[0].strip(), "ground_truth": parts[1].strip().lower()})
        return pd.DataFrame(samples), f"成功加载 {len(samples)} 条数据"
    except Exception as e:
        return pd.DataFrame(columns=["image_name", "ground_truth"]), f"加载失败: {e}"

global_df, init_msg = load_all_samples()

# ================= 交互功能 =================

def get_ui_update(index):
    total = len(global_df)
    if total == 0:
        return 0, None, "数据列表为空", "N/A", "进度: 0 / 0", 0
    
    # 循环索引处理
    if index >= total: index = 0
    if index < 0: index = total - 1
    
    row = global_df.iloc[index]
    img_name = row['image_name']
    img_path = os.path.join(DATA_DIR, img_name)
    gt = row['ground_truth']
    
    img_display = None
    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                img.thumbnail((600, 600), Image.Resampling.NEAREST)
                img_display = img.convert("RGB")
        except:
            img_display = None
    
    progress = f"进度: {index + 1} / {total}"
    return index, img_display, img_name, gt, progress, index + 1

def handle_search(query):
    if not query:
        return get_ui_update(0)
    matches = global_df[global_df['image_name'].str.contains(query, case=False, na=False)]
    if not matches.empty:
        target_idx = matches.index[0]
        return get_ui_update(target_idx)
    else:
        gr.Warning(f"未找到包含 '{query}' 的文件")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

def update_gt_handler(index, new_gt):
    global global_df
    if len(global_df) > 0:
        global_df.at[index, 'ground_truth'] = new_gt.strip().lower()
        save_all_labels(global_df)
        gr.Info(f"真值已修改并保存: {new_gt}")
    return get_ui_update(index)

def copy_data_handler(index, folder_name):
    """【修改后的核心逻辑：复制图片而非移动】"""
    global global_df
    if len(global_df) == 0:
        return (*get_ui_update(0), gr.update())
    
    if not folder_name:
        gr.Warning("请先选择目标文件夹")
        return (*get_ui_update(index), gr.update())

    row = global_df.iloc[index]
    img_name = row['image_name']
    gt_val = row['ground_truth']
    
    src_img_path = os.path.join(DATA_DIR, img_name)
    dest_dir = os.path.join(SPLIT_BASE_DIR, folder_name.strip())
    dest_img_path = os.path.join(dest_dir, img_name)
    dest_label_path = os.path.join(dest_dir, LABEL_FILE_NAME)

    try:
        if not os.path.exists(dest_dir): 
            os.makedirs(dest_dir)
        
        # 1. 核心修改：使用 copy2 替代 move
        if os.path.exists(src_img_path): 
            shutil.copy2(src_img_path, dest_img_path)
        
        # 2. 将标签写入目标目录的 labels.txt
        with open(dest_label_path, "a", encoding="utf-8") as f:
            f.write(f"{img_name},{gt_val}\n")
            
        gr.Info(f"已复制到 {folder_name}")
        
        # 3. 自动跳转到下一张 (索引+1)，但不删除源列表中的数据
        return (*get_ui_update(index + 1), gr.update(choices=get_existing_split_folders()))
    except Exception as e:
        gr.Error(f"操作失败: {e}")
        return (*get_ui_update(index), gr.update())

# ================= UI 界面 =================

with gr.Blocks(title="Xiaoan 数据分拣系统 (复制模式)") as demo:
    current_index = gr.State(0)
    
    gr.Markdown(f"# Xiaoan 数据分拣系统 (复制模式)")
    gr.Markdown(f"源目录: `{DATA_DIR}` | 将图片**复制**到目标目录，不删除源文件。")
    
    with gr.Row():
        with gr.Column(scale=4):
            image_display = gr.Image(type="pil", label="样本预览 (缩略图模式)", height=600)
        
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 样本定位")
                with gr.Row():
                    search_box = gr.Textbox(label="搜索文件名", placeholder="输入关键词回车...", scale=4)
                    search_btn = gr.Button("查找", scale=1)
                
            with gr.Group():
                status_progress = gr.Label(label="当前状态", value="正在加载...")
                filename_display = gr.Textbox(label="文件名", interactive=False)
                gt_display = gr.Textbox(label="当前样本真值")
            
            with gr.Row():
                btn_yes = gr.Button("修改 GT 为 YES", variant="secondary")
                btn_no = gr.Button("修改 GT 为 NO", variant="secondary")
            
            gr.Markdown("### 执行复制 (点击后跳转下一张)")
            with gr.Row():
                btn_fast_yes = gr.Button("yes_val", variant="primary")
                btn_fast_no = gr.Button("no_val", variant="primary")
                btn_fast_trash = gr.Button("trash", variant="stop")
            
            gr.Markdown("---")
            with gr.Accordion("自定义复制目录", open=False):
                folder_selector = gr.Dropdown(
                    choices=get_existing_split_folders(), 
                    label="选择或输入文件夹", 
                    allow_custom_value=True
                )
                btn_transfer = gr.Button("确认复制")
            
            gr.Markdown("---")
            with gr.Row():
                btn_prev = gr.Button("◀ 上一个")
                btn_next = gr.Button("下一个 ▶")
            
            jump_input = gr.Number(label="跳转索引", value=1, precision=0)
            btn_jump = gr.Button("跳转")

    ui_outputs = [current_index, image_display, filename_display, gt_display, status_progress, jump_input]

    # --- 事件绑定 ---
    search_box.submit(handle_search, [search_box], ui_outputs)
    search_btn.click(handle_search, [search_box], ui_outputs)

    # 快捷复制按钮 (使用修改后的逻辑)
    btn_fast_yes.click(lambda idx: copy_data_handler(idx, "yes_val"), [current_index], ui_outputs + [folder_selector])
    btn_fast_no.click(lambda idx: copy_data_handler(idx, "no_val"), [current_index], ui_outputs + [folder_selector])
    btn_fast_trash.click(lambda idx: copy_data_handler(idx, "trash"), [current_index], ui_outputs + [folder_selector])

    # 自定义复制
    btn_transfer.click(copy_data_handler, [current_index, folder_selector], ui_outputs + [folder_selector])

    # 基础导航
    btn_next.click(lambda idx: get_ui_update(idx + 1), inputs=[current_index], outputs=ui_outputs)
    btn_prev.click(lambda idx: get_ui_update(idx - 1), inputs=[current_index], outputs=ui_outputs)
    btn_jump.click(lambda val: get_ui_update(int(val) - 1), inputs=[jump_input], outputs=ui_outputs)

    # 真值修改 (仅修改源目录的 labels.txt，不复制)
    btn_yes.click(lambda idx: update_gt_handler(idx, "yes"), [current_index], ui_outputs)
    btn_no.click(lambda idx: update_gt_handler(idx, "no"), [current_index], ui_outputs)

    demo.load(lambda: get_ui_update(0), outputs=ui_outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)