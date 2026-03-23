import gradio as gr
import pandas as pd
import os
import shutil
from PIL import Image

# ================= 配置区域 =================
# 默认扫描的基础目录
BASE_PATH = r"/root/XiaoanNew/App_collected_dataset/Xiaoan_datasets"
# 默认备选的待分拣目录名（会自动合并 BASE_PATH 下已有的目录）
DATA_FOLDERS = ["yes_val", "no_val", "train", "val", "test"]

MASK_DIR = r"/root/XiaoanNew/test_outputs/seg_visuals" 
LABEL_FILE_NAME = "labels.txt"
SPLIT_BASE_DIR = BASE_PATH # 分拣目标基准目录

# 自动获取所有可能的待分拣目录
def get_available_sources():
    try:
        existing = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]
        return sorted(list(set(existing + DATA_FOLDERS)))
    except:
        return DATA_FOLDERS

if not os.path.exists(SPLIT_BASE_DIR):
    os.makedirs(SPLIT_BASE_DIR)

# ================= 核心逻辑 =================

def load_samples_from_dir(target_dir_name):
    """加载选定目录的样本"""
    full_path = os.path.join(BASE_PATH, target_dir_name)
    label_path = os.path.join(full_path, LABEL_FILE_NAME)
    
    if not os.path.exists(label_path):
        # 如果不存在标签文件，尝试扫描文件夹下的图片生成临时列表
        imgs = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        samples = [{"image_name": f, "ground_truth": "n/a"} for f in imgs]
        return pd.DataFrame(samples), f"未找到标签文件，已扫描到 {len(samples)} 张图片", full_path
    
    samples = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    samples.append({"image_name": parts[0].strip(), "ground_truth": parts[1].strip().lower()})
        return pd.DataFrame(samples), f"成功加载 {len(samples)} 条数据", full_path
    except Exception as e:
        return pd.DataFrame(columns=["image_name", "ground_truth"]), f"加载失败: {e}", full_path

def fast_save_labels(df, current_data_dir):
    """保存当前目录的标签"""
    label_path = os.path.join(current_data_dir, LABEL_FILE_NAME)
    content = "\n".join([f"{r['image_name']},{r['ground_truth']}" for _, r in df.iterrows()])
    with open(label_path, "w", encoding="utf-8") as f:
        f.write(content + ("\n" if content else ""))

# ================= UI 更新逻辑 =================

def get_ui_update(index, df, current_data_dir):
    total = len(df)
    if total == 0:
        return 0, None, None, "该文件夹无数据", "N/A", "进度: 0 / 0"
    
    # 循环索引处理
    if index >= total: index = 0 
    if index < 0: index = total - 1
    
    row = df.iloc[index]
    img_name = row['image_name']
    gt = row['ground_truth']
    
    img_path = os.path.join(current_data_dir, img_name)
    mask_path = os.path.join(MASK_DIR, img_name)
    
    img_display = None
    mask_display = None

    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                img_display = img.convert("RGB")
        except: img_display = None

    if os.path.exists(mask_path):
        try:
            with Image.open(mask_path) as msk:
                msk.thumbnail((512, 512), Image.Resampling.LANCZOS)
                mask_display = msk.convert("RGB")
        except: mask_display = None
    
    return index, img_display, mask_display, img_name, gt, f"进度: {index + 1} / {total}"

# ================= 搜索逻辑 =================

def search_and_jump(search_str, df, current_data_dir):
    if not search_str or df.empty:
        return get_ui_update(0, df, current_data_dir)
    
    # 尝试精确匹配或包含匹配
    matches = df[df['image_name'].str.contains(search_str, case=False, na=False)]
    if not matches.empty:
        new_index = matches.index[0]
        gr.Info(f"已跳转至匹配项: {df.iloc[new_index]['image_name']}")
        return get_ui_update(new_index, df, current_data_dir)
    else:
        gr.Warning("未找到匹配的文件名")
        return get_ui_update(0, df, current_data_dir)

# ================= 操作逻辑 =================

def process_action(index, df, current_data_dir, folder_name, mode="move"):
    if df.empty: 
        return (*get_ui_update(0, df, current_data_dir), df)
    
    if not folder_name:
        gr.Warning("请先选择目标文件夹")
        return (*get_ui_update(index, df, current_data_dir), df)

    row = df.iloc[index]
    img_name = row['image_name']
    gt_val = row['ground_truth']
    
    src_img_path = os.path.join(current_data_dir, img_name)
    src_mask_path = os.path.join(MASK_DIR, img_name)
    dest_dir = os.path.join(SPLIT_BASE_DIR, folder_name.strip())
    
    try:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        # 1. 文件操作
        target_path = os.path.join(dest_dir, img_name)
        if os.path.exists(src_img_path):
            if mode == "move": shutil.move(src_img_path, target_path)
            else: shutil.copy2(src_img_path, target_path)

        if os.path.exists(src_mask_path):
            mask_target_path = os.path.join(dest_dir, "mask_" + img_name)
            if mode == "move": shutil.move(src_mask_path, mask_target_path)
            else: shutil.copy2(src_mask_path, mask_target_path)

        # 2. 标签与内存更新
        if mode == "move":
            df = df.drop(df.index[index]).reset_index(drop=True)
            fast_save_labels(df, current_data_dir)
            msg = f"已移动至 {folder_name}"
            next_idx = index # 移动后，当前索引即为下一张
        else:
            msg = f"已复制至 {folder_name}"
            next_idx = index + 1
        
        # 3. 追加目标目录标签
        with open(os.path.join(dest_dir, LABEL_FILE_NAME), "a", encoding="utf-8") as f:
            f.write(f"{img_name},{gt_val}\n")
            
        gr.Info(msg)
        return (*get_ui_update(next_idx, df, current_data_dir), df)
    except Exception as e:
        gr.Error(f"操作失败: {e}")
        return (*get_ui_update(index, df, current_data_dir), df)

# ================= Gradio UI =================

with gr.Blocks(title="Xiaoan 数据分拣系统 Pro", theme=gr.themes.Soft()) as demo:
    # 状态变量
    current_index = gr.State(0)
    global_df = gr.State(pd.DataFrame())
    active_data_dir = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            source_selector = gr.Dropdown(
                choices=get_available_sources(), 
                label="选择当前数据源", 
                value=get_available_sources()[0] if get_available_sources() else None
            )
        with gr.Column(scale=2):
            search_box = gr.Textbox(label="搜索并跳转文件名", placeholder="输入文件名关键字...")
        with gr.Column(scale=1):
            btn_search = gr.Button("跳转", variant="secondary")

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                image_display = gr.Image(type="pil", label="原始图片", height=450)
                mask_display = gr.Image(type="pil", label="YOLO掩膜图", height=450)
            
            with gr.Row():
                btn_prev = gr.Button(" 上一个 ")
                btn_next = gr.Button("下一个 ")
        
        with gr.Column(scale=1):
            status_progress = gr.Label(value="请选择数据源")
            filename_display = gr.Textbox(label="文件名", interactive=False)
            gt_display = gr.Textbox(label="当前真值")
            
            with gr.Row():
                btn_yes = gr.Button("标记 YES")
                btn_no = gr.Button("标记 NO")
            
            gr.Markdown("---")
            gr.Markdown("### 快速分拣 (移动)")
            with gr.Row():
                btn_fast_yes = gr.Button("yes_val", variant="primary")
                btn_fast_no = gr.Button("no_val", variant="primary")
            btn_trash = gr.Button("trash", variant="stop")
            
            with gr.Accordion("自定义路径", open=True):
                folder_selector = gr.Dropdown(
                    choices=get_available_sources(), 
                    label="目标文件夹", 
                    allow_custom_value=True
                )
                with gr.Row():
                    btn_custom_move = gr.Button("移动")
                    btn_custom_copy = gr.Button("复制")

    # 定义统一的 UI 输出刷新组
    ui_outputs = [current_index, image_display, mask_display, filename_display, gt_display, status_progress]

    # --- 数据源切换 ---
    def on_source_change(folder_name):
        df, msg, path = load_samples_from_dir(folder_name)
        gr.Info(msg)
        # 获取第一张图的 UI 更新
        idx_res = get_ui_update(0, df, path)
        return [*idx_res, df, path]

    source_selector.change(on_source_change, [source_selector], ui_outputs + [global_df, active_data_dir])

    # --- 搜索跳转 ---
    search_box.submit(search_and_jump, [search_box, global_df, active_data_dir], ui_outputs)
    btn_search.click(search_and_jump, [search_box, global_df, active_data_dir], ui_outputs)

    # --- 基础导航 ---
    btn_next.click(lambda idx, df, d: get_ui_update(idx + 1, df, d), [current_index, global_df, active_data_dir], ui_outputs)
    btn_prev.click(lambda idx, df, d: get_ui_update(idx - 1, df, d), [current_index, global_df, active_data_dir], ui_outputs)

    # --- 快捷动作 ---
    btn_fast_yes.click(lambda idx, df, d: process_action(idx, df, d, "yes_val", "move"), 
                       [current_index, global_df, active_data_dir], ui_outputs + [global_df])
    
    btn_fast_no.click(lambda idx, df, d: process_action(idx, df, d, "no_val", "move"), 
                      [current_index, global_df, active_data_dir], ui_outputs + [global_df])
    
    btn_trash.click(lambda idx, df, d: process_action(idx, df, d, "trash", "move"), 
                    [current_index, global_df, active_data_dir], ui_outputs + [global_df])

    # --- 自定义动作 ---
    btn_custom_move.click(lambda idx, df, d, f: process_action(idx, df, d, f, "move"), 
                          [current_index, global_df, active_data_dir, folder_selector], ui_outputs + [global_df])
    
    btn_custom_copy.click(lambda idx, df, d, f: process_action(idx, df, d, f, "copy"), 
                          [current_index, global_df, active_data_dir, folder_selector], ui_outputs + [global_df])

    # --- 修改真值 ---
    def update_gt(idx, df, d, val):
        if not df.empty:
            df.at[idx, 'ground_truth'] = val
            fast_save_labels(df, d)
        return get_ui_update(idx, df, d)

    btn_yes.click(lambda idx, df, d: update_gt(idx, df, d, "yes"), [current_index, global_df, active_data_dir], ui_outputs)
    btn_no.click(lambda idx, df, d: update_gt(idx, df, d, "no"), [current_index, global_df, active_data_dir], ui_outputs)

    # 初始化启动
    demo.load(on_source_change, [source_selector], ui_outputs + [global_df, active_data_dir])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862)