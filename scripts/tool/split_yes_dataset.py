import os
import random
import shutil

# ================= 配置区域 =================
src_folder = "/root/XiaoanNew/App_collected_dataset/Xiaoan_datasets/yes_val"
dest_folder = "/root/XiaoanNew/App_collected_dataset/Xiaoan_datasets/yes_reserve_val"
label_file_name = "labels.txt"
num_to_sample = 400

# ================= 脚本逻辑 =================


def split_dataset():
    # 1. 准备路径
    src_label_path = os.path.join(src_folder, label_file_name)
    dest_label_path = os.path.join(dest_folder, label_file_name)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"创建目标文件夹: {dest_folder}")

    # 2. 获取源文件夹中所有的图片文件
    image_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    all_images = [f for f in os.listdir(src_folder) if f.endswith(image_extensions)]

    if len(all_images) < num_to_sample:
        print(f"警告: 源文件夹中只有 {len(all_images)} 张图片，不足 {num_to_sample} 张。将转移所有图片。")
        sample_count = len(all_images)
    else:
        sample_count = num_to_sample

    # 3. 随机抽取
    sampled_images = random.sample(all_images, sample_count)
    sampled_images_set = set(sampled_images)  # 转为set提高查找效率

    print(f"开始移动 {sample_count} 张图片...")

    # 4. 移动图片文件
    for img_name in sampled_images:
        src_path = os.path.join(src_folder, img_name)
        dest_path = os.path.join(dest_folder, img_name)
        shutil.move(src_path, dest_path)

    # 5. 处理 labels.txt
    if os.path.exists(src_label_path):
        remaining_labels = []
        moved_labels = []

        with open(src_label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 假设格式是 filename,label 或类似的，取逗号前的文件名
                # 如果你的格式不同，请调整这里的解析逻辑
                file_in_label = line.split(",")[0].strip()

                if file_in_label in sampled_images_set:
                    moved_labels.append(line)
                else:
                    remaining_labels.append(line)

        # 写回原来的 labels.txt (保存剩下的)
        with open(src_label_path, "w", encoding="utf-8") as f:
            for line in remaining_labels:
                f.write(line + "\n")

        # 写入新的 labels.txt (保存移动的)
        # 使用 'a' 追加模式，防止目标文件夹已有内容
        with open(dest_label_path, "a", encoding="utf-8") as f:
            for line in moved_labels:
                f.write(line + "\n")

        print(f"标签处理完成：{len(moved_labels)} 条标签已迁移。")
    else:
        print("未找到 labels.txt 文件，跳过标签迁移。")

    print("任务全部完成！")


if __name__ == "__main__":
    split_dataset()
