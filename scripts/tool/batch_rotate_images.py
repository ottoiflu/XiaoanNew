import os

from PIL import Image
from tqdm import tqdm


def batch_rotate(input_dir, output_dir, angle, expand=False):
    """
    批量旋转 input_dir 中的图片并保存到 output_dir。

    Args:
        input_dir (str): 包含源图片的目录。
        output_dir (str): 保存旋转后图片的目录。
        angle (float): 旋转角度。
        expand (bool): 是否扩展输出图片以包含旋转后的所有内容。
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        except OSError as e:
            print(f"创建目录 {output_dir} 时出错: {e}")
            return

    # 支持的图片扩展名
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 '{input_dir}' 不存在。")
        return

    files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in valid_extensions]

    if not files:
        print(f"在 {input_dir} 中未找到图片文件")
        return

    print(f"找到 {len(files)} 张图片。开始旋转 {angle} 度...")

    success_count = 0
    for filename in tqdm(files, desc="正在旋转"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                # 旋转图片
                # PIL 是逆时针旋转
                converted_img = img
                # 如果需要，可以更好地处理带有透明度或调色板的图片，但基本的 open 通常没问题。
                # 如果进行扩展，背景将默认为黑色/透明，具体取决于模式。

                rotated_img = converted_img.rotate(angle, expand=expand)

                # 保存旋转后的图片
                rotated_img.save(output_path)
                success_count += 1
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print(f"完成！成功处理了 {success_count}/{len(files)} 张图片。")


if __name__ == "__main__":
    # 在这里直接修改参数
    input_directory = r"/root/XiaoanNew/App_collected_dataset/zz03"  # 请修改为实际的输入图片路径
    output_directory = r"/root/XiaoanNew/App_collected_dataset/zz03_rotate"  # 请修改为实际的输出保存路径
    rotation_angle = 270.0  # 旋转角度（逆时针，单位：度）
    expand_canvas = True  # 是否扩展画布以包含完整图片（避免四角被裁剪），True为扩展，False为保持原尺寸

    batch_rotate(input_directory, output_directory, rotation_angle, expand_canvas)
