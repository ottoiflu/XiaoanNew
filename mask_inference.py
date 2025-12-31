import torch
import torchvision
import numpy as np
import io
import time
import random
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from PIL import Image, ImageDraw, ImageFont
import base64

class MaskRCNNInference:
    def __init__(self, weights_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kinds = ['background', 'electric_bike', 'parking_lane'] # 你的类别
        self.box_conf = 0.5
        self.mask_conf = 0.5
        
        print(f"[AI Engine] Initializing model on {self.device}...")
        self.model = self._load_model(weights_path)
        print("[AI Engine] Model loaded successfully!")

    def _load_model(self, weights_path):
        # 1. 构建模型架构
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        
        num_classes = 3  # 背景 + 2类
        # 替换 Box 头
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # 替换 Mask 头
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

        # 2. 加载权重
        checkpoint = torch.load(weights_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval() # 必须开启验证模式
        
        # 3. 热身 (Warm-up) 防止第一次推理卡顿
        print("Warming up...")
        dummy_img = torch.rand(1, 3, 640, 640).to(self.device)
        with torch.no_grad():
            model(dummy_img)
            
        return model

    def random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    @torch.inference_mode()
    def predict_memory(self, image_bytes):
        """
        全能版：返回透明 PNG，包含 掩膜 + 边框 + 置信度文字
        """
        t0 = time.time()
        
        # 1. 字节流 -> Tensor
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = F.to_tensor(pil_img)
        
        # 2. 推理
        predictions = self.model([img_tensor.to(self.device)])[0]
        scores = predictions['scores'].cpu()
        
        # 3. 过滤
        inds = torch.where(scores >= self.box_conf)[0]
        
        # 如果无目标，返回全透明图
        if len(inds) == 0:
            width, height = pil_img.size
            empty_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            output_buffer = io.BytesIO()
            empty_img.save(output_buffer, format="PNG")
            output_buffer.seek(0)
            return output_buffer

        # 获取数据
        boxes = predictions['boxes'].cpu()[inds]
        labels = predictions['labels'].cpu()[inds]
        masks = predictions['masks'].cpu()[inds]
        scores = scores[inds]

        # 4. 绘图准备
        masks = masks > self.mask_conf
        masks = masks.squeeze(1) # [N, H, W]
        
        width, height = pil_img.size
        
        # 创建 Numpy 数组用于快速填充 Mask 颜色
        final_mask_array = np.zeros((height, width, 4), dtype=np.uint8)
        
        # 5. 绘制 Mask (先在 Numpy 里填色)
        # 为了对应颜色，我们先存下每个目标的颜色
        colors = [] 
        for i in range(len(inds)):
            color = self.random_color() # (R, G, B)
            colors.append(color)
            
            m = masks[i].numpy()
            # 填充 Mask 颜色
            final_mask_array[m, 0] = color[0] 
            final_mask_array[m, 1] = color[1] 
            final_mask_array[m, 2] = color[2] 
            final_mask_array[m, 3] = 180      # Mask 透明度 (0-255)，180 比较适中

        # 6. 转回 PIL Image 以便绘制 框 和 文字
        # Numpy (Mask) -> PIL (RGBA)
        overlay_img = Image.fromarray(final_mask_array, mode="RGBA")
        draw = ImageDraw.Draw(overlay_img)
        
        # 尝试加载字体，如果失败用默认
        try:
            # Linux 服务器通常有这个字体，大小设为 15 (针对 320x240 图片够大了)
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()

        # 7. 绘制 边框 和 文字
        for i in range(len(inds)):
            # 获取坐标
            box = boxes[i].tolist() # [x1, y1, x2, y2]
            score = scores[i].item()
            label_idx = labels[i].item()
            label_name = self.kinds[label_idx] if label_idx < len(self.kinds) else str(label_idx)
            
            text = f"{label_name}: {score:.2f}"
            color = colors[i] # 使用和 Mask 相同的颜色
            
            # 画矩形框 (outline颜色, width线宽)
            draw.rectangle(box, outline=color + (255,), width=2)
            
            # 画文字背景 (可选，为了看清文字)
            # text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
            # draw.rectangle(text_bbox, fill=color + (255,))
            
            # 画文字
            # text_pos = (box[0], box[1] - 10) if box[1] > 10 else (box[0], box[1])
            draw.text((box[0], box[1]), text, fill=(255, 255, 255, 255), font=font)

        # 8. 输出
        output_buffer = io.BytesIO()
        overlay_img.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        print(f"[AI Engine] Inference completed in {time.time() - t0:.3f} seconds.")
        
        return output_buffer
    

    # ==========================================
    # [新增] 静态分析专用方法
    # ==========================================
    @torch.inference_mode()
    def predict_static_json(self, image_bytes):
        """
        静态分析专用：返回 JSON 数据 (含 Base64 掩膜 + 检测框数据)
        """
        # 1. 图像预处理
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = F.to_tensor(pil_img)
        
        # 2. 推理
        predictions = self.model([img_tensor.to(self.device)])[0]
        scores = predictions['scores'].cpu()
        
        # 3. 过滤
        inds = torch.where(scores >= self.box_conf)[0]
        
        detections = []
        
        # 准备透明掩膜画布
        width, height = pil_img.size
        final_mask_array = np.zeros((height, width, 4), dtype=np.uint8)

        if len(inds) > 0:
            boxes = predictions['boxes'].cpu()[inds]
            labels = predictions['labels'].cpu()[inds]
            masks = predictions['masks'].cpu()[inds]
            scores = scores[inds]

            masks = masks > self.mask_conf
            masks = masks.squeeze(1)

            for i in range(len(inds)):
                # 提取数据
                box = boxes[i].tolist() # [x1, y1, x2, y2]
                score = scores[i].item()
                label_idx = labels[i].item()
                label_name = self.kinds[label_idx] if label_idx < len(self.kinds) else str(label_idx)

                # 添加到 JSON 列表
                detections.append({
                    "label": label_name,
                    "score": score,
                    "box": box
                })

                # 绘制掩膜
                color = self.random_color()
                m = masks[i].numpy()
                final_mask_array[m, 0] = color[0]
                final_mask_array[m, 1] = color[1]
                final_mask_array[m, 2] = color[2]
                final_mask_array[m, 3] = 180 # 透明度

        # 5. 生成 Base64 掩膜图片
        overlay_img = Image.fromarray(final_mask_array, mode="RGBA")
        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG")
        mask_bytes = buffer.getvalue()
        # 转 Base64 字符串
        mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')

        # 6. 返回字典
        return {
            "detections": detections,
            "mask_base64": mask_base64,
            "image_width": width,
            "image_height": height
        }