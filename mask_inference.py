import torch
import torchvision
import numpy as np
import io
import time
import random
from PIL import Image, ImageDraw, ImageFont
import base64
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2 # 需要用到OpenCV进行几何计算

class MaskRCNNInference:
    def __init__(self, weights_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ### 修改点 1：同步更新类别列表，增加 'curb'
        self.kinds = ['_background_', 'electric_bike', 'parking_lane', 'curb'] 
        
        self.box_conf = 0.5
        self.mask_conf = 0.5
        
        print(f"[AI Engine] Initializing model on {self.device}...")
        self.model = self._load_model(weights_path)
        print("[AI Engine] Model loaded successfully!")

    def _load_model(self, weights_path):
        # 1. 构建模型架构
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        
        # ### 修改点 2：类别数必须改为 4 (背景 + 3个目标)
        num_classes = 4  
        
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
        model.eval() 
        
        # 3. 热身 (Warm-up)
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
        
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = F.to_tensor(pil_img)
        
        predictions = self.model([img_tensor.to(self.device)])[0]
        scores = predictions['scores'].cpu()
        
        inds = torch.where(scores >= self.box_conf)[0]
        
        if len(inds) == 0:
            width, height = pil_img.size
            empty_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            output_buffer = io.BytesIO()
            empty_img.save(output_buffer, format="PNG")
            output_buffer.seek(0)
            return output_buffer

        boxes = predictions['boxes'].cpu()[inds]
        labels = predictions['labels'].cpu()[inds]
        masks = predictions['masks'].cpu()[inds]
        scores = scores[inds]

        masks = (masks > self.mask_conf).squeeze(1) 
        
        width, height = pil_img.size
        final_mask_array = np.zeros((height, width, 4), dtype=np.uint8)
        
        colors = [] 
        for i in range(len(inds)):
            color = self.random_color() 
            colors.append(color)
            m = masks[i].numpy()
            final_mask_array[m, 0] = color[0] 
            final_mask_array[m, 1] = color[1] 
            final_mask_array[m, 2] = color[2] 
            final_mask_array[m, 3] = 180      

        overlay_img = Image.fromarray(final_mask_array, mode="RGBA")
        draw = ImageDraw.Draw(overlay_img)
        
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()

        for i in range(len(inds)):
            box = boxes[i].tolist() 
            score = scores[i].item()
            label_idx = labels[i].item()
            label_name = self.kinds[label_idx] if label_idx < len(self.kinds) else str(label_idx)
            
            text = f"{label_name}: {score:.2f}"
            color = colors[i] 
            
            draw.rectangle(box, outline=color + (255,), width=2)
            draw.text((box[0], box[1]), text, fill=(255, 255, 255, 255), font=font)

        output_buffer = io.BytesIO()
        overlay_img.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        print(f"[AI Engine] Inference (Memory) completed in {time.time() - t0:.3f} seconds.")
        
        return output_buffer

    @torch.inference_mode()
    def predict_static_json(self, image_bytes):
        """
        静态分析专用：返回标准化的 JSON 数据
        Base64 掩膜图片中现在包含：半透明色块 + 实心边框 + 类别文字(置信度)
        """
        # 1. 图像预处理
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = F.to_tensor(pil_img).to(self.device)
        
        # 2. 推理
        predictions = self.model([img_tensor])[0]
        
        # 3. 数据过滤
        scores = predictions['scores'].cpu()
        inds = torch.where(scores >= self.box_conf)[0]
        
        detections = []
        width, height = pil_img.size
        # 创建透明底图 (RGBA)
        final_mask_array = np.zeros((height, width, 4), dtype=np.uint8)

        # 类别颜色定义
        COLOR_MAP = {
            1: (0, 255, 0),    # 车：绿色
            2: (255, 255, 0),  # 线：黄色
            3: (255, 0, 255)   # 马路牙子：紫色
        }

        if len(inds) > 0:
            boxes = predictions['boxes'].cpu()[inds]
            labels = predictions['labels'].cpu()[inds]
            masks = predictions['masks'].cpu()[inds]
            scores = scores[inds]

            masks = (masks > self.mask_conf).squeeze(1).numpy()

            # 先转换成 PIL Image 方便后续使用 ImageDraw 绘制文字和框
            # 我们先在 Numpy 里把色块涂好
            for i in range(len(inds)):
                label_idx = int(labels[i].item())
                color = COLOR_MAP.get(label_idx, (255, 255, 255))
                m = masks[i]
                final_mask_array[m, 0] = color[0]
                final_mask_array[m, 1] = color[1]
                final_mask_array[m, 2] = color[2]
                final_mask_array[m, 3] = 120  # 色块透明度设为 120，稍微浅一点好透视原图

            # 将 Numpy 转为 PIL 开始画框和写字
            overlay_img = Image.fromarray(final_mask_array, mode="RGBA")
            draw = ImageDraw.Draw(overlay_img)

            # 加载字体（如果没有指定字体，建议放在工程目录下，这里用默认兜底）
            try:
                # 针对 1080p 图片，字体大小设为 30-40 比较合适
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
            except:
                font = ImageFont.load_default()

            for i in range(len(inds)):
                label_idx = int(labels[i].item())
                label_name = self.kinds[label_idx] if label_idx < len(self.kinds) else "unknown"
                score_val = float(scores[i].item())
                box = boxes[i].tolist() # [x1, y1, x2, y2]
                color = COLOR_MAP.get(label_idx, (255, 255, 255))

                # --- A. 绘制方框 (Alpha 设为 255 不透明，更清晰) ---
                draw.rectangle(box, outline=color + (255,), width=4)

                # --- B. 准备文字 (Label + Score) ---
                display_text = f"{label_name} {score_val:.2f}"
                
                # 画一个文字背景小框，让字更清楚
                text_pos = (box[0] + 5, box[1] + 5)
                # 如果方框太靠顶部，把字往框内挪
                if box[1] < 50: text_pos = (box[0] + 5, box[1] + 5)
                
                # 绘制文字
                draw.text(text_pos, display_text, fill=(255, 255, 255, 255), font=font)

                # --- C. 存入 JSON 数据列表 ---
                detections.append({
                    "category_id": label_idx,
                    "label": label_name,
                    "score": round(score_val, 4),
                    "box": [round(c, 2) for c in box]
                })

            # 4. 生成最终 Base64
            buffer = io.BytesIO()
            overlay_img.save(buffer, format="PNG")
            mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            # 如果没识别到，返回空透明图的 Base64
            overlay_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            buffer = io.BytesIO()
            overlay_img.save(buffer, format="PNG")
            mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            "status": "success",
            "detections": detections, # 结构化数据，用于后端计算
            "mask_base64": mask_base64 # 增强后的可视化图片，用于前端叠加展示
        }


    

    def get_geometry_lines(self, image_bytes):
        """
        计算并绘制辅助线：
        绿色：车辆中轴线
        红色：停车线/马路牙子参考线
        """
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_tensor = F.to_tensor(pil_img).to(self.device)
        
        predictions = self.model([img_tensor])[0]
        scores = predictions['scores'].cpu().numpy()
        inds = np.where(scores >= self.box_conf)[0]
        
        if len(inds) == 0: return img_cv, False

        masks = predictions['masks'].cpu().numpy()[inds]
        labels = predictions['labels'].cpu().numpy()[inds]

        for i in range(len(inds)):
            m = (masks[i] > self.mask_conf).squeeze().astype(np.uint8)
            label = labels[i]
            
            # --- 寻找几何特征 ---
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            cnt = max(contours, key=cv2.contourArea)

            if label == 1: # electric_bike -> 画绿色轴线
                # 使用最小外接矩形找中轴
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                # 确定长边作为轴线
                p1, p2 = self._get_long_axis(box) 
                cv2.line(img_cv, p1, p2, (0, 255, 0), 5) # 绿色粗线
                cv2.putText(img_cv, "BIKE_AXIS", p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif label in [2, 3]: # 线或马路牙子 -> 画红色参考线
                # 拟合直线
                [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((img_cv.shape[1] - x) * vy / vx) + y)
                cv2.line(img_cv, (img_cv.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 5) # 红色粗线
                cv2.putText(img_cv, "BOUNDARY", (10, lefty - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return img_cv, True

    def _get_long_axis(self, box):
        # 辅助函数：根据矩形四个顶点返回长边的中心线起点和终点
        # ... (简单的欧几里得距离逻辑)
        return (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1]))