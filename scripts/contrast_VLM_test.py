import os
import base64
import csv
import re
import io
import json
import time
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompt_manager import load_prompt
from utils.scoring import ScoringEngine

# ================= 实验矩阵配置 =================

# 1. 待处理的文件夹列表 (脚本会将它们合并视为“一次实验”)
DATA_FOLDERS = [
    r"/root/XiaoanNew/Compliance_test_data/no_val",
    r"/root/XiaoanNew/Compliance_test_data/yes_val",
]

# 2. 输出结果保存路径
SAVE_DIR = "/root/XiaoanNew/experiment_outputs"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 3. 实验配置 (此处仅设置一个配置，确保输出单一结果)
CONFIG = {
    "exp_name": "v1_standard_p6_weighted",
    "model": "qwen/qwen3-vl-30b-a3b-instruct",
    "max_size": (768, 768),
    "quality": 80,
    "prompt_id": "standard_p6",
    "scoring_config": "configs/scoring_default.yaml",  # 设为 None 使用一票否决
}

# 初始化评判引擎
_scoring_engine = None
if CONFIG.get("scoring_config"):
    _scoring_engine = ScoringEngine.from_yaml(CONFIG["scoring_config"])
    print(f">>> 加权评判引擎已加载: {CONFIG['scoring_config']} (阈值={_scoring_engine.config.threshold})")
else:
    print(">>> 使用一票否决评判模式")

# 4. 提示词库
PROMPT_LIB = {
    "cv_enhanced_p3_compare": """
# Role
你是一位专业的共享单车运维质检员，负责通过照片判定车辆停放是否符合城市管理严苛标准。


# Task
锁定画面中【最显著】的一辆共享单车/电动车为主体，执行以下判定流程。

# Guidelines & Criteria

## 1. 图像构图合规性 (Image Quality & Composition)
- **状态选项**：
  - `[合规]`：画面清晰，车辆大部分在框内，有明确参照物。
  - `[基本合规]`：关键部位虽部分截断，但仍可判定空间关系。
  - `[不合规-构图]`：过暗/过曝、拍摄距离过远。
  - `[不合规-无参照]`：AI 未检出且原图也无法识别任何物理边界线。

## 2. 摆放角度合规性 (Angle Compliance)
- **基准线**：停车线长边或马路牙子边缘。
- **向量**：后轮中点 -> 把手中心。
- **准则**：
  - 必须与基准线保持【垂直】，偏差不得超过 ±30°（即夹角 > 60°）。
  - 若车辆贴边顺向摆放（平行），直接判定为不合规。
- **状态选项**：`[合规]`、`[不合规-角度]`。

## 3. 停放距离合规性 (Distance Compliance)
- **判定基准**：以车座中心点与后轮中点的【连线中点】为界限。
- **状态选项**：
  - `[完全合规]`：车辆主体完全在线内。
  - `[基本合规-压线]`：界限中点处于停车线/边界**内侧**，仅后轮局部压线或向**后轮方向**超出。
  - `[不合规-超界]`：界限中点已跨越边界线向**车头/车座方向**偏移（即车身 1/2 以上出线）。

## 4. 路面环境合规性 (Contextual Compliance)
- **核心判定：盲道占用**。
- **逻辑优先级**：
  - **P1 (看接地点)**：若可见轮迹，任一车轮接地点完全压在盲道纹路上 → `[不合规-环境]`。
  - **P2 (看投影)**：若接地点被遮挡，车座投影面积 > 50% 覆盖盲道纹路 → `[不合规-环境]`。
- **状态选项**：`[合规]`、`[不合规-环境]`。

# 极严输出约束 (Mandatory Constraints)
1. **标签匹配**：status 字段必须严格从给定选项中精确选择，严禁自创。
2. **纯净输出**：status 字段内严禁出现任何解释性文字或括号内容。
3. **逻辑一致**：`step_by_step_analysis` 的描述必须支持最后的 `scores` 结论。

# Output Format (JSON)
必须严格按以下格式回答：
```json
{
  "step_by_step_analysis": {
    "ai_detection_summary": "总结 AI 检测到的对象（电动车、停车线、马路牙子、盲道的数量）",
    "composition_check": "描述主体可见性、识别到的最强参照物具体名称及画质评估",
    "angle_analysis": "识别车辆向量长轴，明确说明是相对于哪类参考线，估算偏离夹角",
    "distance_analysis": "描述轮迹点与边界位置，根据【车座-后轮连线中点】说明车身超出比例",
    "context_analysis": "判断是否有盲道检测，以及车辆接地点或车座是否压在盲道上"
  },
  "scores": {
    "composition_status": "[合规] / [基本合规] / [不合规-构图] / [不合规-无参照]",
    "angle_status": "[合规] / [不合规-角度]",
    "distance_status": "[完全合规] / [基本合规-压线] / [不合规-超界]",
    "context_status": "[合规] / [不合规-环境]"
  }
}
```""",
    "standard_p2": """
# Role
你是一位专业的共享单车运维质检员，负责通过照片判定车辆停放是否符合城市管理严苛标准。

# Task
请观察图片中【最显著】的一辆共享单车/电动车，按照以下四个维度进行逻辑评估。

# Guidelines & Criteria

## 1. 参照物识别与优先级 (Reference Priority)
- **核心原则**：判定方向与位置时，必须遵循以下证据优先级：
  1. **第一优先级**：地面白色/黄色停车标线。
  2. **第二优先级**：马路牙子（路缘石）或绿化带边缘。
  3. **第三优先级**：周围同类车辆的停放趋势。
  4. **第四优先级**：仅在无上述物体时，参考地砖缝隙方向。**注意：地砖方向不能作为判定合规的唯一强证据。**

## 2. 摆放角度合规性 (Angle)
- **基准线选择优先级**：
  1. **第一优先（标线参照）**：若画面中有明确的地面停车标线（白线/黄线），则以标线为基准。
  2. **第二优先（几何参照）**：若无标线，以马路牙子（路缘石）、墙边或花坛边缘为基准。
  3. **第三优先（趋势参照）**：若均无上述物体，参考地砖缝隙走向或周围车辆停放的统一趋势。

- **分类判定标准**：
  - **【场景 A：有停车标线】标准最宽松**
    - 允许**垂直**或**平行**：车辆中心轴线与标线夹角在 $0^\circ \pm 30^\circ$（平行）或 $90^\circ \pm 30^\circ$（垂直）范围内均视为**[合规]**。
  
  - **【场景 B：无标线，仅有路缘石/几何边界】标准严苛（强制垂直）**
    - **仅限垂直**：车辆中心轴线必须与路缘石保持垂直（夹角在 $90^\circ \pm 30^\circ$ 之间）。
    - **一票否决**：在此场景下，若车辆与路缘石“平行”或“斜向（偏离垂直度 $>30^\circ$）”摆放，即使未挡路也必须判定为**[不合规]**。

  - **【场景 C：弱参照（仅地砖/邻车趋势）】**
    - 车辆轴线应与参照物走向保持一致（偏离度 $\le 30^\circ$），不应出现明显的突兀摆放。


## 3. 停放距离与边界 (Distance & Boundary)
- **要求**：车身主体必须完全处于停车带或路缘石内侧。
- **严苛标准**：
  - [完全合规]：车身整体在内，未压线。
  - [基本合规-压线]：仅车轮压线，或车身超出线外部分小于 1/4（显著部分仍在内）。
  - [不合规-超界]：车身一半以上在线外，或停在马路牙子坡面上，或完全脱离路缘石参考区域。

## 4. 路面环境禁忌 (Contextual Veto)
- **一票否决项**：只要触碰以下区域，直接判为 [不合规-环境]。
  - **盲道识别强化**：注意寻找路面上【黄色】且带有【条状或点状凸起纹理】的区域。即便光线不佳，只要车辆压在带纹理的黄色色块上，即判定为压盲道。
  - **其他禁停区**：绿化带、机动车道、消防通道、人行道正中央出入口。

# Execution Process
1. **寻找强参照物**：先找线和路缘，不要被地砖缝隙误导。
2. **视觉特征检测**：专门扫描车辆下方是否有黄色凸起（盲道）。
3. **几何测量**：估算车辆中心轴与强参照物的夹角。
4. **综合判定**：若证据模糊，优先考虑对行人通行的潜在阻碍，收紧“合格”发放。

# Output Format (JSON)
必须严格按以下 JSON 格式回答：
{
  "step_by_step_analysis": {
    "composition_check": "描述识别到的最强参照物是什么（线/路缘/砖缝），并评估构图",
    "angle_analysis": "描述车辆相对于强参照物的具体夹角估算",
    "distance_analysis": "描述轮迹点与边界的距离，说明车身超出比例",
    "context_analysis": "详细描述地面是否有黄色凸起纹理（盲道）或其他禁停特征"
  },
  "scores": {
    "is_image_valid": boolean,
    "angle_status": "合规/不合规",
    "distance_status": "完全合规/基本合规-压线/不合规-超界",
    "context_status": "合规/不合规-环境"
  },
  "final_label": "根据权重输出最终标签：[合规] / [不合规-原因]"
}
""",
    "standard_p3": """
# Role
你是一位富有经验的共享单车运维质检员。你必须像几何测量仪一样精确，不仅要看停放位置，还要先判定照片是否具备质检条件。

# Task
请锁定图片中【最靠近画面中心、面积占比最大】的一辆共享单车/电动车作为主体，排除背景干扰，执行以下评估。

# 1. 前置质检准入 (Pre-check)
**在进行几何分析前，必须审查画面完整性。若符合以下任一条件，直接判定为“不合格-重拍”：**
- **[不合规-构图]**：画面过暗、过曝、拍摄距离过远（看不清地面纹理），或车辆主体被截断（例如只拍到一个车轮或车座，导致无法判定车长轴方向）。
- **[不合规-无参照]**：画面中完全缺失第一类（标线）、第二类（路缘/边界）参照物，且周围无其他车辆提供停放趋势参考。

# 2. 判定核心维度

## A. 参照物识别与优先级 (Reference Priority)
1. **第一优先级**：地面停车标线（白线/黄线）。
2. **第二优先级（几何边界）**：马路牙子（路缘石）、墙边、台阶边缘、花坛接缝。
3. **第三优先级**：周围同类车辆的整体停放朝向。

## B. 摆放角度：车辆向量定义 (Vector Definition)
- **车辆长轴向量**：定义为“后轮中点 -> 前轮/车把中心”的中心连线。
- **分类判定逻辑（解决幻觉核心）**：
  - **场景 A（有标线）**：车辆长轴必须与标线 **平行** 或 **垂直**（偏离 $0^\circ \pm 30^\circ$ 或 $90^\circ \pm 30^\circ$）。
  - **场景 B（无标线，仅有马路牙子/几何边界）**：**【强制垂直】**。
    - **视觉纠偏检查**：如果车辆像墙贴一样“贴着”或“沿着”马路牙子摆放（即长轴与路缘平行），**一票否决，判定为[不合规-角度]**。
    - **合规特征**：车辆必须呈“T字形”正对着路缘（长轴与边界夹角为 $90^\circ \pm 30^\circ$）。

## C. 停放距离与边界 (Distance)
- **要求**：车身主体必须完全处于停车带或路缘石内侧。
- **判定标准**：车身超出边界线/路缘石超过 1/2，或停在马路牙子下方的行车道上，判定为 **[不合规-超界]**。

## D. 环境禁忌 (Veto)
- **盲道识别**：寻找【黄色】且带有【条状/点状凸起纹理】的区域。只要车辆压住此类区域，立即判定为 **[不合规-压盲道]**。

# 3. 执行流程 (Execution Process)
1. **锁定主体**：识别画面中心最显著的车，忽略背景中远处的成排车辆。
2. **审查构图**：判断是否因为截断、模糊或无参照需要“重拍”。
3. **几何测量**：提取“车辆长轴向量”与“最高级基准线”，通过“T字形（垂直）”或“I字形（平行）”逻辑进行比对。
4. **决策分流**：输出最终结论。

# 4. Output Format (JSON)
必须严格按以下 JSON 格式回答：
{
  "step_by_step_analysis": {
    "composition_check": "描述识别到的最强参照物是什么（线/路缘/砖缝），并评估构图",
    "angle_analysis": "描述车辆相对于强参照物的具体夹角估算",
    "distance_analysis": "描述轮迹点与边界的距离，说明车身超出比例",
    "context_analysis": "详细描述地面是否有黄色凸起纹理（盲道）或其他禁停特征"
  },
  "scores": {
    "is_image_valid": boolean,
    "angle_status": "合规/不合规",
    "distance_status": "完全合规/基本合规-压线/不合规-超界",
    "context_status": "合规/不合规-环境"
  },
  "final_label": "根据权重输出最终标签：[合规] / [不合规-原因]"
}""",
  "standard_p4": """
# Role
你是一位专业的共享单车运维质检员，负责通过照片判定车辆停放是否符合城市管理严苛标准。你必须像高精度传感器一样，严格按照几何准则和环境优先级进行判定。

# Task
锁定画面中【最显著】的一辆共享单车/电动车为主体，排除背景干扰，执行以下评估流程。

# 参照物识别与优先级 (Reference Priority)
在执行任何合理性判断之前，必须按照以下证据优先级确立基准线：
1. **第一优先（标线参照）**：地面白色/黄色/红色等停车位标线。
2. **第二优先（几何参照）**：马路牙子（路缘石）、墙边与地面接缝、台阶边缘、花坛接缝等几何直线元素。
3. **第三优先（趋势参照）**：人行道地砖缝隙走向或周围车辆统一停放趋势。

# Guidelines & Criteria

## 1. 图像构图合规性 (Image Quality & Composition)
- **评估标准**：完整性（把手、坐垫、前后轮必须在取景框内）、参考物充分性。
- **状态选项（Options）**：
  - `[合规]`：画面清晰，车辆大部分在框内，背景有能分辨的第一优先或第二优先参照。
  - `[基本合规]`：车辆关键部位虽被截断在画面外，但画面中保留了第一/二类参考线，或周围邻车停放趋势明确。
  - `[不合规-构图]`：画面过暗/过曝、拍摄距离过远导致无法获取有效信息。
  - `[不合规-无参照]`：画面中未捕捉到标线、路缘石、墙边等标识，且也无相邻车辆方向参考。

## 2. 摆放角度合规性 (Angle Compliance)
- **基准线选择**：严格遵循上述【参照物优先级】。
- **向量定义**：以“后轮中点 -> 把手中心”为车辆中心轴线长轴向量。
- **判定准则**：
  - **优先以第一优先参照物为基准**：长轴向量与标线必须保持【平行】或【垂直】，偏差不得超过 30度。
  - **其次以第二优先参照物为基准**：**【强制垂直】**（长轴与边界呈T字形），偏差不得超过 30度。若长轴与边界平行（贴边/顺行摆放），直接判定为不合规。
- **状态选项（Status Options）**：`[合规]`、`[不合规-角度]`。

## 3. 停放距离合规性 (Distance Compliance)
- **评估标准**：前轮触地边缘至停车边线的垂直距离 Df，后轮触地边缘至停车边线的垂直距离 Dr。
- **判定准则**：所有关键点距离基准线误差不得超过 5cm（基于深度估算）。
- **状态选项（Status Options）**：
  - `[完全合规]`：车辆主体完全处于停车线/边界内侧且未压线。
  - `[基本合规-压线]`：车辆大部分位于内侧，超出线外部分不多于一个轮胎。
  - `[不合规-超界]`：车身 1/3 以上位于线外/边界外，或离线距离远超一个车身的宽度。

## 4. 路面环境合规性 (Contextual Compliance)
- **判定准则**：停放介质必须在硬化路面、砖面或指定格位。严禁停在绿化带、机动车道、消防通道、**盲道（带黄色条状或点状凸起纹理区域）**。
- **状态选项（Status Options）**：`[合规]`、`[不合规-环境]`。

# 极严输出约束 (Mandatory Constraints)
1. **严格匹配标签**：下方的 `status` 字段【必须且只能】从上述给定的选项中精确选择，严禁自创词汇。
2. **严禁额外解释**：不要在 `scores` 块的 status 字段中写任何括号、原因或补充描述。


# Output Format (JSON)
必须严格按以下 JSON 格式回答：
```json
{
  "step_by_step_analysis": {
    "composition_check": "描述主体可见性、识别到的最强参照物具体名称及画质评估",
    "angle_analysis": "识别车辆向量长轴，明确说明是相对于哪类参考线，估算偏离夹角，并指出几何关系（垂直/平行/斜停）",
    "distance_analysis": "估算前/后轮边距（Df/Dr），描述轮迹点与边界位置，说明车身超出比例",
    "context_analysis": "判断停放介质，说明是否有黄色盲道纹理、绿化带或其他禁停特征"
  },
  "scores": {
    "composition_status": "仅限选: [合规] / [基本合规] / [不合规-构图] / [不合规-无参照]",
    "angle_status": "仅限选: [合规] / [不合规-角度]",
    "distance_status": "仅限选: [完全合规] / [基本合规-压线] / [不合规-超界]",
    "context_status": "仅限选: [合规] / [不合规-环境]"
  }
}
""",

    "standard_p5": """
# Role
你是一位专业的共享单车运维质检员，负责通过照片判定车辆停放是否符合城市管理严苛标准。你必须像高精度传感器一样，严格按照几何准则和环境优先级进行判定。

# Task
锁定画面中【最显著】的一辆共享单车/电动车为主体，排除背景干扰，执行以下评估流程。

# 参照物识别与优先级 (Reference Priority)
在执行任何合理性判断之前，必须按照以下证据优先级确立基准线：
1. **第一优先（标线参照）**：地面白色/黄色/红色等停车位标线。
2. **第二优先（几何参照）**：马路牙子（路缘石）、墙边与地面接缝、台阶边缘、花坛接缝等几何直线元素。
3. **第三优先（趋势参照）**：人行道地砖缝隙走向或周围车辆统一停放趋势。

# Guidelines & Criteria

## 1. 图像构图合规性 (Image Quality & Composition)
- **评估标准**：完整性（把手、坐垫、前后轮必须在取景框内）、参考物充分性。
- **状态选项（Options）**：
  - `[合规]`：画面清晰，车辆大部分在框内，背景有能分辨的第一优先或第二优先参照。
  - `[基本合规]`：车辆关键部位虽被截断在画面外，但画面中保留了第一/二类参考线，或周围邻车停放趋势明确。
  - `[不合规-构图]`：画面过暗/过曝、拍摄距离过远导致无法获取有效信息。
  - `[不合规-无参照]`：画面中未捕捉到标线、路缘石、墙边等标识，且也无相邻车辆方向参考。

## 2. 摆放角度合规性 (Angle Compliance)
- **基准线选择**：严格遵循上述【参照物优先级】。
- **向量定义**：以“后轮中点 -> 把手中心”为车辆中心轴线长轴向量。
- **判定准则**：
  - **优先以第一优先参照物为基准，其次以第二优先参照物为基准**：长轴向量与基准必须保持【垂直】，偏差不得超过正负 30度。
  - **【强制垂直】**（长轴与边界呈T字形），偏差不得超过正负 30度。若长轴与边界平行（贴边/顺行摆放），直接判定为不合规。
- **状态选项（Status Options）**：`[合规]`、`[不合规-角度]`。

## 3. 停放距离合规性 (Distance Compliance)
- **评估标准**：前轮触地边缘至停车边线的垂直距离 Df，后轮触地边缘至停车边线的垂直距离 Dr。
- **判定准则**：所有关键点距离基准线误差不得超过 5cm（基于深度估算）。
- **状态选项（Status Options）**：
  - `[完全合规]`：车辆主体完全处于停车线/边界内侧且未压线。
  - `[基本合规-压线]`：车辆大部分位于内侧，超出线外部分不多于一个轮胎。
  - `[不合规-超界]`：车身 1/3 以上位于线外/边界外，或离线距离远超一个车身的宽度。

## 4. 路面环境合规性 (Contextual Compliance)
- **判定准则**：停放介质必须在硬化路面、砖面或指定格位。严禁停在绿化带、机动车道、消防通道、**盲道（带黄色条状或点状凸起纹理区域）**。
- **状态选项（Status Options）**：`[合规]`、`[不合规-环境]`。

# 极严输出约束 (Mandatory Constraints)
1. **严格匹配标签**：下方的 `status` 字段【必须且只能】从上述给定的选项中精确选择，严禁自创词汇。
2. **严禁额外解释**：不要在 `scores` 块的 status 字段中写任何括号、原因或补充描述。


# Output Format (JSON)
必须严格按以下 JSON 格式回答：
```json
{
  "step_by_step_analysis": {
    "composition_check": "描述主体可见性、识别到的最强参照物具体名称及画质评估",
    "angle_analysis": "识别车辆向量长轴，明确说明是相对于哪类参考线，估算偏离夹角，并指出几何关系（垂直/平行/斜停）",
    "distance_analysis": "估算前/后轮边距（Df/Dr），描述轮迹点与边界位置，说明车身超出比例",
    "context_analysis": "判断停放介质，说明是否有黄色盲道纹理、绿化带或其他禁停特征"
  },
  "scores": {
    "composition_status": "仅限选: [合规] / [基本合规] / [不合规-构图] / [不合规-无参照]",
    "angle_status": "仅限选: [合规] / [不合规-角度]",
    "distance_status": "仅限选: [完全合规] / [基本合规-压线] / [不合规-超界]",
    "context_status": "仅限选: [合规] / [不合规-环境]"
  }
}
"""
}


# API 配置池
BASE_URL = "https://api.ppinfra.com/openai"
API_KEYS = [
    "REDACTED_API_KEY_3",
    #"REDACTED_API_KEY_2",  # 新添加的 Key
    "REDACTED_API_KEY_1",
    "REDACTED_API_KEY_5"
]
MAX_WORKERS = 10

# ================= 核心工具函数 =================

def norm_yesno(x: str) -> str:
    if not x: return ""
    s = str(x).strip().lower()
    if any(k in s for k in ["yes", "true", "1", "合格", "合规", "[合规]"]): return "yes"
    if any(k in s for k in ["no", "false", "0", "不合格", "违规", "[不合格]", "不合规"]): return "no"
    return ""

def parse_vlm_response(response_text):
    try:
        # 1. 提取 JSON 块
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return "error", "未匹配到JSON结构", "fail", "fail", "fail", "fail", 0.0
            
        data = json.loads(json_match.group())
        scores = data.get("scores", {})
        
        # 提取各个子项
        comp = str(scores.get("composition_status", "")).strip()
        ang = str(scores.get("angle_status", "")).strip()
        dist = str(scores.get("distance_status", "")).strip()
        cont = str(scores.get("context_status", "")).strip()
        
        # 2. 评判逻辑：加权得分 或 一票否决
        if _scoring_engine:
            sr = _scoring_engine.score(comp, ang, dist, cont)
            res = "yes" if sr.is_compliant else "no"
            score_val = sr.final_score
        else:
            res = ScoringEngine.veto_judge(comp, ang, dist, cont)
            score_val = 1.0 if res == "yes" else 0.0
            
        reason = str(data.get("step_by_step_analysis", ""))
        
        return res, reason, comp, ang, dist, cont, score_val

    except Exception as e:
        return "error", f"解析崩溃: {str(e)} | 原文: {response_text[:50]}", "fail", "fail", "fail", "fail", 0.0

def process_single_image(args):
    image_name, folder_path, client, labels_dict, config = args
    image_path = os.path.join(folder_path, image_name)
    gt = labels_dict.get((image_name, folder_path), "N/A")
    
    start_t = time.time()
    try:
        with Image.open(image_path) as img:
            if img.mode == 'RGBA': img = img.convert('RGB')
            img.thumbnail(config['max_size'], Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=config['quality'])
            b64_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        res = client.chat.completions.create(
            model=config['model'],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT_LIB.get(config['prompt_id']) or load_prompt(config['prompt_id'])},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]}],
            max_tokens=600, temperature=0.1
        )
        vlm_out = res.choices[0].message.content
        pred, reason, comp, ang, dist, cont, w_score = parse_vlm_response(vlm_out)
        return [image_name, os.path.basename(folder_path), pred, gt, comp, ang, dist, cont, reason, round(time.time()-start_t, 3), w_score]
    except Exception as e:
        return [image_name, os.path.basename(folder_path), "error", gt, "err", "err", "err", "err", str(e), 0, 0.0]

# ================= 评估与报告 =================

def calculate_and_report(results):
    tp, tn, fp, fn, inv = 0, 0, 0, 0, 0
    lats = []
    for r in results:
        pred, gt = norm_yesno(r[2]), norm_yesno(r[3])
        if r[2] == "error": inv += 1; continue
        if gt == 'yes':
            if pred == 'yes': tp += 1
            else: fn += 1
        elif gt == 'no':
            if pred == 'no': tn += 1
            else: fp += 1
        if r[9] > 0: lats.append(r[9])
    
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    avg_lat = round(sum(lats)/len(lats), 3) if lats else 0

    print(f"\n{'='*20} 评估报告 (全量汇总) {'='*20}")
    print(f"总样本数 (Total Samples): {total}")
    print(f"无效/错误预测 (Invalid): {inv}")
    print("-" * 50)
    print(f"准确率 (Accuracy) : {acc:.2%}  (整体判断对的比例)")
    print(f"精确率 (Precision): {pre:.2%}  (模型判定为合规中实际合规的比例)")
    print(f"召回率 (Recall)    : {rec:.2%}  (合规样本被正确检出的比例)")
    print(f"F1分数 (F1-Score) : {f1:.2f}")
    print("-" * 50)
    print("混淆矩阵详情:")
    print(f"  [TP] 预测正确(合规): {tp}")
    print(f"  [TN] 预测正确(违规): {tn}")
    print(f"  [FP] 误判为合规 (实际违规): {fp} -> 漏抓违停")
    print(f"  [FN] 误判为违规 (实际合规): {fn} -> 过于严苛")
    print(f"平均单样本耗时: {avg_lat}s")
    print("=" * 60)
    return {"acc": acc, "f1": f1, "pre": pre, "rec": rec, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total, "invalid": inv, "avg_lat": avg_lat}

# ================= 主程序逻辑 =================

def main():
    print(f">>> 实验启动！模型: {CONFIG['model']}")
    print(f">>> 检测到 {len(API_KEYS)} 个 API Key，正在初始化资源池...")
    all_tasks = []
    global_labels = {}
    # 初始化客户端池
    clients = [OpenAI(base_url=BASE_URL, api_key=k) for k in API_KEYS]

    all_tasks = []
    global_labels = {}

    for folder in DATA_FOLDERS:
        if not os.path.exists(folder): continue
        l_path = os.path.join(folder, "labels.txt")
        if os.path.exists(l_path):
            with open(l_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split(",", 1)
                    if len(p) == 2: global_labels[(p[0].strip(), folder)] = norm_yesno(p[1])

        imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 均匀分配任务给客户端池
        for i, img_name in enumerate(imgs):
            # 关键：Round Robin 算法分配客户端
            assigned_client = clients[i % len(clients)]
            all_tasks.append((img_name, folder, assigned_client, global_labels, CONFIG))

    print(f">>> 任务分发完毕，共计 {len(all_tasks)} 个图片请求。")

    out_csv = os.path.join(SAVE_DIR, f"results_{CONFIG['exp_name'].replace('/','_')}_detailed.csv")
    final_results = []

    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['image_name', 'folder', 'result', 'ground_truth', 'composition', 'angle', 'distance', 'context', 'reason', 'latency_sec', 'weighted_score'])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 开启多线程推理
            for row in tqdm(executor.map(process_single_image, all_tasks), total=len(all_tasks), desc="多API联合推理"):
                writer.writerow(row)
                f_out.flush()
                final_results.append(row)

    # 3. 输出终端评估报告
    metrics = calculate_and_report(final_results)

    # 4. 保存汇总指标 (Append 模式)
    summary_path = os.path.join(SAVE_DIR, "all_experiments_summary.csv")
    metrics.update({"exp_name": CONFIG['exp_name'], "folders": len(DATA_FOLDERS), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    file_exists = os.path.exists(summary_path)
    with open(summary_path, 'a', newline='', encoding='utf-8-sig') as f:
        dict_writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists: dict_writer.writeheader()
        dict_writer.writerow(metrics)

    print(f"\n>>> 实验结束！详细结果已实时写入: {out_csv}")

if __name__ == "__main__":
    main()