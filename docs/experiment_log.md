# 实验日志 — 共享单车停放检测 benchmark v2 优化

> 记录所有实验的思想、尝试路径、结果、问题、迭代方法。每轮实验追加一节,便于后续总结。
> 起始:2026-07-02。目标:总 Acc≥90%、违规召回≥90%、延迟可接受(实时还车)。

## 0. 背景与目标

- **任务**:共享单车停放合规判定(端云协同,VLM 四维判定)
- **benchmark v2**:600 张(300 yes + 300 no),标准场景,四维 GT 齐全(position/medium/angle/state)。train/val/test = 300/150/150。
- **目标**:Acc≥90% + ViolRec≥90% + 延迟可接受(p95<15s 理想)
- **优化思路**:提示词与权重参数配套迭代;防过拟合(子集调参/全集验证);置信度+区间映射替代死板离散分(规划中)。

## 1. 旧四维基线(2026-07-01)

- **维度**:构图/角度/距离/环境(旧四维)
- **prompt**:cv_enhanced_p5
- **scoring**:scoring_optimized_cv_p4(权重 5%/20%/45%/30%,阈值 0.35)
- **结果(base_prompt_cv 轮)**:Acc 0.582, ViolRec 0.373, F1 0.654
- **对比 pure 轮**(纯prompt不带CV):Acc 0.5117, ViolRec 0.10 → CV 链路让违规召回 0.10→0.373(+27pp),验证"CV为骨"方向
- **问题**:旧四维 distance 维 71% 合规样本被 VLM 误判"超界"(无参照时默认判超界)

## 2. 新四维重构(2026-07-02)

- **动机**:旧四维 distance 感知不可靠(71% 误判),根因是"无参照→默认超界"
- **新维度**(对齐 IDEA.md §3):停放位置(position,合并区域+距离,基准线>路缘>邻车,无白线降级用路缘不算违规)/禁停介质(medium,盲道/绿化/禁停区)/角度(angle,[N/A]无参照时触发)/车辆状态(state,正立/倒伏)
- **关键标签**:`[无参照]` 修掉旧版 71% distance 误判
- **prompt**:cv_enhanced_v2_new4dim
- **parser/scoring**:适配新四维,angle=[N/A]时权重归一化
- **结果(新四维 baseline)**:Acc 0.542, ViolRec 0.207
- **问题**:[无参照] bug 修了,但 [无参照] 得 0.5 中性分让违规车过线,ViolRec 反降

## 3. grid_search 调参(2026-07-02)

- **做法**:在 train(300)扫权重+阈值+[无参照]分值,验证不退化
- **结果**:无配置达 ViolRec≥0.6,最高 0.247。[无参照]分值 0.0~0.5 影响小
- **结论**:问题不在 scoring 聚合,在 VLM 对 medium 判定力(79% 违规样本 medium 判合规)

## 4. CV 硬事实规则注入(2026-07-02)

- **做法**:prompt 加"CV 检测到盲道+重叠>0.1 → medium 必判 [不合规-盲道]"强规则
- **问题**:backend 第一版只改 prompt 文字,没改代码注入 medium_analysis 字段 → 强规则失效(medium 81%→86.7% 反升)
- **修复**:补 medium_analysis 字段注入
- **结果**:ViolRec 0.207→0.303(+46%),盲道识别 32→66(翻倍),Acc 0.600
- **教训**:prompt 引用的字段必须和代码注入的字段对齐(backend 改一半 bug 反复出现)

## 5. 2D 视觉直接标注(2026-07-02)

- **动机**:JSON 数值注入有翻译损失(几何→数值→文本→VLM再映射回空间);VLM 读 `overlap=0.3` 不如直接看图
- **做法**:升级 draw_wireframe_visual——轮廓+中文标签+Set-of-Mark数字①②③+2D重叠高亮(车∩盲道红/车∩停车线绿)+主车加粗;prompt 改"看图判"删数值强规则
- **结果**:Acc 0.588, ViolRec 0.360(+19%),盲道 66→93,parse错误 17→9
- **问题**:2D 掩模重叠判断3D空间关系不靠谱(遮挡假象:车挡住后面的盲道,2D重叠但3D没接触)

## 6. 深度辅助 3D 接触判断(2026-07-02)

- **动机**:2D 重叠的遮挡假象。用 Depth Anything V2 判断车接地部分与盲道/停车线的真实3D接触,过滤遮挡假象,只对真实接触高亮
- **做法**:depth_assist.py(Depth Anything V2 Large)+ 3D接触判断 + 只对real_contact=True高亮
- **结果**:Acc 0.607(最高), ViolRec 0.370(最高), F1 0.682, 盲道FP 20.4%→16.9%, 延迟+0.3s, 0错误
- **结论**:深度辅助过滤了部分遮挡假象,但改善小 → 遮挡假象不是主瓶颈,主瓶颈仍是 VLM medium 感知力

## 7. 模型对比矩阵(2026-07-02)

- **做法**:5 模型在同一套深度辅助方案上跑 600 张
- **结果**:

| 模型 | Acc | ViolRec | 盲道 | 延迟P95 | 评估 |
|---|---|---|---|---|---|
| qwen3-vl-30b-a3b-instruct | 0.615 | 0.370 | 87 | 11.3s | **最优,实时可用** |
| 30b-thinking | 0.578 | 0.350 | 140 | 30.5s | 慢4倍,盲道过判 |
| 235b-a22b | 0.590 | 0.237 | 38 | 14.1s | 太宽容 |
| glm-4.6v | 0.483 | 0.120 | 33 | 42.2s | 15.7%失败 |
| ernie-424b | 0.270 | 0.013 | 0 | 16.3s | 51.7%错误,废了 |

- **结论**:模型不是瓶颈。换大模型(235B/ERNIE)反而更"宽容"、ViolRec 更低。30B 已是最优。

## 8. 逐维准确率诊断(2026-07-02,诊断反转)

- **做法**:join 预测四维 vs GT 四维,算逐维准确率
- **结果**:

| 维度 | 准确率 | 主要问题 |
|---|---|---|
| state | 99.8% | 几乎全对 |
| medium | 78.0% | 还行 |
| **position** | **44.4%** | GT=[合规]174张误判[无参照] |
| **angle** | **46.9%** | 联动误判[N/A] |

- **诊断大反转**:真正瓶颈不是 medium(78%还行),是 position 滥用[无参照](Pred 300 vs GT 164)+ angle 联动[N/A]。VLM 没用路缘降级,没看到清晰白线就判[无参照]
- **这也解释了模型矩阵为何无效**:所有模型都滥用[无参照],是 prompt 设计问题不是模型能力

## 9. position 修复(2026-07-02,摆钟效应)

- **做法**:prompt 收窄[无参照]门槛——强化路缘降级优先、判断顺序(线→缘→邻车→无参照)、加示例
- **结果**:position[无参照]304→58(修对了!),angle[N/A]335→67。但 medium 退化(盲道89→51),Acc 0.615→0.550
- **问题**:摆钟效应——position 强了 medium 降。VLM 注意力被 position 抢走

## 10. 平衡版(2026-07-02,摆钟放大)

- **做法**:position 精简保留核心 + medium 强化加回去(盲道视觉特征+CV硬事实红高亮+绿化特征)
- **结果**:ViolRec 0.887(最高!),但 medium 过度敏感(盲道468!),position 修复丢失([无参照]293),Acc 0.605
- **根因分析(当前评分机制放大)**:
  - medium 权重 0.45(最高)+ [不合规-盲道]=0.0 → VLM 一判盲道,总分最多 0.55<0.60 锁违规
  - 平衡版 VLM 判 468 张盲道(过敏感)→ 大量违规判定 → ViolRec 0.887 但 FP 暴增,Acc 没涨
  - scoring 放大了 medium 的波动 → 摆钟
- **教训**:提示词和权重必须配套,平衡版 prompt 变了 scoring 没跟着调,放大了 medium 波动

## 11. 置信度+区间映射(规划中,2026-07-02)

### 动机
死板离散映射([合规]=1.0/[不合规]=0.0)是摆钟根因——一票否决式放大波动。改成置信度+区间:
- VLM 输出每维标签+置信度
- YOLO 检测置信度已有
- 标签→区间(如[合规]=[0.7,1.0]),按置信度插值
- 低置信[不合规]给0.3而非0.0,不再一票否决,摆钟缓解

### 阶段1:区间+置信度
- prompt 加 confidence 字段(0-1)
- 标签→区间映射,置信度插值
- 加权逻辑不变,分数连续

### 阶段2:特征融合+逻辑回归
- 特征:四维标签one-hot + VLM置信度 + YOLO conf + CV几何(IoU/重叠/深度接触)
- 逻辑回归/GBDT 做最终判定(可学习+可解释,比小网络稳,600张不易过拟合)

### 阶段3:小神经网络(终极,慎用)
- 仅阶段2不够时上,严格正则+交叉验证,防过拟合

### 置信度获取方式(C:对比验证)
- A. prompt 自报 confidence(JSON字段,简单但VLM可能不准)
- B. VLM logprob(API token概率,客观但要改调用)
- C. 两者都试,benchmark对比哪个更可靠 ← **当前选C**

### 注意
- VLM置信度需校准(温度缩放/Platt),否则过度自信不可信
- 别丢标签(标签是语义判断,作特征保留)
- 可解释性优先(逻辑回归>小网络,竞赛答辩友好)
- 和OPRO配套(特征工程后OPRO迭代prompt输出更准标签+置信度)

## 待续

下一轮实验追加新节,记录:思想/尝试路径/结果/问题/迭代方法。


## 第11轮: 区间+置信度实验 (2026-07-02)

### 思想
旧scoring标签->固定分数(0.0/0.5/1.0)丢失信息量。引入置信度让分数连续化:标签决定区间,置信度决定在区间内的位置。

### 实验设计
- A版: VLM自报confidence(JSON加position_confidence等字段)
- B版: VLM logprob(API logprobs=True,取标签token概率)
- 评分:区间映射[合规]->[0.7,1.0],[基本合规-压线]->[0.4,0.7],[不合规]->[0.0,0.3],[无参照]->[0.3,0.5]
- 加权逻辑不变,但分数按置信度在区间内线性插值

### 改动
- assets/prompts/cv_enhanced_v2_new4dim.yaml: JSON加confidence字段+置信度说明
- assets/configs/scoring_new4d_conf.yaml: 新区间映射配置
- modules/experiment/scoring.py: 新增score_interp(支持区间+置信度)
- modules/vlm/parser.py: VLMResult加confidence字段
- scripts/run_benchmark_v2.py: logprobs=True+top_logprobs=5

### A版结果(self-reported confidence)
| 指标 | 值 |
|------|-----|
| Acc | 0.582 |
| ViolRec | 0.940 |
| F1 | 0.348 |
| FP | 18 |
| FN | 233 |
| 延迟中位数 | 4.5s |

### B版结果(logprobs)
| 指标 | 值 |
|------|-----|
| Acc | 0.577 |
| ViolRec | 0.933 |
| F1 | 0.342 |
| FP | 20 |
| FN | 234 |
| 延迟中位数 | 4.8s |

### A/B对比
两版结果几乎一致。A版自报confidence与B版logprob结果等价,但A版更简单(无需logprobs API)。

### 问题
1. 区间映射threshold(0.45)太松 -- [不合规-盲道]区间[0.0,0.3]+[无参照]区间[0.3,0.5]加权后仍过线
2. VLM confidence值偏高(平均0.85+),校准不足
3. ViolRec 0.94但FN 233 -- scoring太偏向违规检出

### 下一步
- 调整区间映射(收紧[无参照]区间,降低medium权重)
- 置信度校准(Platt scaling / 温度缩放)
- 或用旧scoring(固定值)但保留confidence做ECA/Brier分析


## 第13轮: 特征融合+逻辑回归 (2026-07-02)

### 思想
纯加权评分(死板或区间)使ViolRec和FN同涨同跌——摆钟效应。用可学习模型替代手调权重,学组合规则(如"medium判盲道+CV重叠高+深度接触→真违规"vs"medium判盲道+无接触→过敏感降权")。

### 特征向量(35维)
- 四维标签 one-hot 13维 + VLM置信度 4维 + 区间连续分 4维 + final_score 1维
- CV几何: iou_parking/overlap_parking/overlap_tactile 3维
- 深度接触: contact_tactile/contact_parking 2维
- YOLO检测: tactile/curb/parking_lane detected + main_bike_conf 4维
- 交互特征: med_盲道×contact, med_盲道×nocontact, pos_超界×noiou, pos_无参照×curb 4维

### 训练
- LogisticRegression(class_weight=balanced, L2正则, C搜索)
- train(300)训, val(150)调正, test(150)最终报告

### 结果 (test 150张)
| 指标 | 纯加权(深度版) | 逻辑回归 | delta |
|------|--------------|---------|-------|
| Acc | 0.607 | 0.600 | -0.007 |
| Violation Recall | 0.370 | 0.680 | +0.310 |
| F1 | 0.691 | 0.565 | -0.126 |
| FN | 42 | 36 | -6 |
| FP | 189 | 24 | -165 |

### 关键突破
- 交互特征打破摆钟: inter_med_blind_x_nocontact 系数为负(-0.019),模型学会"VLM判盲道但无3D接触=过敏感,降权"
- FN从63(初版)降到36, FP从189降到24
- 特征重要性: tactile_detected > parking_lane_detected > pos_压线 > overlap_tactile > inter_med_blind_x_nocontact

### 问题
- 逻辑回归输出合规概率,需转为二分类判定(当前阈值0.5)。可调阈值优化。
- 置信度校准未做(温度缩放),下步可改进。
- 绿化带/禁停区仍依赖VLM视觉,CV不检测这两类。

### 下一步
- 优化阈值(ROC曲线搜索)
- 置信度校准(温度缩放)
- 组合OPRO(迭代prompt改善VLM标签+置信度输出)

## 第13轮: 逻辑回归真实特征版 (2026-07-02)

### 思想
纯加权(死板/区间)摆钟无解,改用逻辑回归学组合规则。真实CV特征(iou/overlap/contact/detected)+VLM置信度+区间连续分+交互特征(medium×contact等)。

### 结果(test 150)
| 指标 | 值 |
|---|---|
| Acc | 0.600 |
| ViolRec | 0.680 |
| F1 | 0.565 |
| 混淆 | TP39/FN36/FP24/TN51 |

对比:纯加权深度辅助 Acc 0.607/ViolRec 0.370;逻辑回归初版(占位符)ViolRec 0.947是过拟合假象。

### 特征重要性(top5)
- tactile_detected -0.037(盲道检出→违规)
- parking_lane_detected +0.030(停车线检出→合规)
- pos_压线 +0.029
- overlap_tactile -0.027(与盲道重叠→违规)
- inter_med_blind_x_nocontact -0.019(盲道判违规但无3D接触→降敏感,交互特征生效)

### 结论
逻辑回归ViolRec 0.37→0.68,学到组合规则,但离0.9远。瓶颈:VLM标签本身不准(position 44%/medium误判),逻辑回归救不回来。交互特征方向对但效果有限。

### 下一步候选
- B. few-shot示例注入提升VLM标签质量
- C. CV硬规则兜底(medium真实接触→硬否决)+逻辑回归组合
- D. 扩数据(全量3374标完)
## 14. v4 Baseline (2026-07-03, qwen3.6-35b-a3b)

### 动机
切换到 benchmark v4 数据集（1152 张，576 yes + 576 no，含四维 GT），用当前最优配置跑基线，输出逐维准确率供用户微调 prompt。

### 配置
- 模型: qwen/qwen3.6-35b-a3b（ppinfra 上 qwen3-vl-30b 下线，qwen3.6-35b 可用且视觉能力合格）
- 模式: cv（YOLOv8-Seg + Depth Anything V2 + 视觉标注 + VLM + scoring）
- Prompt: cv_enhanced_v2_new4dim
- Scoring: scoring_new4d_gs_best.yaml（threshold=0.6，weights: position=0.15/medium=0.45/angle=0.30/state=0.10）
- Workers: 8（32 导致 API rate limit 429 雪崩）
- 数据集: fourdim_gt_v4.json（1152 张，standard scene 全部）

### 结果

#### 整体指标
| 指标 | 值 |
|------|-----|
| Accuracy | 0.5460 |
| Precision | 0.5296 |
| Recall（合规） | 0.8229 |
| F1 | 0.6445 |
| Violation Recall（违规召回） | 0.2691 |
| 有效解析率 | 89.2%（88 张因 API 429 重试耗尽标记为 err） |

#### 逐维准确率
| 维度 | 准确率 |
|------|--------|
| position（停放位置） | 66.08% |
| medium（禁停介质） | 80.87% |
| angle（摆放角度） | 58.37% |
| state（车辆状态） | 92.12% |

#### Top 错误模式
**position**（最严重）:
- GT=[无参照] → Pred=[合规]: 134 张（11.9%）— VLM 在无线无缘时默认合规而非标记无参照
- GT=[合规] → Pred=[无参照]: 62 张（5.5%）— 无参照过度收紧
- GT=[合规] → Pred=err: 58 张（5.1%）— 解析失败

**medium**:
- GT=[合规] → Pred=err: 76 张（6.7%）
- GT=[合规] → Pred=[不合规-盲道]: 60 张（5.3%）— 盲道误报（摆钟问题延续）
- GT=[不合规-盲道] → Pred=[合规]: 41 张（3.6%）— 盲道漏检

**angle**（最差维度 58.37%）:
- GT=[不合规-斜停] → Pred=[合规]: 104 张（9.2%）— 斜停大量漏判
- GT=[N/A] → Pred=[合规]: 97 张（8.6%）— 角度不适用时错误判合规
- GT=[合规] → Pred=[不合规-斜停]: 69 张（6.1%）

**state**（最佳维度 92.12%）:
- GT=[正立] → Pred=err: 88 张（7.8%）
- 倒伏仅 1 张漏判（0.1%）

#### 延迟
| 指标 | 值 |
|------|-----|
| Mean | 52.33s |
| Median | 25.52s |
| P95 | 125.04s |
| 范围 | 5.85s ~ 1508.66s |

延迟偏高因 API 429 rate limit 导致大量重试（tenacity 默认 2s 间隔）。脚本侧 latency 统计有 bug（错误汇总为单值 34.77s）。

### 关键发现
1. **angle 是最大薄弱点**（58.37%），斜停漏判 104 张（被判定为合规），角度参照基准提取不稳定。
2. **position 的无参照处理混乱**：134 张应判无参照却判合规，62 张应判合规却判无参照——VLM 对"无线无缘"场景的判断不一致。
3. **medium 摆钟效应在 v4 上重现**：盲道误报（60 张合规判盲道）和盲道漏检（41 张）并存。
4. **state 几乎完美**（92.12%），倒伏识别仅 1 张漏判。
5. **API 429 问题严重**：88 张耗尽重试，延迟暴增。需增加 retry 次数或限流策略。

### 产出
- results.csv: 1130 行（含四维 status + 置信度 + CV 特征 + latency）
- summary.json
- confusion_matrix.txt

