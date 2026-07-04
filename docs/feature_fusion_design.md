# 特征融合设计 — 阶段2(逻辑回归)

> 打破纯加权+阈值摆钟的方案。纯加权(死板或区间)ViolRec 与 FN 同涨同跌,无法同时优化。阶段2用可学习模型替代手调权重,学组合规则。

## 1. 动机

第9-11轮反复出现摆钟效应:
- position 强化 → medium 退化
- medium 强化 → 合规误杀(FN 暴增)
- 区间+置信度 → 摆动连续化但没打破

根因:单维一票否决式聚合(medium 判违规就压分)放大任何单维波动。要破局,需模型能学"组合规则"而非"单维加权"——如"medium 判违规 + CV 重叠高 + 深度接触 → 真违规",与"medium 判违规 + CV 无接触 → medium 过敏感,判合规"区分。

## 2. 特征向量设计

每张图组成特征向量:

| 类别 | 特征 | 来源 | 说明 |
|---|---|---|---|
| **四维标签 one-hot** | position 4维 + medium 4维 + angle 3维 + state 2维 = 13维 | VLM 输出 | 语义判断,结构化 |
| **VLM 置信度** | position_conf/medium_conf/angle_conf/state_conf = 4维 | VLM 输出(自报,第11轮A版验证有效) | 第11轮A/B等价,A版简单 |
| **YOLO 检测置信度** | 主车 conf + 是否检出停车线/路缘/盲道 = 4维 | YOLO | CV 侧置信度 |
| **CV 几何** | iou_with_parking_lane + overlap_with_tactile + overlap_with_curb = 3维 | YOLO mask 计算 | 空间关系硬事实 |
| **深度辅助** | real_contact_tactile(bool) + real_contact_parking(bool) + 深度差 = 3维 | Depth Anything | 3D 接触判断,过滤遮挡假象 |
| **区间连续分** | position_score/medium_score/angle_score/state_score(区间插值后) = 4维 | scoring_new4d_conf | 保留阶段1成果作特征 |
| **总计** | ~31 维 | | |

标签 + 置信度 + CV + 深度 + 连续分,多源融合。

## 3. 模型选型

| 模型 | 优点 | 缺点 | 600张适用性 |
|---|---|---|---|
| **逻辑回归** | 可解释(系数可看)、稳、不易过拟合、竞赛答辩友好 | 只学线性组合,需手工构造交互特征 | **推荐**,31维+交互特征足够 |
| GBDT(LightGBM/XGBoost) | 自动学交互、非线性、特征重要性可看 | 易过拟合、稍黑盒 | 次选,需强正则 |
| 小神经网络 | 可学复杂非线性 | 黑盒、600张严重过拟合风险、难解释 | **不推荐**,数据太少 |

### 推荐方案:逻辑回归 + 手工交互特征
- 基础:31维线性组合
- **关键交互特征**(手工构造,让逻辑回归能学组合规则):
  - medium_盲道 × real_contact_tactile(盲道判违规且真3D接触 → 强违规信号)
  - medium_盲道 × (1-real_contact_tactile)(盲道判违规但无3D接触 → 过敏感,降权)
  - position_超界 × iou_with_parking_lane_low(超界且CV说车在线外 → 强违规)
  - position_无参照 × curb_detected(无白线但有路缘 → 该降级判不是无参照)
- 这些交互特征让逻辑回归能区分"真违规"和"过敏感",正是打破摆钟的关键

## 4. 训练/验证/测试流程

- **train 300**:训逻辑回归(含交互特征),学权重
- **val 150**:early stop + 超参调(L1/L2 正则强度),退化则回滚
- **test 150**:最终报告,全过程不参与调参
- 评分:balanced_accuracy + ViolRec≥0.6 约束(逐步逼近0.9)
- 防过拟合:L2 正则(Ridge) + 交叉验证(train内5折) + 特征数控制在 ~40维以内

## 5. 特征重要性分析

- 逻辑回归系数绝对值 → 各特征对合规/违规的贡献
- 标准化特征后比较系数 → 重要性排序
- 输出特征重要性条形图,验证"哪些特征真有判别力"(预期:medium×real_contact、position×iou 等交互特征靠前)

## 6. 实现要点

- sklearn LogisticRegression(class_weight='balanced' 处理 yes/no 均衡)
- PolynomialFeatures(degree=2, interaction_only=True) 自动生成交互,但只保留手工选的关键交互(避免维度爆炸)
- 训练数据:从 results.csv(逐张四维+置信度)+ fourdim_gt_v2.json(GT) + CV几何/深度特征 join
- 模型存 joblib,推理时直接 predict_proba

## 7. 与 OPRO 配套

特征融合落地后,OPRO 继续迭代 prompt,但目标变了:让 VLM 输出的标签+置信度更适合逻辑回归消费(更准、更校准)。提示词与模型参数(逻辑回归权重)配套迭代,正是用户最初要求。

## 8. 预期

- 打破摆钟:逻辑回归学到"组合规则",不再单维一票否决,ViolRec 和 FN 可同时优化
- 可解释:系数说话,竞赛答辩能讲清"为什么这么判"
- 目标:Acc≥90% + ViolRec≥90%(逻辑回归在 train 上学最优组合,test 验证)
