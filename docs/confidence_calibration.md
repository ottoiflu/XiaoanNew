# 置信度校准方法调研

> 第11轮验证 VLM 置信度有效(A版自报 vs B版logprob 等价,A版简单),但 VLM 置信度偏高(均值0.85+),需校准才能在阶段2逻辑回归中可靠使用。

## 1. 为什么需要校准

VLM(及多数 LLM)输出置信度往往"过度自信":说0.9的实际准确率可能0.6。直接用会误导下游模型(逻辑回归把高置信当强信号)。校准让"置信度=真实准确率"。

第11轮观察:VLM confidence 均值 0.85+,但实际各维准确率 position 44%/medium 78%/angle 47%/state 99.8%——置信度与准确率严重不匹配,必须校准。

## 2. 主流校准方法

### 温度缩放(Temperature Scaling)— 推荐首选
- 原理: logits / T,T>1 降温(变平),T<1 升温。只调一个参数 T
- 在 val 集上优化 T 使 NLL 最小
- 优点:简单、不改变排序(只调锐度)、对分类任务标准做法
- 实现:`sklearn` 无直接,手写几行:对每个样本 logit 除以 T,sigmoid,在 val 上最小化 NLL

### Platt Scaling
- 原理:用一个逻辑回归把原始置信度映射到校准概率 P = sigmoid(a*logit+b)
- 适合:原始置信度与准确率呈 S 形关系
- 实现:`sklearn.calibration.CalibratedClassifierCV(method='sigmoid')`

### Isotonic Regression
- 原理:非参数,学一个单调映射(任意形状)
- 适合:置信度与准确率关系不规则
- 风险:数据少(600张)易过拟合,需更多数据
- 实现:`CalibratedClassifierCV(method='isotonic')`

## 3. 评估指标

| 指标 | 含义 | 实现 |
|---|---|---|
| **Brier Score** | 预测概率与真实标签的均方误差,越低越好 | `sklearn.metrics.brier_score_loss` |
| **ECE**(Expected Calibration Error) | 按置信度分桶,各桶"平均置信度-实际准确率"差的加权平均,越低越好 | 手写,分10桶 |
| **MCE**(Maximum Calibration Error) | 各桶差的最大值,看最差情况 | 手写 |
| **可靠性图** | 置信度 vs 准确率曲线,直观 | matplotlib |

## 4. VLM 视觉判定的推荐方案

**温度缩放(首选)** + ECE 评估:
1. 收集 train 集每张图的四维置信度 + 是否判对(1/0)
2. 每维独立校准:在 val 上找最优 T(position/medium/angle/state 各一个 T)
3. 校准后置信度 = sigmoid(logit(原始conf)/T),或简化:conf_calibrated = (conf)^(1/T) 类似
4. 用 ECE/Brier 在 test 验证校准效果
5. 校准后的置信度喂给阶段2逻辑回归

**Platt Scaling 备选**:如果温度缩放不够(关系非 S 形),用 Platt(逻辑回归映射),`CalibratedClassifierCV` 一行代码。

**Isotonic 慎用**:600 张数据易过拟合,除非数据量翻倍。

## 5. 实现要点

- 校准在 val 集上拟合(不能用 train/test,防泄漏)
- 每维独立校准(各维准确率差异大,position 44% vs state 99.8%,一个 T 不通用)
- 校准后重新跑阶段2逻辑回归,看 Acc/ViolRec 是否提升
- 校准参数 T 存模型,推理时用

## 6. 与阶段2的关系

阶段2逻辑回归的输入置信度,必须是校准后的。流程:
1. VLM 输出原始 confidence
2. 温度缩放校准(conf → conf_calibrated)
3. 校准置信度作特征喂逻辑回归
4. 逻辑回归输出最终合规概率

校准是阶段2的前置步骤,不单独做最终判定。
