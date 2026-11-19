# 电动汽车充电站负荷 LSTM 预测（2026-11-19）

> 主题：`lstm-ev-charge-forecast` · 关键词：充电站、LSTM、负荷预测、交通电气化  
> 定位：教辅 / 技术速览，供课堂与课题讨论参考；不是毕业论文代写。

## 1. 今日切入

**一句话**：把一个充电站（或一片园区的若干充电桩）的功率记录按 15 min 聚合成时序，用"昨天同一时刻"做朴素基线，再训练一个 LSTM 做日前（次日 96 点）预测，比较两者误差——讲清楚"交通侧的不确定性怎样变成配电侧的峰荷"。

**为何今天写**：前几篇做过居民负荷 BiLSTM（11-04 的 `cnn-bilstm-residential-load`）和储能调度，充电负荷正好是两者之间缺的那块：它比居民负荷更"稀疏、尖峰、受人行为驱动"，是讲**基线比较**和**数据量陷阱**的好素材。主题池附带的两篇预印本也刚好一正一反：一篇是校园充电站 LSTM 全流程（还接到了配电网潮流仿真），一篇是跨 4 个城市、5 类模型的系统性基准，结论是"没有哪个模型通吃"。

## 2. 背景 5 分钟

**电气问题**：充电桩功率大（交流慢充约 7 kW，直流快充几十到上百 kW），且到达时间集中在早上上班、傍晚下班等时段。大量充电叠加到配电台区上，会造成**配变过载、末端电压跌落、峰谷差拉大**。配电运维和充电运营方都需要提前知道"明天各时段大概要多少功率"，用于台区容量评估、有序充电、电价引导或储能配合。难点在于：单个站的曲线很"毛"（有车就几十 kW，没车就是 0），聚合到区域或城市层面才平滑。

**AI 方法**：LSTM（长短期记忆网络）是一种循环神经网络，靠输入门、遗忘门、输出门控制"记住什么、忘掉什么"，适合学习日周期、周周期这类时间依赖。典型做法是：取过去 L 个时刻（如 7 天 × 96 点）的功率加上时间特征（时刻、星期几、是否节假日），输出未来 H 个时刻（如 96 点）的功率。它的对手不该只是 ARIMA，更重要的是**朴素基线**：日翻日（明天 = 今天）、周翻周（下周二 = 本周二）。如果 LSTM 赢不了周翻周，模型再复杂也没意义。

## 3. 方法草图

**数据准备**
1. 原始数据通常是"一次充电会话"一行（开始时间、结束时间、电量 kWh）。把每次会话的电量按时长均摊到经过的 15 min 时段，得到站级功率序列（kW）；
2. 缺失时段插值、异常值（超过桩额定功率）截断，做 min-max 归一化（只用训练集统计量）；
3. 按时间顺序切分训练/验证/测试（**不能随机打乱**，否则未来信息泄漏）。

**模型（概念级伪代码）**
```python
# X: [样本, L=672(7天×96), 特征=功率+sin/cos(时刻)+sin/cos(星期)+节假日]
# y: [样本, H=96]  次日 96 个点
model = Sequential([
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(96)
])
model.compile(loss="mse", optimizer=Adam(1e-3))
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          callbacks=[EarlyStopping(patience=10)])

baseline_day  = 前一天的 96 点          # 日翻日
baseline_week = 上周同一天的 96 点      # 周翻周
```

**评价指标**
- MAE、RMSE（单位 kW）——注意**不要用 MAPE**：充电负荷大量时段为 0，MAPE 会除以 0 或爆炸；
- 峰值误差：预测日最大功率与实际日最大功率之差、峰值出现时刻偏差（对配变容量评估更关键）；
- 分工作日 / 周末统计误差。

## 4. 最新公开信息

1. **LSTM-Based Forecasting and Analysis of EV Charging Demand in a Dense Urban Campus**（Ressler, Grijalva, Ignacio, Torres, Cuadra Rojas, Moghadam, Narimani；arXiv:2510.16719）  
   以加州州立大学北岭分校（CSUN）停车楼 B2 充电站数据为主：先做插值、归一化、滚动均值和 FFT 周期分析，再用 2 层 LSTM（MSE 损失、Adam）做日 / 周 / 月多尺度预测；最后把预测负荷与实测负荷分别放进校园配电网的 PowerWorld 模型，代表母线电压差小于 0.04%（约 4×10⁻⁴ pu）。值得课堂讨论的一点：作者坦言只有 1 年数据时精度明显不足，文中对比了加入 10 年带噪**合成数据**后的结果——这正好可以引出"数据增强能否代替真实数据、测试集是否也是合成的"这一批判性问题。  
   https://arxiv.org/abs/2510.16719

2. **Electric Vehicle Charging Load Forecasting: An Experimental Comparison of Machine Learning Methods**（Iason Kyriakopoulos, Yannis Theodoridis；arXiv:2512.17257）  
   用 Palo Alto、Boulder、Dundee、Perth 四个公开充电数据集，在站级 / 区域级 / 城市级三种聚合尺度、短期（≤30 min，10 min 步长）/ 中期（≤8 h，2 h 步长）/ 长期（≤5 天，1 天步长）三种预测时长下，对比 ARIMA、XGBoost、GRU、LSTM、Transformer。摘要结论：短期区域/城市级 Transformer 的 MAE 常比 ARIMA、XGBoost 低 10–15%；中长期 GRU/LSTM 多数情况误差最低，部分中期城市级场景 MAE 比 ARIMA 降低 50% 以上；XGBoost 在局部（站级）场景有竞争力；ARIMA 难以扩展到长时长与粗聚合。代码与数据链接开源（MIT 许可）。  
   https://arxiv.org/abs/2512.17257 · https://github.com/DataStories-UniPi/electric-vehicle-charging-load-forecasting

3. **可选公开数据：Caltech ACN-Data**（加州理工自适应充电网络公开的充电会话数据，需注册获取 API token）——若没有本校数据，可用来复现"会话→15 min 功率"的预处理。  
   https://ev.caltech.edu/dataset

## 5. 教辅提示

**可布置的小实验（2–4 学时）**
1. 用基准仓库或 ACN-Data 中任一站点，做"会话→15 min 功率"转换，画日均曲线与周热力图（行=星期，列=时刻），先让学生**用眼睛**看出周期；
2. 实现日翻日、周翻周两条基线，算测试集 MAE/RMSE；
3. 训练单层/双层 LSTM，和基线同表对比；再把同一模型用在"单站"和"多站求和"两种聚合上，观察误差（相对值）变化——呼应文献 2 的"尺度决定模型排名"；
4. 专升本延伸：把预测曲线叠加到一个简单台区（配变容量假设为 400 kVA），计算过载时段数，体会"预测误差→容量判断"的链条。

**常见坑**
- 随机划分训练/测试集 → 未来信息泄漏，结果虚高；
- 用 MAPE 评价含大量 0 值的充电负荷；
- 只和 ARIMA 比、不和周翻周比；
- 归一化用了全数据集的最大最小值（又一种泄漏）；
- 单站数据量太小却堆很深的网络，验证集早早过拟合。

**延伸阅读**
- 本仓 [`works/2026-11-04-cnn-bilstm-residential-load`](../2026-11-04-cnn-bilstm-residential-load/README.md)：居民负荷 BiLSTM 对比，可与本篇对照"平滑负荷 vs 尖峰负荷"；
- 文献 2 的 GRU 与 LSTM 对比结果：参数更少的 GRU 常常不输 LSTM，可让学生自行验证。

## 6. 免责声明

本文为教学参考与公开资料整理，文中数字均引自所列预印本或其摘要，未经独立复现；不作为充电设施规划、配电网改造或工程投运的依据。
