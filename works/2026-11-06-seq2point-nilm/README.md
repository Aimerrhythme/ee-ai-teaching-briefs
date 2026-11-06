# 非侵入式负荷分解 seq2point CNN 入门

日期：2026-11-06  
slug：`seq2point-nilm`  
关键词：NILM，CNN，负荷分解，seq2point

## 今日切入

主题池下一条是非侵入式负荷监测（NILM）里的经典入门架构：**sequence-to-point（seq2point）+ 一维 CNN**——只给总表有功功率的一段滑动窗口，网络只预测「窗口中点」那一个时刻某台电器的功率。今天值得写，是因为 2026 年工业侧又出现了以 Seq2Point 为骨干的轻量改进（SEDR-Seq2P），同时户用侧在光伏注入场景下把 seq2point 当成对照基线（DualNILM）。把 2018 年的原型和这两条新工作放在一起，正好讲清三件事：**为什么只预测中点、居民公开数据怎么开课、以及「多机并发 / 表后光伏」会怎样打穿朴素假设**。本篇按「问题 → 窗口设定 → CNN 草图 → 诚实评估」做方法速览。

## 背景 5 分钟

**电气问题。** 智能电表通常只报整栋楼（或整户）的总功率 \(y_t\)，但需求侧管理、分时电价解释、异常用电排查都想知道「此刻是冰箱还是洗衣机在跑」。给每台电器装子表叫侵入式监测（ILM），成本高；NILM 试图只从总表还原各电器功率：

\[
y_t = \sum_{i=1}^{I} x_{i,t} + u_t + \varepsilon_t
\]

其中 \(x_{i,t}\) 是目标电器 \(i\) 的功率，\(u_t\) 是未建模负荷，\(\varepsilon_t\) 是噪声。这是典型的**单通道盲源分离**：观测只有一路，解不唯一。教学上要先让学生分清：

- **事件检测**：找开关沿（功率阶跃），再分类是哪台电器；适合二态电器，对渐变、多档位较弱。
- **稳态回归 / 状态估计**：每个时刻估计功率或 ON/OFF；seq2point 属于这一路。
- **采样率**：UK-DALE 常用约 6 s 电器通道、REDD 约 3 s；课堂演示别混用不同采样率却不重采样。

**AI 方法。** 早期常用加性阶乘隐马尔可夫模型（AFHMM）等，要手工塞功率档位、持续时间等先验。深度学习把「签名」交给卷积自己学：

1. **Seq2Seq**：输入一段总功率窗口，输出同长度的电器功率窗口；每个时刻被多个重叠窗口预测，取平均后边沿容易被抹平。
2. **Seq2Point**：同样输入长度 \(W\) 的总功率窗口，但**只预测该窗口中点**的电器功率。表示能力集中在「前后文都看得见」的中点，每个时刻只有一次预测，不必再平均。
3. **一维 CNN**：卷积核在功率序列上滑动，捕捉开关沿、平台高度、典型通电时长；可视化特征图时，常能看到网络自动学到了这些「手工签名」。

## 方法草图

**输入（课堂尺度）**

- **数据**：UK-DALE（英国家庭）与 REDD（美国家庭）公开集；专科演示可先只做 kettle / microwave / fridge 中的一两台。
- **窗口**：原型论文对五类电器统一用 \(W=599\) 点（约对应 UK-DALE 6 s 采样下近一小时上下文的量级，具体以你重采样后的秒数为准）；工业多任务设定里也有用统一窗长（如 155）的做法。
- **标签**：目标电器在窗口中点 \(\tau = t+\lfloor W/2\rfloor\) 的功率 \(x_\tau\)；序列两端先用零填充 \(\lceil W/2\rceil\)。
- **预处理**：按电器的「导通均值 / 标准差」做 z-score（原型表里给出了各电器参数）；**统计量只能在训练集上估计**。
- **划分**：按**房屋**划分训练 / 测试（例如 UK-DALE 用 1、3、4、5 训、2 测），模拟「装到没见过的户」的迁移场景，不要按时间片随机打乱同一户。

**模型或算法**

基线 A — **阈值事件法**：总功率差分超过阈值则判开/关，再按功率档位归类（适合引入问题，精度有限）。  
基线 B — **Seq2Seq CNN**：同结构网络，但输出整段窗口再对重叠预测取平均。  
模型 — **Seq2Point CNN**：多层一维卷积 + 全连接，输出标量中点功率；每台目标电器通常**各训一个网络**（one-to-one）。  
扩展（只讲概念）— **多任务 one-to-many**：一个网络同时估多台机器（工业场景）；或把表后光伏注入当成额外任务（DualNILM）。

**输出与指标**

- **MAE**（W）：逐时刻绝对误差，对离群预测不那么敏感。
- **SAE**（归一化信号聚合误差）：按日（或固定时段）对能量求和再比相对误差，回答「日报电量能不能用」。
- **可选**：\(F_1\) / 匹配率（先把功率阈值成 ON/OFF 再算事件对齐）；工业文里还会报 NDE、\(R^2\)、推理时延。

课堂用草图（概念级，不是某一仓库的训练脚本）：

```text
# 以 UK-DALE kettle 为例（采样与房屋划分按你的预处理而定）
mains, kettle = load_ukdale(houses_train=[1,3,4,5], house_test=2)
W = 599
mains_pad = zero_pad(mains, W//2)

X, y = [], []
for t in range(len(mains)):
    window = mains_pad[t : t+W]
    X.append( (window - mu_mains) / sigma_mains )
    y.append( (kettle[t] - mu_app) / sigma_app )   # 中点标签

cnn = Conv1Dstack(...)   # 多层 1D CNN + Dense -> 1
fit(cnn, X_train, y_train, loss=MSE_or_MAE)

y_hat = inverse_scale( cnn.predict(X_test) )
report MAE_W, SAE_daily
plot(mains_test, kettle_test, y_hat)   # 看开关沿是否对齐、平台高度是否漂
```

## 最新公开信息

1. **原型：Zhang、Zhong、Wang、Goddard、Sutton，arXiv:1612.09106（2016 年 12 月预印本；AAAI 2018）。** 提出 seq2point：输入总功率滑动窗口，输出目标电器中点功率；与同结构 seq2seq 对比，并证明在相同架构下 seq2point 对中点后验的 KL 近似不劣于 seq2seq。UK-DALE 上相对 Kelly 的 seq2seq，文中报告整体 MAE 约降 84%、SAE 约降 92%；相对本文实现的 seq2seq，整体 MAE 约 15.5 W vs 18.0 W。REDD 上 seq2point 整体 MAE 约 23.7 W，优于同文 seq2seq 的约 26.6 W。窗长统一 599；可视化最后一层特征图显示网络会自动抓住开关沿、功率档与持续时间。实现为 TensorFlow；作者后续仓库 [MingjunZhong/seq2point-nilm](https://github.com/MingjunZhong/seq2point-nilm) 可作对照。  
   https://arxiv.org/abs/1612.09106

2. **工业多任务轻量改进：Haddad、Jerbi、Smaali（Wattnow），arXiv:2607.28693（SEDR-Seq2P，2026）。** 在工业数据集 IMDELD（1 Hz 有功、八台机器）上做 **one-to-many**：一个网络同时分解多台设备。统一协议下对比 Seq2Seq / Seq2SubSeq / Seq2Point / GRU / WaveNet：Seq2Point 平均 MAE 约 2814，优于 Seq2Seq 约 2999；WaveNet 精度更高（平均 MAE 约 2541）但推理更慢。提出的 SEDR-Seq2P 在 Seq2Point 上加膨胀残差块与 Squeeze-and-Excitation，相对 Seq2Point 基线平均 MAE 降到约 2608（文中概括为 MAE 约降 7%、\(R^2\) 约升 1%、匹配率约升 0.8%），相对 WaveNet 推理时延约降 58%。课堂可借此讨论：**居民「一台模型对一台电器」≠ 工业「多机并发」**，以及准确率之外还要看部署时延。  
   https://arxiv.org/abs/2607.28693

3. **表后光伏注入下的对照与扩展：Wang、Tang 等，arXiv:2508.14600v2（DualNILM，2025）。** 指出户用光伏 / 微逆等表后（BTM）注入会让净功率出现「负向」成分，电器签名被掩盖（signal eclipse）；传统只做消耗侧分解的假设被打破。DualNILM 用 CNN + Transformer，**电器状态用 seq2point 式投影、注入功率用 seq2seq**，在实验室微逆数据与 REDD / UK-DALE 合成光伏增强集上，状态识别 F1 明显高于把 Seq2Point 当基线的结果（实验室交叉验证中 DualNILM 平均 F1 约 84.5%，同表 Seq2Point 约 34.3%——说明**朴素 seq2point 不是为双向潮流设计的**）。合成光伏增强数据与方法计划审稿后开源。课堂启示：讲 NILM 时加一张「有光伏的净表读数」示意图，再问学生误差会往哪飘。  
   https://arxiv.org/abs/2508.14600

开源与教具方面，除作者仓库外，课程向 Keras 实现见 [redefik/ConvNILM](https://github.com/redefik/ConvNILM)；NILMTK 生态的 PyTorch 接口里有 `Seq2PointTorch`（[nilmtk/nilmtk-contrib](https://github.com/nilmtk/nilmtk-contrib)）。数据需从 UK-DALE / REDD 官方渠道申请或下载，注意许可与体积。

## 教辅提示

**小实验（由易到难）**

1. **先看签名。** 在 UK-DALE 或 REDD 上画出总功率与 kettle / fridge 子表曲线，标出开、关沿和平台功率，口头描述「用人眼怎么分解」。
2. **阈值基线。** 用功率差分 + 固定阈值做开/关检测，算一天的 SAE；记下误检（同时开两台、功率接近的电器）。
3. **最小 seq2point。** 只做一台电器、缩短窗长（如 99 或 199）先跑通训练曲线，再与 599 对比 MAE。
4. **Seq2Seq vs Seq2Point。** 同深度 CNN，比较边沿附近的误差和训练时间；对照原型论文 Table 2。
5. **进阶讨论。** （a）把训练户与测试户对调，看迁移跌多少；（b）在总功率上叠加一段「假光伏」（白天减去平滑鼓包），观察 MAE / F1 崩塌，对应 DualNILM 提出的问题；（c）阅读 SEDR-Seq2P 的 AccD 表，讨论工业现场为何不能只报 MAE。

**常见坑**

- 按时间点随机划分训练 / 测试：同一户相邻窗口几乎一样，指标虚高；应按房屋（或至少按天块）划分。
- 归一化用了含测试户的全局均值方差。
- 窗长与采样率不匹配（把 1 Hz 工业数据的窗长直接套到 1/6 Hz 居民数据）。
- 只报 MAE、不看 SAE：逐点误差大但日报电量仍可用（或反过来）。
- 多台电器共用一个未校准的功率阈值做 \(F_1\)，和回归 MAE 混着比。
- 在有光伏 / 储能的净表数据上，直接套消耗侧 seq2point 却不声明设定。
- 把不同论文、不同数据集上的 MAE（W）横比成「谁更先进」。

**延伸阅读。** 先读原型论文第 III 节（seq2point 定义与定理）和 Table 2–3；再看特征图一节理解「自动学到的签名」。工业向读 SEDR-Seq2P 第 I、IV 节（one-to-many 与 AccD）；光伏场景读 DualNILM 摘要与第 III 节问题定义。工具链可浏览 NILMTK 文档与 `Seq2PointTorch` 接口。

链接收录见同目录 [`refs.md`](./refs.md)。

## 免责声明

本文是教学参考：公开文献、公开数据与开源代码的方法速览和练习提示，不是智能电表产品、需求响应结算或用电稽查的工程依据。文中 MAE、SAE、F1、时延等数字来自各自的数据集、电器子集与实验设定，彼此不可直接比较，引用时请回到原文核对。UK-DALE、REDD、IMDELD 等数据的使用须遵守其许可与申请要求。本文不构成毕业论文、课程设计或实验报告代写；仿真、实验、图表和结论须由学习者本人完成并独立撰写。
