# 主题队列

状态：`todo` → `doing` → `done`（完成时在 INDEX 留链接）。

格式：`- [status] slug | 中文标题 | 关键词`

## 种子主题（来自电气×AI 毕设方向池，仅作教辅选题参考）

- [done] pid-nn-motor-speed | 基于 PID 优化与神经网络的电机调速控制系统 | PID, 神经网络, 电机调速
- [done] pv-lstm-short-term | 面向配网消纳的光伏系统仿真及 LSTM 短期功率预测 | 光伏, LSTM, 配网消纳
- [done] wind-rf-abnormal | 基于随机森林的风电机组出力工况识别与异常预警 | 风电, 随机森林, 异常预警
- [done] park-load-clustering | 基于聚类的工业园区典型用电负荷模式识别 | 聚类, 负荷模式, 工业园区
- [done] dc-microgrid-voltage-ml | 基于仿真数据与机器学习的直流微网电压波动抑制 | 直流微网, 电压波动, 机器学习
- [done] renewables-load-partition | 基于聚类分析的新能源配网负荷特性分区仿真 | 聚类, 新能源配网, 负荷特性
- [done] lstm-voltage-sag | 基于 LSTM 时序模型的配电网电压暂降故障预测 | LSTM, 电压暂降, 故障预测
- [done] rf-pv-inverter-fault | 基于随机森林的光伏逆变器故障诊断仿真 | 随机森林, 光伏逆变器, 故障诊断

## 自定义主题

（在下方追加；Routine 优先取本文件中第一条 `todo`。补货条目可附课堂角度与参考链接。）

- [done] cnn-bilstm-residential-load | 居民侧日前负荷 BiLSTM / CNN-BiLSTM 对比 | BiLSTM, 居民负荷, 短期预测
  - 课堂角度：用公开居民侧聚合负荷做「昨日→次日」滚动预测，对比朴素日翻日、LSTM、BiLSTM、CNN-BiLSTM 的 RMSE，适合专科讲清双向时序与卷积特征提取各自带来什么。
  - 参考：https://arxiv.org/abs/2302.05033 · https://github.com/Varat7v2/STLF-BiLSTM-CNNBiLSTM

- [done] cnn-rf-battery-soh | 锂电健康状态 CNN+随机森林估算 | SOH, CNN, 随机森林, 储能
  - 课堂角度：从部分充放电曲线入手，先用 CNN 抽特征再用随机森林回归容量衰减，接储能/电动汽车导论里的「健康状态」概念，不要求搭完整电池电化学模型。
  - 参考：https://arxiv.org/abs/2010.10452

- [done] seq2point-nilm | 非侵入式负荷分解 seq2point CNN 入门 | NILM, CNN, 负荷分解
  - 课堂角度：只给总表功率，用滑动窗口 CNN 估计单台电器功率（seq2point），可选用 REDD/UK-DALE 公开数据做电器开关识别演示，贴近智能用电与需求侧管理。
  - 参考：https://arxiv.org/abs/1612.09106 · https://github.com/redefik/ConvNILM

- [done] cnn-pf-surrogate-5bus | 五母线潮流 CNN 代理模型课堂实验 | CNN, 潮流, 代理模型
  - 课堂角度：把蒙特卡洛潮流结果当标签，训练一维 CNN 近似电压/支路潮流，对照 pandapower 精确潮流，让学生体会「物理仿真 vs 数据代理」的速度与误差权衡。
  - 参考：https://github.com/enliten/IEEE-course_AI-for-Power-Energy-Systems

- [done] cnn-voltage-quality | 电压质量扰动时序 CNN 分类 | 电能质量, CNN, 暂降 swell
  - 课堂角度：把暂降、暂升、中断、谐波等波形做成多类分类任务，强调特征窗长与噪声鲁棒性，可与继电保护/电能质量实验课联动。
  - 参考：https://arxiv.org/abs/1904.00228

- [done] microgrid-inverter-switch-ml | 微网逆变器开关故障与异常 ML 检测 | 微网, 逆变器, 故障定位, KNN
  - 课堂角度：在含光伏与储能的逆变器主导微网里区分内部开关故障与虚假数据注入类异常，用 KNN 等分类器做开关定位，扩展已有「光伏逆变器故障」选题到系统级场景。
  - 参考：https://arxiv.org/abs/2411.08761

- [done] ml-fault-class-loc | 配网短路故障分类与定位 ML 对比 | 继电保护, 故障分类, 故障定位, 时序窗口
  - 课堂角度：用电磁暂态合成的短窗电压/电流，对比决策树、随机森林与浅层神经网络在故障类型与区段定位上的准确率与推理时延，把「保护定值」和「数据驱动」放在同一节课里对照。
  - 参考：https://arxiv.org/abs/2510.00831

- [done] xgb-dga-transformer | 油浸式变压器 DGA + XGBoost 诊断 | DGA, 变压器, XGBoost, 油色谱
  - 课堂角度：先讲 IEC 三比值编码的局限，再用公开 DGA 样本训练 XGBoost，对照特征重要度与气体产气机理，适合专科「变压器运维」实验把传统经验规则接到可复现的 ML 流水线。
  - 参考：https://arxiv.org/abs/2110.11467

- [todo] iforest-pmu-anomaly | PMU 同步相量 Isolation Forest 异常检测 | PMU, 异常检测, Isolation Forest, 广域监测
  - 课堂角度：在合成或公开 PMU 流上注入毛刺/丢包/频率偏移，用 Isolation Forest 做无监督告警，并与重构类方法做 F1 对比，强调「先检出再解释」的课堂节奏。
  - 参考：https://arxiv.org/abs/2509.25612

- [todo] pinn-swing-smib | 单机无穷大摇摆方程 PINN 入门 | PINN, 摇摆方程, 暂态稳定, 物理信息
  - 课堂角度：同一 MLP 结构分别做纯数据拟合与带配点残差的 PINN，观察稀疏样本下功角/转速轨迹，让学生体会「把微分方程写进损失函数」相对黑箱拟合的增益。
  - 参考：https://arxiv.org/abs/1911.03737 · https://arxiv.org/abs/2608.02599 · https://github.com/enliten/IEEE-course_AI-for-Power-Energy-Systems

- [todo] drl-bess-dispatch | 分时电价下储能 DQN 调度沙盒 | 储能, DQN, 强化学习, 分时电价
  - 课堂角度：用 24 小时负荷与峰谷电价搭一个充/放/待机三动作玩具环境，跑浅层 DQN 并画累计电费曲线，只要求讲清状态—动作—奖励，不要求上真实微网控制器。
  - 参考：https://arxiv.org/abs/2608.02599 · https://github.com/enliten/IEEE-course_AI-for-Power-Energy-Systems

- [todo] lstm-ev-charge-forecast | 电动汽车充电站负荷 LSTM 预测 | 充电站, LSTM, 负荷预测, 交通电气化
  - 课堂角度：把校园或公开充电桩功率按 15 min 聚合，对比日翻日基线与 LSTM 的日前误差，串起「交通侧不确定性 → 配电侧峰荷」的教学故事线。
  - 参考：https://arxiv.org/abs/2510.16719 · https://arxiv.org/abs/2512.17257

- [todo] gnn-grid-state-estimation | 配电网状态估计 GNN 入门 | GNN, 状态估计, 图结构, 量测
  - 课堂角度：把母线当节点、线路当边，用少量量测加伪量测训练图神经网络估计电压幅值/相角，与加权最小二乘结果对照，帮学生理解「电网拓扑本身就是一张图」。
  - 参考：https://arxiv.org/abs/2201.04056 · https://arxiv.org/abs/2207.11465

- [todo] cnn-electricity-theft | 智能电表窃电检测 CNN 课堂案例 | 窃电检测, CNN, 智能电表, 不平衡分类
  - 课堂角度：把用户日用电序列按周折成二维矩阵用 CNN 分类，重点讲正负样本极不平衡时为何看 AUC/召回而不是准确率，贴合用电检查与营销稽查岗位。
  - 参考：https://arxiv.org/abs/2102.06039 · https://github.com/henryRDlab/ElectricityTheftDetection

- [todo] yolo-insulator-defect | 无人机巡检绝缘子缺陷 YOLO 检测 | 绝缘子, YOLO, 目标检测, 输电巡检
  - 课堂角度：用公开无人机绝缘子图像跑现成 YOLO 检测缺陷（掉串/破损），讲清标注框、mAP 与漏检代价，连接输电线路运维与智能巡检实训。
  - 参考：https://arxiv.org/abs/2502.07179 · https://arxiv.org/abs/2410.11727 · https://github.com/InsulatorData/InsulatorDataSet

- [todo] cnn-pv-el-defect | 光伏组件 EL 图像缺陷 CNN 分类 | 光伏, 电致发光, CNN, 缺陷识别
  - 课堂角度：用公开 EL 电池片图像做「正常/缺陷」分类，从隐裂、断栅的物理成因讲到卷积特征，适合光伏运维与新能源专业的图像类入门实验。
  - 参考：https://arxiv.org/abs/2402.17611 · https://github.com/zae-bayern/elpv-dataset

- [todo] epf-day-ahead-lear-dnn | 日前电价预测 LEAR 与 DNN 对比 | 电价预测, 电力市场, LEAR, DNN
  - 课堂角度：借开源基准数据对比朴素基线、带 LASSO 的线性自回归（LEAR）与 DNN 的日前电价误差，顺带讲电力现货市场「日前出清」概念，专升本可延伸到储能套利。
  - 参考：https://arxiv.org/abs/2008.08004 · https://github.com/jeslago/epftoolbox

- [todo] wind-power-curve-cleaning | 风机功率曲线异常点识别与清洗 | 风电, 功率曲线, 异常数据, 数据清洗
  - 课堂角度：在风速—功率散点上识别限电、停机与传感器异常点，对比分箱统计法与基于图像/聚类的清洗方法，强调「先洗数据再建模」对风电预测的影响。
  - 参考：https://arxiv.org/abs/2307.08539 · https://arxiv.org/abs/2210.12104
