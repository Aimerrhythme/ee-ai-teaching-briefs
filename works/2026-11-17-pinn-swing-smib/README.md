# 单机无穷大摇摆方程 PINN 入门（2026-11-17）

> 主题：`pinn-swing-smib` · 关键词：PINN、摇摆方程、暂态稳定、物理信息  
> 定位：教辅 / 技术速览，供课堂与课题讨论参考；不是毕业论文代写。

## 1. 今日切入

**一句话**：同一个小 MLP，只喂 9 个稀疏采样点去拟合发电机功角 δ(t)、转速偏差 ω(t)，和在损失里再加上"摇摆方程残差"——后者就是 PINN（物理信息神经网络）。比较两条曲线在采样点之间的表现，是让学生直观理解"把微分方程写进损失函数"的最小实验。

**为何今天写**：本月（2026-11-09）arXiv 上刚出现一篇以单机无穷大（SMIB）为基准、改进 PINN 振荡拟合能力的预印本（mac-PIKAN，附代码）；而 2026 年 8 月 IEEE 在线课程配套的开源笔记本里正好有一个"普通 NN vs PINN 拟合摇摆方程"的入门 notebook。经典起点文献（Misyris 等，2019）+ 课程笔记本 + 最新改进，刚好凑成"从入门到前沿"的一条线。

## 2. 背景 5 分钟

**电气问题**：单机无穷大系统（SMIB）是《电力系统暂态分析》里讲功角稳定的标准模型。发电机转子运动满足摇摆方程（标幺制、忽略细节）：

```
dδ/dt = ω
M·dω/dt = Pm − Pmax·sin δ − D·ω
```

其中 M 为惯性常数相关系数，D 为阻尼系数，Pm 为机械功率，Pmax·sin δ 为电磁功率。小扰动下在工作点 δ0 线性化，令 Ks = Pmax·cos δ0（同步功率系数），得到二阶线性系统 M·δ'' + D·δ' + Ks·δ = ΔP，欠阻尼时就是一条衰减振荡曲线——有解析解，非常适合当"标准答案"来检验神经网络。

**AI 方法**：普通神经网络只做"数据拟合"：输入 t，输出 (δ̂, ω̂)，损失只有采样点误差。样本稀疏时，网络在采样点之间可以随便"拐弯"，振荡相位、衰减速度都可能不对。PINN 的做法是在时间轴上再撒一批**配点（collocation points）**，用自动微分算出 dδ̂/dt、dω̂/dt，把摇摆方程的残差平方也加进损失。这些配点**不需要标签**，物理方程本身就是监督信号。Misyris 等（2019）在 SMIB 上展示：PINN 所需训练数据显著减少，并能顺带辨识惯量、阻尼等不确定参数。

## 3. 方法草图

**输入**：时间 t ∈ [0, T]（建议归一化到 [0, 1] 或 [−1, 1]）；可选再把扰动量 ΔP、初值作为输入。

**模型**：MLP，tanh 激活（需要光滑可导；ReLU 二阶导为 0，不适合），2–3 个隐层 × 20–50 个神经元；输出 (δ̂, ω̂)。

**损失**（概念级伪代码，PyTorch）：
```python
t_c = torch.rand(N_c, 1, requires_grad=True) * T      # 配点，无需标签
d, w = net(t_c).split(1, dim=1)
dd = autograd.grad(d.sum(), t_c, create_graph=True)[0]
dw = autograd.grad(w.sum(), t_c, create_graph=True)[0]
r1 = dd - w                                            # 运动学残差
r2 = M*dw - (dP - D*w - Ks*d)                          # 线性化摇摆方程残差
# 非线性版本：r2 = M*dw - (Pm - Pmax*torch.sin(d) - D*w)

loss_data = mse(net(t_data), y_data)                   # 9 个稀疏样本
loss_ic   = mse(net(t0), y0)                           # 初值 δ(0), ω(0)
loss_phys = (r1**2).mean() + (r2**2).mean()
loss = loss_data + loss_ic + lam * loss_phys           # lam 为物理项权重
```
对照组 = 同一网络、同一初始化，但 `lam = 0`。

**输出与评价**
- 在密集网格（如 500 点）上与解析解 / ODE 数值积分对比：δ、ω 的 RMSE 或相对 L2 误差；
- 画 δ(t)、ω(t) 曲线 + 残差 |r2| 随 t 的分布；
- 外推测试：只在 [0, T/2] 给数据，看 [T/2, T] 段两种模型的表现；
- 进阶：把 D 或 M 设为可训练参数，观察 PINN 能否从数据中"反推"阻尼/惯量（参数辨识）。

## 4. 最新公开信息

1. **Physics-Informed Neural Networks for Power Systems**（Misyris, Venzke, Chatzivasileiadis，arXiv:1911.03737，2019-11 首发，v3 2020-01）  
   作者称是首次把 PINN 框架系统引入电力系统，以 SMIB 为引导算例：确定转子角与频率等动态状态及惯量、阻尼等不确定参数，摘要报告比传统方法最高快约 87 倍，并强调所需训练数据显著减少、网络结构更简单。是本主题的"起点文献"。  
   https://arxiv.org/abs/1911.03737
2. **Bridging AI and Power Systems Education Using a Hands-On Executable Framework**（Yin, She, Feng, Fangxing Li，arXiv:2608.02599，2026-08-03）及配套仓库 `enliten/IEEE-course_AI-for-Power-Energy-Systems`  
   面向初学者的电力 AI 可执行模块库（IEEE 在线课程 + IEEE PES 网络研讨会），难度阶梯中的前沿模块包括"摇摆方程 PINN"。仓库中的 `06_PINN_Frequency/PINN_Frequency.ipynb` 用线性化摇摆方程（示例参数 M=2、D=1、Ks=2，阻尼比 ζ=0.25）、9 个无噪声合成稀疏样本，对比"普通 NN vs PINN"，可在笔记本 CPU 或 Colab 上运行，可直接作为课堂演示底稿。  
   https://arxiv.org/abs/2608.02599 · https://github.com/enliten/IEEE-course_AI-for-Power-Energy-Systems
3. **Modal-Augmented Chebyshev PIKAN for Power Systems**（Nosrati 等，arXiv:2611.13340，2026-11-09，投 2027 IEEE PES GM，附代码）  
   指出常规 PINN 结构难以捕捉机电振荡；用线性化摇摆方程的闭式模态响应作为试探解主干，再由 Chebyshev 型物理信息 KAN 学非线性修正，并精确满足初始功角/转速。在 SMIB 基准上报告：相对 PINN，功角误差由 1.59% 降至 0.29%、频率误差由 5.26% 降至 0.99%，且在小样本和低阻尼下更稳。适合给专升本 / 本科学生作为"PINN 的局限与改进"延伸阅读。  
   https://arxiv.org/abs/2611.13340 · https://github.com/komeilnosrati/mac-PIKAN

## 5. 教辅提示

**可布置的小实验（2 学时）**
1. 用 `scipy.integrate.solve_ivp`（或线性化解析解）生成 SMIB 参考轨迹，T = 10–20 s。
2. 取 5 / 9 / 20 个稀疏样本，分别训练普通 NN（lam=0）与 PINN（lam=1），记录密集网格上的 δ、ω 误差。
3. 把阻尼 D 调小（ζ 从 0.25 降到 0.05），观察两者在"振荡更多、衰减更慢"时谁先崩——对应文献 3 里"低阻尼更难"的结论。
4. 选做：把 D 设为 `nn.Parameter`，看 PINN 反推的阻尼与真值差多少。

**预期讨论点**：样本越少，PINN 相对普通 NN 的优势越明显；样本足够多时两者差距缩小。PINN 不是"更强的网络"，而是"多了一份免费的监督信号"。

**常见坑**
- 用 ReLU 激活：二阶导恒为 0，物理残差项无效；
- 时间不做归一化：t 取 0–20 时 tanh 很快饱和，长时间段学不出振荡（可分段训练或做时间缩放）；
- 物理项权重 lam 过大：网络先满足方程却偏离初值，得到"平凡解"；过小则退化为普通 NN；
- 只看采样点误差：普通 NN 在采样点上可能和 PINN 一样好，差别在点与点之间；
- 把线性化模型用在大扰动场景：大扰动必须用 sin δ 非线性形式，否则谈不上暂态稳定。

**延伸阅读**：电力系统暂态分析教材中的等面积法则（可与 PINN 结果对照临界切除时间概念）；同仓库已有 [五母线潮流 CNN 代理模型](../2026-11-09-cnn-pf-surrogate-5bus/README.md)，两篇可组成"稳态代理 → 动态物理信息"的对照课。

## 6. 免责声明

本文为教学参考与公开资料整理，文中数字均引自所列公开文献或开源笔记本，未经独立复现；不构成暂态稳定校核、保护整定或调度运行依据。
