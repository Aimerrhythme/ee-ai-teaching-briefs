# 参考链接 · seq2point-nilm

## 论文与预印本

- Zhang C., Zhong M., Wang Z., Goddard N., Sutton C. *Sequence-to-point learning with neural networks for non-intrusive load monitoring*. arXiv:1612.09106；AAAI 2018.  
  https://arxiv.org/abs/1612.09106
- Haddad H., Jerbi F., Smaali I. *SEDR-Seq2P: A Lightweight Dilated Residual Sequence-to-Point Network for Multi-Task Industrial NILM*. arXiv:2607.28693 (2026).  
  https://arxiv.org/abs/2607.28693
- Wang X., Tang G., Xue J., Keshav S., Li T., Ding C. *DualNILM: Energy Injection Identification enabled Disaggregation with Deep Multi-Task Learning*. arXiv:2508.14600 (2025).  
  https://arxiv.org/abs/2508.14600

## 开源代码与数据

- MingjunZhong/seq2point-nilm（作者相关 seq2point 实现，UK-DALE / REDD 预处理说明）  
  https://github.com/MingjunZhong/seq2point-nilm
- redefik/ConvNILM（课程向 Keras CNN，结构接近 Zhang et al.）  
  https://github.com/redefik/ConvNILM
- nilmtk/nilmtk-contrib（含 `Seq2PointTorch` 等 PyTorch 分解器）  
  https://github.com/nilmtk/nilmtk-contrib
- UK-DALE 数据集（Kelly & Knottenbelt, *Scientific Data* 2015）  
  https://jack-kelly.com/data/
- REDD 数据集（Kolter & Johnson）  
  http://redd.csail.mit.edu/
- IMDELD 工业负荷分解数据（IEEE DataPort）  
  https://ieee-dataport.org/documents/industrial-machines-dataset-electrical-load-disaggregation-imdeld

## 延伸（仅读到摘要 / 题录）

- *Towards Robust NILM: A Unified Seq2Point Benchmarking Framework with Balanced Model*. IEEE ISGT Europe 2025：统一 UKDALE / REFIT / ECO 上的 Seq2Point 评测管线，并提出参数量约降 94% 的 Balanced 变体。  
  https://doi.org/10.1109/isgteurope64741.2025.11305695
- *Transfer capabilities of Seq2Seq and Seq2Point CNN architectures in Non-intrusive Load Monitoring with unseen appliances*. *Mathematics and Computers in Simulation* (2025)：分析训练聚合中未见电器带来的误差结构。  
  https://doi.org/10.1016/j.matcom.2025.05.021
