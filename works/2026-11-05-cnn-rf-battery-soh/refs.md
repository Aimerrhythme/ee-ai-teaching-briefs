# 参考链接 · cnn-rf-battery-soh

## 论文与预印本

- Yang N., Song Z., Hofmann H., Sun J. *Robust State of Health Estimation of Lithium-ion Batteries Using Convolutional Neural Network and Random Forest*. arXiv:2010.10452 (2020)；*Journal of Energy Storage* 48 (2022) 103857.  
  https://arxiv.org/abs/2010.10452
- Ispizua B., Gil-López S., Arrizabalaga L., Laña I. *Real-Time State-of-Health Estimation and Online Degradation Prognosis from Partial Battery Discharge Using Physics-Informed Neural Networks*. arXiv:2608.14764 (2026).  
  https://arxiv.org/abs/2608.14764
- Wang F., Zhai Z., Liu B., Zheng S., Zhao Z., Chen X. *Open access dataset, code library and benchmarking deep learning approaches for state-of-health estimation of lithium-ion batteries*. *Journal of Energy Storage* 77 (2024) 109884.  
  https://doi.org/10.1016/j.est.2023.109884

## 开源代码与数据

- SOHbenchmark（XJTU / MIT 数据，CNN / LSTM / GRU / MLP / Attention，支持部分充电输入）  
  https://github.com/wang-fujin/SOHbenchmark
- Severson K. A. 等 *Data-driven prediction of battery cycle life before capacity degradation*. *Nature Energy* 4 (2019) 383–391（124 节 A123 LFP 电芯老化数据来源）  
  https://doi.org/10.1038/s41560-019-0356-8

## 延伸（仅读到摘要）

- *Transfer learning-based SOH estimation from partial charging data with polarization-aware modeling*. *Scientific Reports* (2026)：在 Severson 数据 20% 容量充电片段上，LSTM 先预训练电压预测再微调 SOH，摘要报告 MAE 从 1.75% 降到 0.91%。  
  https://www.nature.com/articles/s41598-026-48906-4
