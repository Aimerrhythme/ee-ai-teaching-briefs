"""
日前电价预测：朴素基线 vs LEAR（LASSO 自回归）vs DNN —— 纯 numpy 课堂玩具
- 合成 3 年（3×364 天）逐时「日前负荷预测 / 风电预测 / 日前电价」，电价由「剩余负荷 → 阶梯报价曲线」生成，
  含周内效应、燃料价格慢漂移、高剩余负荷尖峰、大风低谷负电价。
- 特征照搬 Lago 等 (2021) 的 247 维设定：p(d-1,d-2,d-3,d-7) 各 24 点 + 两个外生量 d 日 / d-1 / d-7 各 24 点 + 7 维星期哑变量。
- LEAR：每小时一个 LASSO 线性模型（共 24 个），asinh 变换，坐标下降手写；对比 56 天与 728 天两种标定窗口。
- DNN：247 → 64 → 32 → 24 的多输出 MLP（ReLU、Adam、早停），手写反向传播；再做 4 个随机种子的集成。
- 前 2 年用于标定，最后 1 年（364 天）为测试集；每 28 天滚动重标定一次（真实做法是逐日重标定）。
仅依赖 numpy；数据全为合成，结果不代表任何真实市场。
"""
import numpy as np
import time

rng = np.random.default_rng(20261126)
D, H = 3 * 364, 24
t0 = time.time()

# ---------------- 1. 合成数据 ----------------
d_idx = np.arange(D)[:, None]; h_idx = np.arange(H)[None, :]
dow = np.arange(D) % 7                                     # 0=周一 … 6=周日
season = 1 + 0.12 * np.cos(2 * np.pi * d_idx / 364)        # 冬高夏低
profile = 0.75 + 0.15 * np.exp(-((h_idx - 9) ** 2) / 8) + 0.22 * np.exp(-((h_idx - 19) ** 2) / 6) \
          - 0.12 * np.exp(-((h_idx - 3) ** 2) / 10)
wk = np.where(dow >= 5, 0.85, 1.0)[:, None]                # 周末负荷低
load = 30.0 * season * profile * wk * (1 + rng.normal(0, 0.02, (D, H)))   # GW，日前负荷预测

w_day = np.zeros(D); w_day[0] = 6
for d in range(1, D):                                      # 风电日均值：AR(1) 天气过程
    w_day[d] = 6 + 0.75 * (w_day[d - 1] - 6) + rng.normal(0, 2.2)
wind = np.clip(w_day[:, None] * (1 + 0.15 * np.cos(2 * np.pi * (h_idx - 2) / 24))
               + rng.normal(0, 0.6, (D, H)), 0, 18)       # GW，日前风电预测

fuel = np.zeros(D); fuel[0] = 45
for d in range(1, D):                                      # 燃料/碳价：缓慢均值回归的漂移（准非平稳）
    fuel[d] = fuel[d - 1] + 0.01 * (45 - fuel[d - 1]) + rng.normal(0, 0.6)
fuel = np.clip(fuel, 20, 80)[:, None]

resid = load - wind                                        # 剩余负荷 GW
price = fuel * (0.35 + 0.025 * resid) + 4.0 * np.exp(0.45 * (resid - 30)) \
        - 25.0 * np.exp(-0.9 * (resid - 6).clip(0))         # 尖峰 + 低谷负电价
noise = np.zeros((D, H))
for d in range(1, D):
    noise[d] = 0.5 * noise[d - 1] + rng.normal(0, 2.5, H)  # 日间持续的残差
price = price + noise                                      # €/MWh

# ---------------- 2. 特征（247 维）----------------
def features(d):
    z = np.zeros(7); z[dow[d]] = 1
    return np.concatenate([price[d - 1], price[d - 2], price[d - 3], price[d - 7],
                           load[d], wind[d], load[d - 1], load[d - 7], wind[d - 1], wind[d - 7], z])
days = np.arange(7, D)
X_all = np.zeros((D, 247)); X_all[days] = np.array([features(d) for d in days])
TEST0 = D - 364
test_days = np.arange(TEST0, D)

def scale(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd

# 价格 asinh 变换（Uniejewski 等的方差稳定化）：先用中位数/MAD 归一化
def asinh_fit(y):
    med = np.median(y); mad = np.median(np.abs(y - med)) * 1.4826 + 1e-8
    return med, mad
def asinh_fwd(y, med, mad): return np.arcsinh((y - med) / mad)
def asinh_inv(z, med, mad): return np.sinh(z) * mad + med

# ---------------- 3. 模型 ----------------
def lasso_cd(X, y, lam, n_iter=200, w=None):
    n, p = X.shape
    w = np.zeros(p) if w is None else w.copy()
    col2 = (X ** 2).sum(0) / n
    r = y - X @ w
    for _ in range(n_iter):
        wmax = 0.0
        for j in range(p):
            if col2[j] == 0: continue
            wj = w[j]
            rho = X[:, j] @ r / n + col2[j] * wj
            nw = np.sign(rho) * max(abs(rho) - lam, 0) / col2[j]
            if nw != wj:
                r -= X[:, j] * (nw - wj); w[j] = nw; wmax = max(wmax, abs(nw - wj))
        if wmax < 1e-5: break
    return w

LAMS = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
def lear_fit_predict(train_days, pred_days):
    Xtr, Xte = scale(X_all[train_days], X_all[pred_days])
    med, mad = asinh_fit(price[train_days])
    Y = asinh_fwd(price[train_days], med, mad)
    nv = max(7, len(train_days) // 5)                      # 末尾 20% 做验证选 λ
    out = np.zeros((len(pred_days), H)); nz = []
    for h in range(H):
        y = Y[:, h]; ym = y[:-nv].mean()
        best, bl = 1e9, LAMS[0]
        w = None
        for lam in LAMS:                                   # λ 从大到小，热启动
            w = lasso_cd(Xtr[:-nv], y[:-nv] - ym, lam, w=w)
            e = np.mean(np.abs(Xtr[-nv:] @ w + ym - y[-nv:]))
            if e < best: best, bl = e, lam
        ym = y.mean()
        w = lasso_cd(Xtr, y - ym, bl)
        nz.append((w != 0).sum())
        out[:, h] = Xte @ w + ym
    return asinh_inv(out, med, mad), np.mean(nz)

def dnn_fit_predict(train_days, pred_days, seed, epochs=400):
    r = np.random.default_rng(seed)
    Xtr, Xte = scale(X_all[train_days], X_all[pred_days])
    med, mad = asinh_fit(price[train_days])
    Y = asinh_fwd(price[train_days], med, mad)
    nv = len(train_days) // 5
    Xa, Ya, Xv, Yv = Xtr[:-nv], Y[:-nv], Xtr[-nv:], Y[-nv:]
    sizes = [247, 64, 32, 24]
    Ws = [r.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    bs = [np.zeros(b) for b in sizes[1:]]
    params = Ws + bs; m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    lr, b1, b2, l2, step = 1e-3, 0.9, 0.999, 1e-4, 0
    def fwd(X, W, b):
        a1 = np.maximum(X @ W[0] + b[0], 0); a2 = np.maximum(a1 @ W[1] + b[1], 0)
        return a1, a2, a2 @ W[2] + b[2]
    best, best_p, bad = 1e9, None, 0
    for ep in range(epochs):
        perm = r.permutation(len(Xa))
        for k in range(0, len(Xa), 32):
            idx = perm[k:k + 32]; X, Yb = Xa[idx], Ya[idx]
            a1, a2, out = fwd(X, Ws, bs)
            g = np.sign(out - Yb) / len(idx)               # MAE 损失（与评价指标一致）
            gW2 = a2.T @ g; gb2 = g.sum(0); g = (g @ Ws[2].T) * (a2 > 0)
            gW1 = a1.T @ g; gb1 = g.sum(0); g = (g @ Ws[1].T) * (a1 > 0)
            gW0 = X.T @ g; gb0 = g.sum(0)
            grads = [gW0 + l2 * Ws[0], gW1 + l2 * Ws[1], gW2 + l2 * Ws[2], gb0, gb1, gb2]
            step += 1
            for i, (p, gr) in enumerate(zip(params, grads)):
                m[i] = b1 * m[i] + (1 - b1) * gr; v[i] = b2 * v[i] + (1 - b2) * gr ** 2
                p -= lr * (m[i] / (1 - b1 ** step)) / (np.sqrt(v[i] / (1 - b2 ** step)) + 1e-8)
        ev = np.mean(np.abs(fwd(Xv, Ws, bs)[2] - Yv))
        if ev < best - 1e-4: best, best_p, bad = ev, [p.copy() for p in params], 0
        else:
            bad += 1
            if bad >= 30: break                            # 早停
    W, b = best_p[:3], best_p[3:]
    return asinh_inv(fwd(Xte, W, b)[2], med, mad)

# ---------------- 4. 滚动预测（每 28 天重标定）----------------
REC = 28
preds = {k: np.zeros((364, H)) for k in ["LEAR-56", "LEAR-728", "DNN (seed 0)", "DNN 集成(4)"]}
nz56, nz728 = [], []
for s in range(0, 364, REC):
    pd_ = test_days[s:s + REC]; d0 = pd_[0]
    preds["LEAR-56"][s:s + REC], n1 = lear_fit_predict(np.arange(d0 - 56, d0), pd_); nz56.append(n1)
    preds["LEAR-728"][s:s + REC], n2 = lear_fit_predict(np.arange(d0 - 728, d0), pd_); nz728.append(n2)
    dn = [dnn_fit_predict(np.arange(d0 - 728, d0), pd_, seed=sd) for sd in range(4)]
    preds["DNN (seed 0)"][s:s + REC] = dn[0]
    preds["DNN 集成(4)"][s:s + REC] = np.mean(dn, 0)
preds["LEAR 集成(56+728)"] = (preds["LEAR-56"] + preds["LEAR-728"]) / 2

# ---------------- 5. 评价 ----------------
y = price[test_days]
naive7 = price[test_days - 7]
naive_std = np.where(np.isin(dow[test_days], [0, 5, 6])[:, None], price[test_days - 7], price[test_days - 1])
mae7 = np.mean(np.abs(y - naive7))
spike = y >= np.quantile(y, 0.95)
def report(name, p):
    e = y - p
    mae = np.mean(np.abs(e)); rmse = np.sqrt(np.mean(e ** 2))
    smape = 100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p) + 1e-8))
    print(f"| {name} | {mae:.2f} | {mae / mae7:.3f} | {smape:.2f} | {rmse:.2f} | {np.mean(np.abs(e[spike])):.2f} |")

print(f"测试集 364 天 × 24 h；价格均值 {y.mean():.2f}，最小 {y.min():.2f}，最大 {y.max():.2f} €/MWh；"
      f"负电价小时占比 {100 * np.mean(y < 0):.2f}%；尖峰阈值（95 分位）{np.quantile(y, 0.95):.2f}")
print("| 方法 | MAE | rMAE | sMAPE[%] | RMSE | 尖峰小时 MAE |")
print("|---|---|---|---|---|---|")
report("朴素：一周前同小时 p(d-7,h)", naive7)
report("朴素：标准日翻日（周一/六/日用 d-7）", naive_std)
for k in ["LEAR-56", "LEAR-728", "LEAR 集成(56+728)", "DNN (seed 0)", "DNN 集成(4)"]:
    report(k, preds[k])
print(f"LEAR 每小时模型平均非零系数（共 247 个候选）：56 天窗 {np.mean(nz56):.1f}，728 天窗 {np.mean(nz728):.1f}")
print(f"总耗时 {time.time() - t0:.0f} s")
