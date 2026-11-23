"""合成数据课堂演示：智能电表日用电序列 → 按周折成 2D 矩阵 → 卷积特征 → 窃电二分类。
对比 (a) 全判"正常"的哑分类器  (b) 1D 原始序列逻辑回归  (c) 2D 卷积特征 + 逻辑回归（不加权 / 类别加权）。
重点看：样本极不平衡（窃电约 9%）时，准确率为什么会骗人，AUC / 召回为什么更有用。
仅依赖 numpy；所有数据随机合成，不对应任何真实用户。"""
import numpy as np
rng = np.random.default_rng(20261123)
W, D = 12, 7                    # 12 周 × 7 天 = 84 天日用电量 (kWh)
N, theft_ratio = 4000, 0.09     # 窃电占比约 9%（与公开 SGCC 数据集的量级相近）

# ---- 1. 正常用户：基础用量 × 周内模式(工作日/周末) × 季节漂移 × 噪声 ----
def normal_users(n):
    base = rng.lognormal(np.log(8), 0.5, (n, 1, 1))
    weekly = 1 + rng.normal(0, 0.15, (n, 1, D)); weekly[:, :, 5:] *= rng.uniform(0.9, 1.4, (n, 1, 1))
    trend = 1 + np.linspace(0, 1, W)[None, :, None] * rng.normal(0, 0.2, (n, 1, 1))
    x = base * weekly * trend * rng.lognormal(0, 0.12, (n, W, D))
    return x

# ---- 2. 窃电：从某周起改变计量（四种常见合成方式，参考文献中常用的 FDI 类型） ----
def inject_theft(x):
    x = x.copy()
    for i in range(len(x)):
        s = rng.integers(3, W - 3)                     # 窃电起始周
        kind = rng.integers(4)
        seg = x[i, s:]
        if kind == 0:   seg *= rng.uniform(0.3, 0.7)                                  # 整体按比例少计
        elif kind == 1: seg *= rng.uniform(0.1, 0.8, seg.shape)                       # 每天随机少计
        elif kind == 2: seg[:] = seg.mean() * rng.uniform(0.4, 0.8)                   # 抹平成常数（失去周期）
        else:           seg[rng.random(seg.shape) < 0.5] = 0.0                       # 随机日计量归零
    return x

n_t = int(N * theft_ratio)
X = np.concatenate([normal_users(N - n_t), inject_theft(normal_users(n_t))])
y = np.r_[np.zeros(N - n_t), np.ones(n_t)]
X = X / (X.mean(axis=(1, 2), keepdims=True) + 1e-6)   # 每户除以自身均值：只看"形状"，不看用量大小
idx = rng.permutation(N); X, y = X[idx], y[idx]
tr, te = slice(0, 3000), slice(3000, N)

# ---- 3. 2D 卷积特征：固定 3×3 卷积核 + ReLU + 全局均值/最大池化 ----
# 竖向核比较"不同周的同一天"（周期性是否被破坏），横向核比较"同一周相邻天"
K = {
    "周间差分":   np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], float),
    "日间差分":   np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], float),
    "周间一阶":   np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]], float),
    "局部均值":   np.ones((3, 3)) / 9,
}
def conv2d_valid(x, k):
    out = np.zeros((x.shape[0], x.shape[1] - 2, x.shape[2] - 2))
    for a in range(3):
        for b in range(3):
            out += k[a, b] * x[:, a:a + out.shape[1], b:b + out.shape[2]]
    return out
def cnn_features(x):
    f = []
    for k in K.values():
        for sign in (1, -1):                          # 正负两路 ReLU，相当于 8 个通道
            h = np.maximum(sign * conv2d_valid(x, k), 0)
            f += [h.mean(axis=(1, 2)), h.max(axis=(1, 2))]
    return np.stack(f, 1)

# ---- 4. 逻辑回归（梯度下降，可选类别加权） ----
def fit_logreg(F, y, weighted=False, lr=0.1, it=3000, l2=1e-3):
    mu, sd = F.mean(0), F.std(0) + 1e-9; Z = (F - mu) / sd
    w = np.where(y == 1, (1 - y.mean()) / y.mean(), 1.0) if weighted else np.ones_like(y)
    beta, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Z @ beta + b))); g = w * (p - y)
        beta -= lr * (Z.T @ g / len(y) + l2 * beta); b -= lr * g.mean()
    return lambda F2: 1 / (1 + np.exp(-(((F2 - mu) / sd) @ beta + b)))

def auc(score, y):                                    # Mann-Whitney 形式
    r = np.argsort(np.argsort(score)) + 1
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def report(name, score, y, thr=0.5):
    pred = score >= thr
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
    acc = (pred == y).mean(); rec = tp / y.sum(); prec = tp / max(tp + fp, 1)
    a = auc(score, y) if score.std() > 0 else 0.5
    print(f"{name:<26s} 准确率 {acc:6.3f}  召回 {rec:6.3f}  精确率 {prec:6.3f}  AUC {a:6.3f}")

yte = y[te]
print(f"测试集 {len(yte)} 户，其中窃电 {int(yte.sum())} 户（{yte.mean():.1%}）")
report("全判正常(哑分类器)", np.zeros(len(yte)), yte)
F1 = X.reshape(N, -1)
report("1D原始序列+逻辑回归", fit_logreg(F1[tr], y[tr])(F1[te]), yte)
F2 = cnn_features(X)
report("2D卷积特征+逻辑回归", fit_logreg(F2[tr], y[tr])(F2[te]), yte)
report("2D卷积特征+类别加权", fit_logreg(F2[tr], y[tr], weighted=True)(F2[te]), yte)
