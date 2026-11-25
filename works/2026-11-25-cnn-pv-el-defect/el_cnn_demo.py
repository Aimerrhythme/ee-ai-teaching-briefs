"""
光伏组件 EL（电致发光）电池片「正常 / 缺陷」二分类 —— 纯 numpy 课堂玩具
目的：用合成的 EL 电池片小图，对比三种分类器：
      ① 平均亮度阈值（"看整体暗不暗"）  ② 像素级逻辑回归（不懂"局部形状"）
      ③ 一层卷积 + ReLU + 全局最大池化 + 逻辑回归 的极简 CNN（手写反向传播）
并演示：单晶→多晶的"纹理域偏移"对模型的影响。
仅依赖 numpy；所有图像均为合成数据，不代表真实 EL 图像上的性能。
"""
import numpy as np

rng = np.random.default_rng(20261125)
S = 32                       # 电池片缩略图边长（像素）；真实 elpv 为 300×300
P_DEF = 0.30                 # 缺陷比例（elpv 中 p≥0.5 的样本约占 31%）

# ---------------- 1. 合成 EL 电池片 ----------------
yy, xx = np.mgrid[0:S, 0:S]

def base_cell(kind):
    """kind: 'mono' 单晶（均匀、四角圆形倒角暗区） / 'poly' 多晶（晶粒明暗斑块纹理）"""
    img = np.full((S, S), 0.75) + rng.normal(0, 0.045, (S, S))
    for gx in (S // 3, 2 * S // 3):                       # 两条主栅线：竖直细暗线
        img[:, gx] -= 0.25
    if kind == 'mono':
        r = np.minimum.reduce([np.hypot(yy - cy, xx - cx) for cy in (0, S - 1) for cx in (0, S - 1)])
        img[r < 4] = 0.1                                   # 单晶片四角倒角（非缺陷）
    else:
        for _ in range(rng.integers(6, 12)):              # 多晶晶粒：随机大斑块明暗起伏
            cy, cx, s = rng.uniform(0, S, 2).tolist() + [rng.uniform(3, 8)]
            img += rng.uniform(-0.18, 0.18) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
        for _ in range(rng.integers(2, 5)):               # 晶界：较淡的细暗线（非缺陷，但长得像隐裂）
            ang = rng.uniform(0, np.pi); c = rng.uniform(0, S, 2); L = rng.uniform(8, 20)
            for s_ in np.linspace(-L / 2, L / 2, 50):
                y, x = int(c[0] + s_ * np.sin(ang)), int(c[1] + s_ * np.cos(ang))
                if 0 <= y < S and 0 <= x < S:
                    img[y, x] *= 0.8
    img *= rng.uniform(0.85, 1.15)                         # 不同组件/电流下整体亮度不同
    return img

def add_defect(img):
    t = rng.integers(3)
    if t == 0:    # 隐裂：一条随机方向的细暗线
        ang = rng.uniform(0, np.pi); c = rng.uniform(8, S - 8, 2); L = rng.uniform(10, 22)
        for s in np.linspace(-L / 2, L / 2, 60):
            y, x = int(c[0] + s * np.sin(ang)), int(c[1] + s * np.cos(ang))
            if 0 <= y < S and 0 <= x < S:
                img[y, x] *= 0.5
    elif t == 1:  # 断栅（细栅中断）：一小段横向暗条
        y = rng.integers(4, S - 4); x = rng.integers(2, S - 10)
        img[y:y + 2, x:x + rng.integers(5, 9)] *= 0.55
    else:         # 失效区（裂片后电隔离）：一块小区域明显变暗
        y, x = rng.integers(2, S - 9, 2); h, w = rng.integers(4, 8, 2)
        img[y:y + h, x:x + w] *= 0.55
    return img

def make_set(n, kind_mix):
    X, y, k = [], [], []
    for _ in range(n):
        kind = 'mono' if rng.random() < kind_mix else 'poly'
        img = base_cell(kind); lab = rng.random() < P_DEF
        if lab: img = add_defect(img)
        X.append(np.clip(img, 0, 1)); y.append(int(lab)); k.append(kind)
    return np.array(X), np.array(y), np.array(k)

# ---------------- 2. 评价指标 ----------------
def auc(score, y):
    order = np.argsort(score); r = np.empty(len(y)); r[order] = np.arange(1, len(y) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)

def report(score, y, thr=0.5):
    p = score >= thr; tp = (p & (y == 1)).sum(); fp = (p & (y == 0)).sum(); fn = (~p & (y == 1)).sum()
    acc = (p == y).mean(); rec = tp / max(tp + fn, 1); pre = tp / max(tp + fp, 1)
    f1 = 2 * pre * rec / max(pre + rec, 1e-9)
    return dict(acc=acc, rec=rec, pre=pre, f1=f1, auc=auc(score, y))

sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def standardize(img):        # 逐图零均值/单位方差，抹掉"整体亮度"差异（EL 常用预处理）
    return (img - img.mean(axis=(1, 2), keepdims=True)) / (img.std(axis=(1, 2), keepdims=True) + 1e-6)

def fit_thr(score, y):       # 在训练集上选使 F1 最大的阈值（避免偷看测试集）
    cands = np.quantile(score, np.linspace(0.02, 0.98, 97))
    return max(cands, key=lambda t: report(score, y, t)['f1'])

# ---------------- 3. 模型 ----------------
def brightness_model(Xtr, ytr, Xte):
    s_tr, s_te = -Xtr.mean(axis=(1, 2)), -Xte.mean(axis=(1, 2))   # 越暗越可疑
    lo, hi = s_tr.min(), s_tr.max()
    return (s_te - lo) / (hi - lo), (s_tr - lo) / (hi - lo)

def logreg(Ftr, ytr, Fte, epochs=300, lr=0.05, l2=1e-3):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6; Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    w = np.zeros(Ftr.shape[1]); b = 0.0
    pw = np.where(ytr == 1, (1 - ytr.mean()) / ytr.mean(), 1.0)      # 类别加权，缓解不平衡
    for _ in range(epochs):
        g = (sig(Ftr @ w + b) - ytr) * pw
        w -= lr * (Ftr.T @ g / len(ytr) + l2 * w); b -= lr * g.mean()
    return sig(Fte @ w + b), sig(Ftr @ w + b)

def im2col(X, k):
    n = X.shape[0]; o = S - k + 1
    cols = np.lib.stride_tricks.sliding_window_view(X, (k, k), axis=(1, 2))  # n,o,o,k,k
    return cols.reshape(n, o * o, k * k)

def tiny_cnn(Xtr, ytr, Xte, F=8, k=5, epochs=60, lr=0.02, bs=64):
    """conv(F 个 k×k 卷积核) → ReLU → 全局最大池化 → 线性 → sigmoid；Adam 优化"""
    Ctr, Cte = im2col(standardize(Xtr), k), im2col(standardize(Xte), k)
    W = rng.normal(0, 0.2, (k * k, F)); cb = np.zeros(F); v = np.zeros(F); vb = 0.0
    params = [W, cb, v, np.array([vb])]; m1 = [np.zeros_like(p) for p in params]; m2 = [np.zeros_like(p) for p in params]
    pw_all = np.where(ytr == 1, (1 - ytr.mean()) / ytr.mean(), 1.0); t = 0
    def forward(C):
        A = C @ params[0] + params[1]                 # n, P, F
        R = np.maximum(A, 0); idx = R.argmax(1); h = np.take_along_axis(R, idx[:, None, :], 1)[:, 0]
        return A, idx, h, sig(h @ params[2] + params[3][0])
    for ep in range(epochs):
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]; C = Ctr[j]; A, idx, h, p = forward(C)
            g = (p - ytr[j]) * pw_all[j] / len(j)                       # dL/dlogit
            gv, gvb = h.T @ g, g.sum(); gh = np.outer(g, params[2])     # n,F
            gA_at = gh * (np.take_along_axis(A, idx[:, None, :], 1)[:, 0] > 0)
            Cm = C[np.arange(len(j))[:, None], idx]                      # n,F,kk：每个卷积核取最大响应处的图块
            gW = np.einsum('nfk,nf->kf', Cm, gA_at); gcb = gA_at.sum(0)
            t += 1
            for q, gq in enumerate([gW, gcb, gv, np.array([gvb])]):
                m1[q] = 0.9 * m1[q] + 0.1 * gq; m2[q] = 0.999 * m2[q] + 0.001 * gq ** 2
                params[q] -= lr * (m1[q] / (1 - 0.9 ** t)) / (np.sqrt(m2[q] / (1 - 0.999 ** t)) + 1e-8)
    return forward(Cte)[3], forward(Ctr)[3]

# ---------------- 4. 实验 ----------------
def run(name, fn, Xtr, ytr, Xte, yte, kte):
    s_te, s_tr = fn(Xtr, ytr, Xte); thr = fit_thr(s_tr, ytr)
    out = {'全部': report(s_te, yte, thr)}
    for kd, cn in (('mono', '单晶'), ('poly', '多晶')):
        m = kte == kd
        if m.sum(): out[cn] = report(s_te[m], yte[m], thr)
    return out

if __name__ == '__main__':
    Xtr, ytr, _ = make_set(1600, 0.5)
    Xte, yte, kte = make_set(600, 0.5)
    print(f"训练 {len(ytr)} 张（缺陷 {ytr.sum()}），测试 {len(yte)} 张（缺陷 {yte.sum()}，单晶 {(kte=='mono').sum()} / 多晶 {(kte=='poly').sum()}）")
    models = [
        ('平均亮度阈值', brightness_model),
        ('像素逻辑回归', lambda a, b, c: logreg(standardize(a).reshape(len(a), -1), b, standardize(c).reshape(len(c), -1))),
        ('极简 CNN', tiny_cnn),
    ]
    print("\n[实验 A] 单晶+多晶混合训练")
    for name, fn in models:
        r = run(name, fn, Xtr, ytr, Xte, yte, kte)
        a = r['全部']
        print(f"{name:8s} 全部 acc={a['acc']:.3f} rec={a['rec']:.3f} pre={a['pre']:.3f} F1={a['f1']:.3f} AUC={a['auc']:.3f} | "
              f"单晶 F1={r['单晶']['f1']:.3f} AUC={r['单晶']['auc']:.3f} | 多晶 F1={r['多晶']['f1']:.3f} AUC={r['多晶']['auc']:.3f}")
    print("\n[实验 B] 域偏移：只用单晶训练 vs 混合训练，均在多晶测试集上评价（极简 CNN）")
    Xm, ym, _ = make_set(1600, 1.0); Xp, yp, kp = make_set(600, 0.0)
    for lab, (A, B) in (('仅单晶训练', (Xm, ym)), ('混合训练', (Xtr, ytr))):
        r = run(lab, tiny_cnn, A, B, Xp, yp, kp)['全部']
        print(f"{lab:6s} → 多晶测试 acc={r['acc']:.3f} rec={r['rec']:.3f} pre={r['pre']:.3f} F1={r['f1']:.3f} AUC={r['auc']:.3f}")
