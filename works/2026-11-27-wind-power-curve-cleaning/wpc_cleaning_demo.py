"""风机功率曲线异常点识别与清洗 —— 纯 numpy 合成数据课堂演示（2026-11-27）

合成 1 台 2 MW 机组一年的 10 min SCADA 数据（风速 v, 有功 P），注入三类异常
（分类沿用 Long 等 2020 的"负值点 / 分散点 / 堆积点"说法）：
  负值点 negative ：v > 切入风速但 P < 0（停机自耗电 / 变送器零漂）
  堆积点 stacked  ：限电平台（P 被压在 ~1000 kW 水平带）+ 停机（中高风速下 P≈0）
  分散点 scattered：风速计 / 功率变送器故障造成的散乱点
比较 5 种清洗方法的检出效果，以及清洗后"分箱平均功率曲线"和年发电量（AEP）估计误差；
再跑一个"重度限电"场景看分箱统计法何时失效。仅依赖 numpy，所有数字只用于讲概念。
"""
import numpy as np

PR, VCI, VR, VCO = 2000.0, 3.0, 12.0, 25.0          # 额定功率 kW / 切入 / 额定 / 切出 m/s
W = 0.5                                              # 分箱宽度 0.5 m/s（IEC 61400-12-1 分箱法的箱宽）
N = 52560                                            # 365 天 × 144 个 10 min 点
centers = (np.arange(int(VCO / W)) + 0.5) * W


def true_curve(v):
    v = np.asarray(v, float)
    x = np.clip((v - VCI) / (VR - VCI), 0, 1)
    p = PR * (3 * x**2 - 2 * x**3) ** 1.15           # 平滑 S 形，近似三次方区 + 额定区
    p[(v < VCI) | (v >= VCO)] = 0.0
    return p


# ---------- 1. 合成数据 ----------
def synth(n_curt, seed=20261127):
    rng = np.random.default_rng(seed)
    v = np.clip(rng.weibull(2.0, N) * 8.0, 0, 27)    # Weibull k=2, c=8 m/s（逐点独立，忽略时间相关）
    p = true_curve(v) + rng.normal(0, 1, N) * (25 + 0.04 * true_curve(v))
    p = np.clip(p, 0, PR * 1.02)                     # 正常点不低于 0
    label = np.zeros(N, int)                         # 0 正常 1 负值 2 堆积 3 分散

    def pick(mask, k):
        idx = np.flatnonzero(mask & (label == 0))
        return rng.choice(idx, size=min(k, idx.size), replace=False)

    i = pick(v > 4, 700); p[i] = rng.uniform(-60, -5, i.size); label[i] = 1
    for s in rng.choice(N - 200, n_curt, replace=False):          # 限电段：连续 30~150 个点
        seg = np.arange(s, s + rng.integers(30, 150))
        seg = seg[(true_curve(v[seg]) > 1150) & (label[seg] == 0)]
        p[seg] = 1000 + rng.normal(0, 30, seg.size); label[seg] = 2
    i = pick((v > 6) & (v < VCO), 1200); p[i] = rng.uniform(0, 15, i.size); label[i] = 2   # 停机
    i = pick(v > 2, 900)
    p[i] = np.clip(true_curve(v[i]) * rng.uniform(0.2, 0.75, i.size) + rng.normal(0, 60, i.size), 0, PR)
    label[i] = 3
    return v, p, label


# ---------- 2. 清洗方法 ----------
def pre_clean(v, p):
    """规则预清洗：负值点 & 中高风速零功率（停机）。"""
    return ((v > VCI) & (p < 0)) | ((v > 5) & (v < VCO) & (p < 20))


def bin_filter(v, p, mask0, center="mean", spread="std", k=2.0, iters=1, min_n=10):
    """分箱统计法（思路同 OpenOA filters.bin_filter）：每个 0.5 m/s 箱内偏离中心超过 k 倍离散度即判异常。"""
    bins = np.floor(v / W).astype(int); flag = mask0.copy()
    for _ in range(iters):
        new = flag.copy()
        for b in np.unique(bins):
            m = (bins == b) & ~flag
            if m.sum() < min_n or b * W < VCI:
                continue                                # 样本太少的箱：统计量不可靠，跳过
            x = p[m]
            c = x.mean() if center == "mean" else np.median(x)
            s = x.std() if spread == "std" else 1.4826 * np.median(np.abs(x - np.median(x)))
            new[m] |= np.abs(x - c) > k * max(s, 10.0)
        flag = new
    return flag


def iqr_filter(v, p, mask0, k=1.5, min_n=10):
    """分箱四分位法：箱内超出 [Q1-k·IQR, Q3+k·IQR] 判异常。"""
    bins = np.floor(v / W).astype(int); flag = mask0.copy()
    for b in np.unique(bins):
        m = (bins == b) & ~flag
        if m.sum() < min_n or b * W < VCI:
            continue
        q1, q3 = np.percentile(p[m], [25, 75]); r = q3 - q1
        flag[m] |= (p[m] < q1 - k * r) | (p[m] > q3 + k * r)
    return flag


def _shift_stack(img, r):
    H, Wd = img.shape; pad = np.pad(img, r, constant_values=False)
    return [pad[r + di:r + di + H, r + dj:r + dj + Wd] for di in range(-r, r + 1) for dj in range(-r, r + 1)]

def erode(img, r):  return np.logical_and.reduce(_shift_stack(img, r))
def dilate(img, r): return np.logical_or.reduce(_shift_stack(img, r))


def image_filter(v, p, mask0, nx=135, ny=110, frac=0.08, max_jump=8):
    """图像法简化版（思路参照 Long 等 2020，非原文复现）：
    散点→二值图（每列密度 ≥ 该列峰值的 frac 记 1）→3×3 开运算去掉细碎散点/细横带→
    从样本最多的列出发向两侧"跟踪"与主体相连的连通段→像素映射回点，主体外的点判异常。"""
    gx = np.clip((v / 27 * nx).astype(int), 0, nx - 1)
    gy = np.clip(((p + 50) / (PR * 1.05 + 50) * ny).astype(int), 0, ny - 1)
    ok = ~mask0
    H = np.zeros((ny, nx)); np.add.at(H, (gy[ok], gx[ok]), 1)
    img = (H >= frac * np.maximum(H.max(0, keepdims=True), 1)) & (H > 0)
    opened = dilate(erode(img, 1), 1)
    sparse = H.sum(0) < 150                         # 高风速稀疏列：开运算会把主体也腐蚀掉，保留原二值图
    img = np.where(sparse[None, :], img, opened)
    j_lo, j_hi = int(VCI / 27 * nx) + 1, int(VCO / 27 * nx)

    def segs_of(j):
        rows = np.flatnonzero(img[:, j])
        return np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if rows.size else []

    main = np.zeros_like(img)
    j0 = j_lo + int(np.argmax(H[:, j_lo:j_hi].sum(0)))
    s0 = max(segs_of(j0), key=lambda s: H[s, j0].sum()); main[s0, j0] = True
    for step in (1, -1):
        prev, j = np.median(s0), j0 + step
        while j_lo <= j < j_hi:
            ss = segs_of(j)
            if ss:
                s = min(ss, key=lambda s: abs(np.median(s) - prev))
                if abs(np.median(s) - prev) <= max_jump:
                    main[s, j] = True; prev = np.median(s)
            j += step
    main = dilate(main, 1)
    return mask0 | (((v > VCI) & (v < VCO)) & ~main[gy, gx])


# ---------- 3. 评价 ----------
def aep(pc, k=2.0, c=8.0):
    """Weibull 风况下按分箱功率曲线估算年发电量 MWh（空箱按 0 计）。"""
    lo, hi = centers - W / 2, centers + W / 2
    prob = np.exp(-(lo / c) ** k) - np.exp(-(hi / c) ** k)
    return np.nansum(np.nan_to_num(pc) * prob) * 8760 / 1000


def evaluate(n_curt, title):
    v, p, label = synth(n_curt)
    bad = label > 0; bins = np.floor(v / W).astype(int)
    print(f"\n===== {title}（限电段 {n_curt} 个）=====")
    print(f"样本 {N}，异常 {bad.sum()} ({bad.mean():.1%})：负值 {(label==1).sum()}，"
          f"堆积 {(label==2).sum()}（其中限电 {(label==2).sum()-1200}、停机 1200），分散 {(label==3).sum()}")
    pre = pre_clean(v, p)
    methods = {
        "不清洗(原始)": np.zeros(N, bool),
        "仅规则预清洗": pre,
        "分箱 均值±2σ(1轮)": bin_filter(v, p, pre, "mean", "std", 2.0, 1),
        "分箱 中位数±3MAD(3轮)": bin_filter(v, p, pre, "median", "mad", 3.0, 3),
        "分箱 IQR×1.5": iqr_filter(v, p, pre),
        "图像/形态学(简化)": image_filter(v, p, pre),
    }
    ref_pc = true_curve(centers); ref_aep = aep(ref_pc)
    dense = (centers > VCI) & (centers < 15); allr = (centers > VCI) & (centers < VCO)
    print(f"真实功率曲线对应 AEP = {ref_aep:.0f} MWh（Weibull k=2, c=8 m/s）")
    print(f"{'方法':<20}{'剔除率':>7}{'精确率':>7}{'召回率':>7}{'F1':>7}{'正常误删':>8}"
          f"{'RMSE3-15':>9}{'RMSE3-25':>9}{'AEP偏差':>8}  召回(负值/堆积/分散)")
    for name, flag in methods.items():
        tp = (flag & bad).sum(); fp = (flag & ~bad).sum(); fn = (~flag & bad).sum()
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        keep = ~flag
        pc = np.array([p[keep & (bins == b)].mean() if (keep & (bins == b)).sum() >= 3 else np.nan
                       for b in range(centers.size)])
        e = pc - ref_pc
        r1 = np.sqrt(np.nanmean(e[dense] ** 2)); r2 = np.sqrt(np.nanmean(e[allr] ** 2))
        dA = (aep(pc) - ref_aep) / ref_aep
        rc = "/".join(f"{(flag & (label==c)).sum()/(label==c).sum():.2f}" for c in (1, 2, 3))
        print(f"{name:<20}{flag.mean():>7.1%}{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}{fp/(~bad).sum():>8.2%}"
              f"{r1:>9.1f}{r2:>9.1f}{dA:>+8.1%}  {rc}")


if __name__ == "__main__":
    evaluate(40, "场景 A：常规")
    evaluate(250, "场景 B：重度限电")
