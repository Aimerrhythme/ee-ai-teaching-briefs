"""
无人机巡检绝缘子「掉串（缺片）」检测 —— 纯 numpy 课堂玩具
目的：不训练 YOLO，而是用一个"手工检测器"把目标检测的完整评价链跑通：
      合成图像 + 标注框 → 候选框 + 置信度 → NMS → IoU 匹配 → P/R 曲线 → AP@0.5 与 AP@[.5:.95]
仅依赖 numpy。所有图像均为合成数据，不代表真实巡检图片上的性能。
"""
import numpy as np

rng = np.random.default_rng(20261124)
H, Wd = 64, 256          # 单张"航拍切片"尺寸（像素）
N_IMG, P_DEF = 400, 0.4  # 图像数、含缺陷图像比例

# ---------------- 1. 合成数据：一串伞裙（高斯亮斑）+ 背景杂波 ----------------
def make_image():
    img = rng.normal(0.25, 0.06, (H, Wd))
    yy, xx = np.mgrid[0:H, 0:Wd]
    # 背景干扰：随机几块亮斑（模拟金具、树木、反光）
    for _ in range(rng.integers(1, 4)):
        cy, cx, s = rng.uniform(0, H), rng.uniform(0, Wd), rng.uniform(3, 8)
        img += rng.uniform(0.2, 0.5) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    row = rng.integers(20, H - 20)              # 串的纵向位置
    pitch = rng.uniform(14, 20)                 # 片间距（像素）
    n = int((Wd - 30) // pitch)
    x0 = rng.uniform(10, Wd - 10 - (n - 1) * pitch)
    xs = x0 + pitch * np.arange(n)
    gt = []
    miss = rng.integers(2, n - 2) if rng.random() < P_DEF else -1   # 首尾两片不缺，便于讲解
    r = pitch * 0.22                            # 伞裙横向半宽（σ）
    for k, cx in enumerate(xs):
        if k == miss:
            gt.append([cx - pitch / 2, row - 3.8 * r, cx + pitch / 2, row + 3.8 * r])  # 缺陷框 = 缺片位置
            continue
        a = rng.uniform(0.6, 0.9)
        img += a * np.exp(-((xx - cx) ** 2 / (2 * r * r) + (yy - row) ** 2 / (2 * (2 * r) ** 2)))
    if rng.random() < 0.15:                     # 15% 图像有局部遮挡/阴影 → 易产生误检
        c = rng.integers(0, n)
        img[:, int(xs[c] - pitch / 3):int(xs[c] + pitch / 3)] *= rng.uniform(0.35, 0.6)
    return img, np.array(gt).reshape(-1, 4)

data = [make_image() for _ in range(N_IMG)]

# ---------------- 2. 手工检测器：找串 → 估片距 → 在"该有亮斑却暗"的位置出框 ----------------
def detect(img):
    rowprof = img.mean(1); row = int(np.argmax(np.convolve(rowprof, np.ones(9) / 9, "same")))
    prof = img[max(row - 3, 0):row + 4].mean(0)
    k = np.exp(-np.arange(-6, 7) ** 2 / (2 * 2.5 ** 2)); k /= k.sum()
    prof = np.convolve(prof - np.median(prof), k, "same")
    # 用自相关估计片间距（周期）
    p = prof - np.convolve(prof, np.ones(31) / 31, "same")   # 去掉缓变背景再求自相关
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    pitch = 12 + int(np.argmax(ac[12:24]))
    # 找峰（各片中心）：半个片距窗口内的局部最大，且高于"典型峰高"的一半
    hw = pitch // 2
    cand = [i for i in range(hw, Wd - hw) if prof[i] == prof[i - hw:i + hw + 1].max()]
    if len(cand) < 3:
        return np.zeros((0, 5))
    ref = np.median(prof[cand])
    pk = np.array([i for i in cand if prof[i] > 0.5 * ref])
    if len(pk) < 3:
        return np.zeros((0, 5))
    ref = np.median(prof[pk])
    boxes = []
    if len(pk) < 3:
        return np.zeros((0, 5))
    # 相邻峰间距约为 2×pitch → 中间缺了一片
    for a, b in zip(pk[:-1], pk[1:]):
        g = (b - a) / pitch
        for m in range(1, int(round(g))):
            cx = a + m * (b - a) / round(g)
            depth = np.clip(1 - max(prof[int(cx)], 0) / (ref + 1e-6), 0, 1)
            score = depth * np.exp(-abs(g - round(g)) * 2)
            r = pitch * 0.22
            boxes.append([cx - pitch / 2, row - 3.8 * r, cx + pitch / 2, row + 3.8 * r, score])
            # 再加一个略偏的重复框，模拟检测器多尺度/多锚框产生的冗余输出
            j = rng.normal(0, 2)
            boxes.append([cx - pitch / 2 + j, row - 3.8 * r, cx + pitch / 2 + j, row + 3.8 * r, score * 0.9])
    return np.array(boxes).reshape(-1, 5)

def iou(a, b):
    ix = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = ix * iy
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)

def nms(d, thr=0.5):
    d = d[np.argsort(-d[:, 4])]; keep = []
    while len(d):
        keep.append(d[0]); d = d[1:][iou(d[0], d[1:]) < thr] if len(d) > 1 else d[1:]
    return np.array(keep).reshape(-1, 5)

# ---------------- 3. 评价：IoU 匹配 → P/R → AP（全点插值，同 VOC2010+/COCO 思路） ----------------
def evaluate(dets, gts, iou_thr):
    recs, n_gt = [], sum(len(g) for g in gts)
    for i, (d, g) in enumerate(zip(dets, gts)):
        used = np.zeros(len(g), bool)
        for b in d[np.argsort(-d[:, 4])]:
            tp = 0
            if len(g):
                o = iou(b, g); j = int(np.argmax(o))
                if o[j] >= iou_thr and not used[j]:
                    used[j] = True; tp = 1
            recs.append((b[4], tp))
    recs.sort(key=lambda t: -t[0])
    tp = np.cumsum([t for _, t in recs]); fp = np.cumsum([1 - t for _, t in recs])
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, 1)
    mrec = np.r_[0, rec, 1]; mpre = np.r_[0, prec, 0]
    for k in range(len(mpre) - 2, -1, -1): mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap, recs, n_gt

gts = [g for _, g in data]
raw = [detect(img) for img, _ in data]
post = [nms(d) for d in raw]
print(f"图像 {N_IMG} 张，含缺陷 {sum(len(g) > 0 for g in gts)} 张，缺陷框共 {sum(len(g) for g in gts)} 个")
print(f"NMS 前候选框 {sum(len(d) for d in raw)} 个 → NMS 后 {sum(len(d) for d in post)} 个")
for name, D in [("无 NMS", raw), ("NMS(IoU 0.5)", post)]:
    ap50 = evaluate(D, gts, 0.5)[0]
    ap = np.mean([evaluate(D, gts, t)[0] for t in np.arange(0.5, 0.96, 0.05)])
    print(f"{name:<14s} AP@0.5 = {ap50:.3f}   AP@[.5:.95] = {ap:.3f}")

# 置信度阈值 → 召回/精确率/漏检数（运维视角：漏检一个缺片 vs 多派一次人工复核）
_, recs, n_gt = evaluate(post, gts, 0.5)
print("置信度阈值  检出框  TP  FP  漏检  召回   精确率")
for th in (0.5, 0.7, 0.8, 0.9):
    sel = [t for s, t in recs if s >= th]; tp = sum(sel); fp = len(sel) - tp
    print(f"   {th:.1f}       {len(sel):4d}  {tp:3d} {fp:3d}  {n_gt - tp:3d}   {tp / n_gt:.3f}  {tp / max(len(sel), 1):.3f}")
