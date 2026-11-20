"""合成数据课堂演示：12 母线辐射状馈线，LinDistFlow 线性化近似。
对比 (a) 加权最小二乘 WLS  (b) 无图传播的逐节点岭回归 (K=0)  (c) 简化图卷积 SGC (K=2,3)。
仅依赖 numpy；所有数据随机合成，非真实电网。"""
import numpy as np
rng = np.random.default_rng(20261120)

# ---- 1. 拓扑：母线 0 为变电站（平衡节点），父节点数组定义辐射状树 ----
parent = [-1, 0, 1, 2, 3, 4, 2, 6, 7, 4, 9, 10]
n = len(parent)
r = rng.uniform(0.004, 0.012, n); x = rng.uniform(0.003, 0.010, n)  # 各支路(子节点->父节点) pu
r[0] = x[0] = 0.0
# 路径矩阵：C[j,k]=1 表示支路 k 位于 0->j 路径上；LinDistFlow: v_j = v0 - 2*sum_k (r_k P_k + x_k Q_k)
C = np.zeros((n, n))
for j in range(1, n):
    k = j
    while k > 0:
        C[j, k] = 1; k = parent[k]
D = C.T  # 支路 k 的下游功率 = sum_{j: k 在 0->j 路径上} p_j
R = 2 * C @ np.diag(r) @ D; X = 2 * C @ np.diag(x) @ D   # v = v0 - R p - X q   (v 为电压幅值平方)

# ---- 2. 合成样本：负荷 = 典型值 × 随机日内系数 × 个体扰动 ----
base_p = np.r_[0, rng.uniform(0.02, 0.08, n - 1)]; base_q = 0.4 * base_p
def sample(N):
    s = rng.uniform(0.3, 1.3, (N, 1)) * rng.normal(1, 0.15, (N, n))
    p = base_p * s; q = base_q * s
    v0 = rng.normal(1.02, 0.005, (N, 1)) ** 2
    v = v0 - p @ R.T - q @ X.T
    return p, q, np.sqrt(v)          # 真实状态：电压幅值 |V| (pu)

meter = [0, 5, 11]                    # 仅 3 个母线装电压量测（含变电站）
sig_v, sig_pseudo = 0.002, 0.30       # 电压表 0.2%，伪量测(按典型负荷)相对误差 30%
def measure(p, q, V):
    N = len(V)
    Vm = V[:, meter] + rng.normal(0, sig_v, (N, len(meter)))
    pp = np.tile(base_p, (N, 1)) * rng.normal(1, sig_pseudo, (N, n))  # 伪量测只知典型值
    qq = 0.4 * pp
    return Vm, pp, qq

# ---- 3. WLS：状态 z=[v0, p, q]，量测 = 电压平方 + 伪量测；线性模型一次求解 ----
H_v = np.hstack([np.ones((n, 1)), -R, -X])[meter]
H_p = np.hstack([np.zeros((n, 1)), np.eye(n), np.zeros((n, n))])
H_q = np.hstack([np.zeros((n, 1)), np.zeros((n, n)), np.eye(n)])
H = np.vstack([H_v, H_p[1:], H_q[1:]])
def wls(Vm, pp, qq):
    out = []
    for i in range(len(Vm)):
        zv = Vm[i] ** 2
        sp = np.maximum(sig_pseudo * base_p[1:], 1e-4)
        w = np.r_[1 / (2 * sig_v) ** 2 * np.ones(len(meter)), 1 / sp ** 2, 1 / (0.4 * sp) ** 2]
        zz = np.r_[zv, pp[i, 1:], qq[i, 1:]]
        Wm = np.diag(w)
        st = np.linalg.solve(H.T @ Wm @ H + 1e-9 * np.eye(H.shape[1]), H.T @ Wm @ zz)
        out.append(np.sqrt(st[0] - R @ st[1:n + 1] - X @ st[n + 1:]))
    return np.array(out)

# ---- 4. 图特征：节点特征 [电压量测或0, 是否有表, 伪p, 伪q, 到变电站电气距离]，SGC = Â^K X 后岭回归 ----
A = np.zeros((n, n))
for j in range(1, n): A[j, parent[j]] = A[parent[j], j] = 1
At = A + np.eye(n); d = At.sum(1); Ahat = At / np.sqrt(np.outer(d, d))
elec = C @ r
def node_feat(Vm, pp, qq):
    N = len(Vm); F = np.zeros((N, n, 5))
    F[:, meter, 0] = Vm; F[:, meter, 1] = 1
    F[:, :, 2] = pp; F[:, :, 3] = qq; F[:, :, 4] = elec
    return F
def propagate(F, K):
    H_ = [F]
    for _ in range(K): H_.append(np.einsum('ij,njf->nif', Ahat, H_[-1]))
    G = np.concatenate(H_, axis=2)                 # 拼接 0..K 跳特征
    # 全图汇总：广播已量测电压的均值，模拟"读出层"
    glob = np.repeat(F[:, meter, 0].mean(1, keepdims=True)[:, None, :], n, axis=1)
    return np.concatenate([G, glob, np.ones_like(glob)], axis=2)
def ridge_fit(G, Y, lam=1e-4):
    Z = G.reshape(-1, G.shape[2]); y = Y.reshape(-1)
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)

ptr, qtr, Vtr = sample(3000); pte, qte, Vte = sample(500)
Mtr = measure(ptr, qtr, Vtr); Mte = measure(pte, qte, Vte)
Ftr = node_feat(*Mtr); Fte = node_feat(*Mte)
unmet = [i for i in range(n) if i not in meter]
def rmse(Vhat): return 1000 * np.sqrt(np.mean((Vhat[:, unmet] - Vte[:, unmet]) ** 2))
print(f"测试样本 {len(Vte)}，未装表母线 {len(unmet)} 个，误差单位 1e-3 pu")
print(f"WLS（线性化模型 + 伪量测）          RMSE = {rmse(wls(*Mte)):.3f}")
for K in [0, 1, 2, 3]:
    w = ridge_fit(propagate(Ftr, K), Vtr)
    Vhat = propagate(Fte, K) @ w
    tag = "逐节点岭回归（无图传播）" if K == 0 else f"SGC 图传播 K={K}"
    print(f"{tag:<28} RMSE = {rmse(Vhat):.3f}")
