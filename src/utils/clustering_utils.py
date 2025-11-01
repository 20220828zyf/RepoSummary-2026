import numpy as np
import igraph as ig
import leidenalg as la
from sklearn.metrics import adjusted_rand_score, silhouette_score
from typing import List, Tuple

def merge_tiny_clusters(labels, W, min_size=2, min_accept_sim=0.0, penalize_large=True, max_passes=5):
    """
    后处理：迭代把规模 < min_size 的簇并入最相似簇。
    参数:
        min_accept_sim: 若最高平均相似度仍低于该值则放弃合并
        penalize_large: 是否对巨大目标簇做轻量惩罚，避免全部吸入同一大簇
        max_passes: 最多迭代轮次
    """
    labels = np.array(labels)
    n_pass = 0
    changed_any = True
    while changed_any and n_pass < max_passes:
        changed_any = False
        n_pass += 1
        uniq = np.unique(labels)
        for c in uniq:
            idx = np.where(labels == c)[0]
            if idx.size >= min_size:
                continue
            best_target, best_score = None, -1.0
            for d in np.unique(labels):
                if d == c:
                    continue
                jdx = np.where(labels == d)[0]
                if jdx.size == 0:
                    continue
                score = W[np.ix_(idx, jdx)].mean()
                if penalize_large:
                    score = score / (1.0 + np.log1p(jdx.size))  # 轻度规模惩罚
                if score > best_score:
                    best_score, best_target = score, d
            if best_target is not None and best_score >= min_accept_sim:
                print(f" Merging tiny cluster {c} (size {idx.size}) into {best_target} (sim={best_score:.4f}) ")
                labels[idx] = best_target
                changed_any = True
    return labels

def symmetrize_zero_diag(M):
    M = np.asarray(M, dtype=float)
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    M[M < 0] = 0.0
    return M

def normalize_01(M, eps=1e-12):
    M = symmetrize_zero_diag(M)
    mmin, mmax = M.min(), M.max()
    if mmax - mmin < eps:
        return np.zeros_like(M)
    M = (M - mmin) / (mmax - mmin)
    np.fill_diagonal(M, 0.0)
    return M

def fuse_weights(A, S, a=0.5):
    A_ = normalize_01(A)
    S_ = normalize_01(S)
    W = a * S_ + (1 - a) * A_
    return symmetrize_zero_diag(W)

def sparsify_knn(W, k=None):
    """
    kNN 稀疏化：每行仅保留 top-k 边；返回对称矩阵（取 max 保留）。
    k=None 表示不启用。
    """
    if k is None or k <= 0:
        return symmetrize_zero_diag(W)
    n = W.shape[0]
    W_keep = np.zeros_like(W)
    for i in range(n):
        row = W[i]
        # 排除对角线
        idx = np.argpartition(-row, min(k, n-1))[:min(k, n-1)]
        idx = idx[idx != i]
        W_keep[i, idx] = row[idx]
    # 对称化：保留两者中的较大者
    W_sym = np.maximum(W_keep, W_keep.T)
    np.fill_diagonal(W_sym, 0.0)
    return W_sym

def sparsify_threshold(W, tau=None):
    """
    按阈值稀疏化：权重 < tau 的边置 0。tau=None 表示不启用。
    """
    if tau is None or tau <= 0:
        return symmetrize_zero_diag(W)
    W2 = W.copy()
    W2[W2 < tau] = 0.0
    return symmetrize_zero_diag(W2)

def build_igraph_from_W(W):
    W = symmetrize_zero_diag(W)
    return ig.Graph.Weighted_Adjacency(W.tolist(), mode=ig.ADJ_UNDIRECTED, attr='weight', loops=False)

def run_leiden(W, gamma, seed=0, objective="CPM", n_iterations=-1):
    G = build_igraph_from_W(W)
    if objective.upper() == "CPM":
        part = la.find_partition(
            G, la.CPMVertexPartition,
            weights='weight',
            resolution_parameter=gamma,
            seed=seed,
            n_iterations=n_iterations
        )
    elif objective.upper() in ("RB", "RBCONFIGURATION", "RBCONFIG"):
        part = la.find_partition(
            G, la.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=gamma,
            seed=seed,
            n_iterations=n_iterations
        )
    else:
        raise ValueError("objective must be 'CPM' or 'RBConfiguration'")
    labels = np.array(part.membership, dtype=int)
    quality = float(part.quality())
    return labels, quality

def cluster_size_stats(labels):
    labels = np.asarray(labels)
    vals, counts = np.unique(labels, return_counts=True)
    sizes = counts
    n = labels.size
    n_clusters = len(sizes)
    singleton_frac = (sizes == 1).sum() / n_clusters if n_clusters > 0 else 0.0
    tiny_frac = (sizes <= 2).sum() / n_clusters if n_clusters > 0 else 0.0
    cv = (np.std(sizes) / (np.mean(sizes) + 1e-12)) if n_clusters > 0 else 0.0
    return dict(n_clusters=n_clusters, sizes=sizes, singleton_frac=singleton_frac, tiny_frac=tiny_frac, size_cv=cv)

def cluster_intra_inter_separation(W, labels):
    """
    返回整体分离度：平均(簇内均值 - 簇外均值)，按簇大小加权。
    值越大越好；范围大致在 [-1,1]。
    """
    labels = np.asarray(labels)
    n = len(labels)
    total = 0.0
    weight_sum = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        m = idx.size
        if m <= 1:
            continue
        comp = np.setdiff1d(np.arange(n), idx, assume_unique=True)
        sub = W[np.ix_(idx, idx)]
        intra = sub.sum() / (m * (m - 1))  # 平均非对角簇内权重（双计数/分母匹配）
        inter = 0.0
        if comp.size > 0:
            inter = W[np.ix_(idx, comp)].mean()
        total += (intra - inter) * m
        weight_sum += m
    return float(total / (weight_sum + 1e-12)) if weight_sum > 0 else 0.0

def cluster_density_std(W, labels):
    """
    计算各簇平均内部密度的标准差（越小说明"均衡"）。
    """
    labels = np.asarray(labels)
    densities = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        m = idx.size
        if m <= 1:
            continue
        sub = W[np.ix_(idx, idx)]
        density = sub.sum() / (m * (m - 1))
        densities.append(density)
    return float(np.std(densities)) if densities else float('inf')

def coassociation_matrix(partitions):
    """
    共识矩阵：C[i,j] = 在多少比例的划分中 i 和 j 同簇。
    注意：O(n^2) 内存，对大图谨慎使用。
    """
    n = len(partitions[0])
    C = np.zeros((n, n), dtype=np.float32)
    for labels in partitions:
        labels = np.asarray(labels)
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            if idx.size == 0:
                continue
            C[np.ix_(idx, idx)] += 1.0
    C /= len(partitions)
    np.fill_diagonal(C, 1.0)
    return C

def consensus_labels_from_coassoc(C, tau=0.6, gamma=0.1, seed=0):
    Wc = C.copy()
    Wc[Wc < tau] = 0.0
    np.fill_diagonal(Wc, 0.0)
    labels, _ = run_leiden(Wc, gamma=gamma, seed=seed, objective="CPM")
    return labels

def pairwise_mean_ari(partitions):
    if len(partitions) <= 1:
        return 1.0
    ARIs = []
    for i in range(len(partitions)):
        for j in range(i+1, len(partitions)):
            ARIs.append(adjusted_rand_score(partitions[i], partitions[j]))
    return float(np.mean(ARIs))

def safe_silhouette_from_similarity(W, labels, use_silhouette=False, sample_size=None, rng=42):
    """
    可选计算 silhouette（使用距离 1-W）。默认关闭以避免误用非度量距离。
    """
    if not use_silhouette:
        return None
    n = W.shape[0]
    dist = 1.0 - np.clip(W, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    labels = np.asarray(labels)
    # 至少要有 2 个簇且每个簇 >= 2
    uniq, counts = np.unique(labels, return_counts=True)
    if uniq.size < 2 or np.any(counts < 2):
        return None
    if sample_size is not None and sample_size < n:
        rng = np.random.default_rng(rng)
        idx = rng.choice(n, size=sample_size, replace=False)
        dist = dist[np.ix_(idx, idx)]
        labels = labels[idx]
    try:
        return float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:
        return None

