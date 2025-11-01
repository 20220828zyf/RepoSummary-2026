import numpy as np
import igraph as ig
import leidenalg as la
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import adjusted_rand_score, silhouette_score
from model.models import method_Cluster, Feature

# 全局变量，用于存储函数调用矩阵
func_adj_matrix = None

def set_func_adj_matrix(matrix):
    """设置全局函数调用矩阵"""
    global func_adj_matrix
    func_adj_matrix = matrix

def compute_function_similarity_matrix(method_cluster: method_Cluster):
    txt_vectors = [func.func_txt_vector for func in method_cluster.cluster_func_list]
    if len(txt_vectors) == 0:
        return np.zeros((0, 0))

    vecs = [np.asarray(v, dtype=float) for v in txt_vectors]
    lengths = [v.size for v in vecs]
    maxlen = max(lengths)

    if any(l != maxlen for l in lengths):
        vecs = [np.pad(v, (0, maxlen - v.size), mode='constant') if v.size < maxlen else v for v in vecs]

    X = np.vstack(vecs)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms

    similarity_matrix = np.dot(Xn, Xn.T)
    similarity_matrix = (similarity_matrix + 1.0) / 2.0
    return similarity_matrix

def compute_link(method_cluster):
    global func_adj_matrix
    if func_adj_matrix is None:
        return np.zeros((len(method_cluster.cluster_func_list), len(method_cluster.cluster_func_list)))
    
    link_matrix = np.zeros((len(method_cluster.cluster_func_list), len(method_cluster.cluster_func_list)))
    i = 0 
    for function in method_cluster.cluster_func_list:
        j = 0
        for other_function in method_cluster.cluster_func_list:
            if i == j:
                j += 1
                continue
            if func_adj_matrix[function.func_id][other_function.func_id] != 0:
                link_matrix[i][j] = 1
                link_matrix[j][i] = 1
            j += 1
        i += 1
    return link_matrix

def symmetrize_zero_diag(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return M
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    M[M < 0] = 0.0
    return M

def normalize_01(M: np.ndarray, eps=1e-12) -> np.ndarray:
    M = symmetrize_zero_diag(M)
    if M.size == 0:
        return np.zeros_like(M)
    mmin, mmax = M.min(), M.max()
    if not np.isfinite(mmin) or not np.isfinite(mmax) or (mmax - mmin) < eps:
        return np.zeros_like(M)
    M = (M - mmin) / (mmax - mmin)
    np.fill_diagonal(M, 0.0)
    return M

def fuse_weights_func(S: np.ndarray, L: np.ndarray, a: float) -> np.ndarray:
    S_ = normalize_01(S)
    L_ = normalize_01(L)
    W = a * S_ + (1 - a) * L_
    return symmetrize_zero_diag(W)

def sparsify_knn(W: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    if k is None or k <= 0:
        return symmetrize_zero_diag(W)
    n = W.shape[0]
    W_keep = np.zeros_like(W)
    for i in range(n):
        row = W[i]
        kk = min(k, n - 1)
        idx = np.argpartition(-row, kk)[:kk]
        idx = idx[idx != i]
        W_keep[i, idx] = row[idx]
    W_sym = np.maximum(W_keep, W_keep.T)
    np.fill_diagonal(W_sym, 0.0)
    return W_sym

def sparsify_threshold(W: np.ndarray, tau: Optional[float] = None) -> np.ndarray:
    if tau is None or tau <= 0:
        return symmetrize_zero_diag(W)
    W2 = W.copy()
    W2[W2 < tau] = 0.0
    return symmetrize_zero_diag(W2)

def build_graph(W: np.ndarray) -> ig.Graph:
    return ig.Graph.Weighted_Adjacency(W.tolist(), mode=ig.ADJ_UNDIRECTED, attr='weight', loops=False)

def run_leiden_on_W(W: np.ndarray, gamma: float, seed: int, objective: str = "CPM", n_iterations: int = -1) -> Tuple[np.ndarray, float]:
    G = build_graph(W)
    if objective.upper() == "CPM":
        part = la.find_partition(G, la.CPMVertexPartition, weights='weight', resolution_parameter=gamma, seed=seed, n_iterations=n_iterations)
    elif objective.upper() in ("RB", "RBCONFIGURATION", "RBCONFIG"):
        part = la.find_partition(G, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=gamma, seed=seed, n_iterations=n_iterations)
    else:
        raise ValueError("objective must be 'CPM' or 'RBConfiguration'")
    labels = np.array(part.membership, dtype=int)
    return labels, float(part.quality())

def cluster_size_stats(labels: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels)
    vals, counts = np.unique(labels, return_counts=True)
    sizes = counts
    n_clusters = len(sizes)
    singleton_frac = (sizes == 1).sum() / n_clusters if n_clusters > 0 else 0.0
    tiny_frac = (sizes <= 2).sum() / n_clusters if n_clusters > 0 else 0.0
    size_cv = (np.std(sizes) / (np.mean(sizes) + 1e-12)) if n_clusters > 0 else 0.0
    return dict(n_clusters=n_clusters, sizes=sizes, singleton_frac=singleton_frac, tiny_frac=tiny_frac, size_cv=size_cv)

def intra_inter_separation(W: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    n = len(labels)
    total, wsum = 0.0, 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        m = idx.size
        if m <= 1:
            continue
        comp = np.setdiff1d(np.arange(n), idx, assume_unique=True)
        sub = W[np.ix_(idx, idx)]
        intra = sub.sum() / (m * (m - 1))
        inter = W[np.ix_(idx, comp)].mean() if comp.size > 0 else 0.0
        total += (intra - inter) * m
        wsum += m
    return float(total / (wsum + 1e-12)) if wsum > 0 else 0.0

def cluster_density_std(W: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    densities = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        m = idx.size
        if m <= 1:
            continue
        sub = W[np.ix_(idx, idx)]
        dens = sub.sum() / (m * (m - 1))
        densities.append(dens)
    return float(np.std(densities)) if densities else float("inf")

def coassociation_matrix(partitions: List[np.ndarray]) -> np.ndarray:
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

def leiden_on_consensus(C: np.ndarray, tau: float = 0.6, gamma: float = 0.1, seed: int = 0) -> np.ndarray:
    Wc = C.copy()
    Wc[Wc < tau] = 0.0
    np.fill_diagonal(Wc, 0.0)
    labels, _ = run_leiden_on_W(Wc, gamma=gamma, seed=seed, objective="CPM")
    return labels

def pairwise_mean_ari(partitions: List[np.ndarray]) -> float:
    if len(partitions) <= 1:
        return 1.0
    ARIs = []
    for i in range(len(partitions)):
        for j in range(i + 1, len(partitions)):
            ARIs.append(adjusted_rand_score(partitions[i], partitions[j]))
    return float(np.mean(ARIs)) if ARIs else 1.0

def safe_silhouette_from_similarity(W: np.ndarray, labels: np.ndarray, use_silhouette: bool = False, sample_size: Optional[int] = None, rng: int = 42) -> Optional[float]:
    if not use_silhouette:
        return None
    dist = 1.0 - np.clip(W, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    if uniq.size < 2 or np.any(counts < 2):
        return None
    if sample_size is not None and sample_size < len(labels):
        rnd = np.random.default_rng(rng)
        idx = rnd.choice(len(labels), size=sample_size, replace=False)
        dist = dist[np.ix_(idx, idx)]
        labels = labels[idx]
    try:
        return float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:
        return None

def _minmax(xs: List[float]) -> Tuple[np.ndarray, float, float]:
    arr = np.array(xs, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr), 0.0, 1.0
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        return np.zeros_like(arr), vmin, vmax
    return (arr - vmin) / (vmax - vmin), vmin, vmax

@dataclass
class ModuleBestResult:
    module_cluster_id: Any
    m_funcs: int
    best_gamma: float
    best_labels: np.ndarray
    combined_score: float
    mean_ARI: float
    mean_sep: float
    mean_sil: Optional[float]
    mean_n_clusters: float
    mean_singleton_frac: float
    mean_tiny_frac: float
    mean_size_cv: float
    notes: str

def evaluate_module_and_pick_best(
    method_cluster: Any,
    weight_parameter: float = 0.25,
    gamma_min: float = 1e-3,
    gamma_max: float = 1.0,
    n_points: int = 24,
    seeds_per_gamma: int = 8,
    use_knn: bool = True,
    knn_k: int = 20,
    use_threshold: bool = False,
    threshold_tau: float = 0.0,
    min_clusters: int = 2,
    max_clusters_ratio: float = 0.2,
    use_silhouette: bool = False,
    silhouette_sample_size: Optional[int] = None,
    objective: str = "CPM",
    consensus_tau: float = 0.6,
    consensus_gamma: float = 0.1,
    rng_seed: int = 2025
) -> ModuleBestResult:
    funcs = getattr(method_cluster, "cluster_func_list", None)
    module_cluster_id = getattr(method_cluster, "cluster_id", None)
    if funcs is None:
        raise ValueError("method_cluster must have attribute 'cluster_func_list'")

    m = len(funcs)
    if m == 0:
        return ModuleBestResult(module_cluster_id, 0, 0.0, np.array([], dtype=int), 0.0, 1.0, 0.0, None, 0.0, 0.0, 0.0, 0.0, "empty module")
    if m == 1:
        labels = np.zeros(1, dtype=int)
        return ModuleBestResult(module_cluster_id, 1, 0.0, labels, 1.0, 1.0, 1.0, None, 1.0, 0.0, 0.0, 0.0, "singleton")

    # 1) 权重矩阵（语义 + 结构）
    S = compute_function_similarity_matrix(method_cluster)
    L = compute_link(method_cluster)
    W = fuse_weights_func(S, L, a=weight_parameter)

    # 2) 稀疏化
    if use_knn:
        W = sparsify_knn(W, k=knn_k)
    if use_threshold and threshold_tau > 0:
        W = sparsify_threshold(W, tau=threshold_tau)
    np.fill_diagonal(W, 0.0)

    # 3) γ 网格
    gamma_values = np.logspace(np.log10(gamma_min), np.log10(gamma_max), num=n_points)
    rng = np.random.default_rng(rng_seed)

    all_results: List[Dict[str, Any]] = []
    n = W.shape[0]
    max_clusters = max(2, int(max_clusters_ratio * n))

    # 4) 扫描 γ
    for gamma in gamma_values:
        partitions, qualities = [], []
        sep_list, dens_std_list, sil_list = [], [], []
        size_stats_list = []

        for _ in range(seeds_per_gamma):
            seed = int(rng.integers(0, 1_000_000))
            labels, q = run_leiden_on_W(W, gamma=float(gamma), seed=seed, objective=objective)
            partitions.append(labels)
            qualities.append(q)
            size_stats_list.append(cluster_size_stats(labels))
            sep_list.append(intra_inter_separation(W, labels))
            dens_std_list.append(cluster_density_std(W, labels))
            sil_val = safe_silhouette_from_similarity(
                W, labels, use_silhouette=use_silhouette,
                sample_size=silhouette_sample_size, rng=seed
            )
            if sil_val is not None:
                sil_list.append(sil_val)

        mean_ARI = pairwise_mean_ari(partitions)
        mean_sep = float(np.mean(sep_list))
        mean_density_std = float(np.mean([d for d in dens_std_list if np.isfinite(d)]) if any(np.isfinite(dens_std_list)) else np.inf)
        mean_sil = float(np.mean(sil_list)) if sil_list else None

        n_clusters_list = [ss["n_clusters"] for ss in size_stats_list]
        singleton_fracs = [ss["singleton_frac"] for ss in size_stats_list]
        tiny_fracs = [ss["tiny_frac"] for ss in size_stats_list]
        size_cvs = [ss["size_cv"] for ss in size_stats_list]

        agg = dict(
            gamma=float(gamma),
            partitions=partitions,
            mean_ARI=mean_ARI,
            mean_sep=mean_sep,
            mean_density_std=mean_density_std,
            mean_sil=mean_sil,
            mean_n_clusters=float(np.mean(n_clusters_list)),
            mean_singleton_frac=float(np.mean(singleton_fracs)),
            mean_tiny_frac=float(np.mean(tiny_fracs)),
            mean_size_cv=float(np.mean(size_cvs)),
        )
        all_results.append(agg)

    # 5) 组合评分
    ARIs = [r["mean_ARI"] for r in all_results]
    SEPs = [r["mean_sep"] for r in all_results]
    SILs = [(-1 if r["mean_sil"] is None else r["mean_sil"]) for r in all_results]
    NCLS = [r["mean_n_clusters"] for r in all_results]
    SINGLE = [r["mean_singleton_frac"] for r in all_results]
    TINY = [r["mean_tiny_frac"] for r in all_results]
    SIZECV = [r["mean_size_cv"] for r in all_results]
    DSTD = [r["mean_density_std"] for r in all_results]

    n_ARIs, _, _ = _minmax(ARIs)
    n_SEPs, _, _ = _minmax(SEPs)
    n_SILs, _, _ = _minmax(SILs) if use_silhouette and any(s >= -1 for s in SILs) else (np.zeros_like(n_ARIs), 0.0, 1.0)

    over_penalty = np.clip((np.array(NCLS) - max_clusters) / max(1, max_clusters), 0, 1)
    under_penalty = np.clip((min_clusters - np.array(NCLS)) / max(1, min_clusters), 0, 1)
    small_penalty = 0.5 * np.array(SINGLE) + 0.5 * np.array(TINY)

    dstd_arr = np.array(DSTD, dtype=float)
    dstd_arr[~np.isfinite(dstd_arr)] = np.nanmax(dstd_arr[np.isfinite(dstd_arr)]) if np.any(np.isfinite(dstd_arr)) else 1.0
    n_DSTD, _, _ = _minmax(dstd_arr)
    sizecv_penalty = np.clip((np.array(SIZECV) - 1.0) / 1.0, 0, 1)

    w_stab = 0.45
    w_sep  = 0.45
    w_sil  = 0
    w_pen_small = 0.40
    w_pen_over  = 0.20
    w_pen_under = 0.10
    w_pen_sizecv= 0.20
    w_pen_dstd  = 0.15

    combined_scores = (
        w_stab * n_ARIs
        + w_sep * n_SEPs
        + (w_sil * n_SILs if use_silhouette else 0)
        - (w_pen_small * small_penalty
           + w_pen_over * over_penalty
           + w_pen_under * under_penalty
           + w_pen_sizecv * sizecv_penalty
           + w_pen_dstd * n_DSTD)
    )

    for i, r in enumerate(all_results):
        r["combined_score"] = float(combined_scores[i])

    # 6) 选出最佳 γ
    best_idx = int(np.argmax(combined_scores))
    best_gamma = float(all_results[best_idx]["gamma"])
    best_partitions = all_results[best_idx]["partitions"]
    C = coassociation_matrix(best_partitions)
    best_labels = leiden_on_consensus(
        C, tau=consensus_tau, gamma=consensus_gamma,
        seed=int(np.random.default_rng(rng_seed).integers(0, 1_000_000))
    )

    best_stats = all_results[best_idx]
    return ModuleBestResult(
        module_cluster_id=module_cluster_id,
        m_funcs=m,
        best_gamma=best_gamma,
        best_labels=best_labels,
        combined_score=float(best_stats["combined_score"]),
        mean_ARI=float(best_stats["mean_ARI"]),
        mean_sep=float(best_stats["mean_sep"]),
        mean_sil=(None if best_stats["mean_sil"] == -1 else best_stats["mean_sil"]),
        mean_n_clusters=float(best_stats["mean_n_clusters"]),
        mean_singleton_frac=float(best_stats["mean_singleton_frac"]),
        mean_tiny_frac=float(best_stats["mean_tiny_frac"]),
        mean_size_cv=float(best_stats["mean_size_cv"]),
        notes="no global cluster-count constraint; per-module resolution selected by combined score"
    )

def cluster_all_functions_to_features(
    method_clusters: List[Any],
    weight_parameter: float = 0.25,
    gamma_min: float = 0.01,
    gamma_max: float = 0.5,
    n_points: int = 24,
    seeds_per_gamma: int = 8,
    use_knn: bool = True,
    knn_k: int = 20,
    use_threshold: bool = False,
    threshold_tau: float = 0.0,
    min_clusters: int = 2,
    max_clusters_ratio: float = 0.2,
    use_silhouette: bool = False,
    silhouette_sample_size: Optional[int] = None,
    objective: str = "CPM",
    consensus_tau: float = 0.6,
    consensus_gamma: float = 0.1,
    rng_seed: int = 2025,
    target_total_features: Optional[int] = None
) -> Tuple[List[Feature], Dict[str, Any]]:
    feature_list: List[Feature] = []
    per_module_summaries: List[Dict[str, Any]] = []

    for mc in method_clusters:
        res = evaluate_module_and_pick_best(
            mc,
            weight_parameter=weight_parameter,
            gamma_min=gamma_min, gamma_max=gamma_max, n_points=n_points,
            seeds_per_gamma=seeds_per_gamma,
            use_knn=use_knn, knn_k=knn_k,
            use_threshold=use_threshold, threshold_tau=threshold_tau,
            min_clusters=min_clusters, max_clusters_ratio=max_clusters_ratio,
            use_silhouette=use_silhouette, silhouette_sample_size=silhouette_sample_size,
            objective=objective,
            consensus_tau=consensus_tau, consensus_gamma=consensus_gamma,
            rng_seed=rng_seed
        )
        
        W = fuse_weights_func(
            compute_function_similarity_matrix(mc),
            compute_link(mc),
            a=weight_parameter
        )
        funcs = getattr(mc, "cluster_func_list", [])
        labels = res.best_labels

        # 组装 Feature
        clusters: Dict[int, Feature] = {}
        for idx_func, cid in enumerate(labels):
            if cid not in clusters:
                clusters[cid] = Feature(
                    cluster_id=getattr(mc, "cluster_id", None),
                    feature_id=0,  # 稍后会重新分配
                    feature_desc="",  # 稍后会填充
                    feature_func_list=[]
                )
            clusters[cid].feature_func_list.append(funcs[idx_func])

        # 填充描述
        for f in clusters.values():
            f.feature_desc = (
                f"gamma={res.best_gamma:.4f}; k={len(clusters)}; "
                f"a={weight_parameter:.2f}; combined={res.combined_score:.3f}; "
                f"stability(ARI)={res.mean_ARI:.3f}; sep={res.mean_sep:.3f}"
            )
            feature_list.append(f)

        per_module_summaries.append({
            "module_cluster_id": res.module_cluster_id,
            "m_funcs": res.m_funcs,
            "chosen_gamma": res.best_gamma,
            "chosen_k": int(len(np.unique(labels))),
            "combined_score": res.combined_score,
            "mean_ARI": res.mean_ARI,
            "mean_sep": res.mean_sep,
            "mean_sil": res.mean_sil,
            "mean_n_clusters": res.mean_n_clusters,
            "mean_singleton_frac": res.mean_singleton_frac,
            "mean_tiny_frac": res.mean_tiny_frac,
            "mean_size_cv": res.mean_size_cv,
            "notes": res.notes
        })

    # 分配全局 feature_id
    for i, feature in enumerate(feature_list, start=1):
        feature.feature_id = i

    summary = {
        "target_total": target_total_features,
        "achieved_total": len(feature_list),
        "per_module": per_module_summaries,
        "notes": "Function-level clustering uses per-module γ selection by combined score; no global count constraint."
    }
    return feature_list, summary

