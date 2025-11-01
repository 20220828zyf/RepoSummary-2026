import numpy as np
from typing import List
from model.models import File, file_Cluster
from utils.matrix_computation import compute_similarity_matrix, compute_link_matrix
from utils.clustering_utils import (
    fuse_weights, sparsify_knn, sparsify_threshold, run_leiden,
    cluster_size_stats, cluster_intra_inter_separation, cluster_density_std,
    coassociation_matrix, consensus_labels_from_coassoc, pairwise_mean_ari,
    safe_silhouette_from_similarity, merge_tiny_clusters
)

def find_best_resolution(
    files,
    a=0.5,
    n_points=30,
    gamma_min=1e-1,
    gamma_max=1.0,
    seeds_per_gamma=8,
    min_clusters=2,
    max_clusters_ratio=0.2,
    min_cluster_size=2,
    use_knn=True,
    knn_k=20,
    use_threshold=False,
    threshold_tau=0.0,
    use_silhouette=False,
    silhouette_sample_size=None,
    objective="CPM",
    consensus_tau=0.6,
    consensus_gamma=0.1,
    random_state=2025
):
    """
    返回:
    - best_gamma: 最优分辨率
    - best_labels: 最优 γ 的"共识"聚类标签（多次运行整合）
    - results: 每个 γ 的统计与指标
    """
    rng = np.random.default_rng(random_state)

    # 1) 计算并融合权重矩阵
    S = compute_similarity_matrix(files)
    A = compute_link_matrix(files)
    W = fuse_weights(A, S, a=a)

    # 2) 稀疏化
    if use_knn:
        W = sparsify_knn(W, k=knn_k)
    if use_threshold and threshold_tau > 0:
        W = sparsify_threshold(W, tau=threshold_tau)
    np.fill_diagonal(W, 0.0)

    n = W.shape[0]
    max_clusters = max(2, int(max_clusters_ratio * n))

    # 3) 准备 γ 网格
    gamma_values = np.logspace(np.log10(gamma_min), np.log10(gamma_max), num=n_points)

    # 4) 扫描 γ：多次运行 + 指标统计
    all_results = []
    for gamma in gamma_values:
        partitions = []
        qualities = []
        size_stats_list = []
        separations = []
        density_stds = []
        sils = []

        for s in range(seeds_per_gamma):
            seed = int(rng.integers(0, 1_000_000))
            labels, quality = run_leiden(W, gamma=gamma, seed=seed, objective=objective)
            partitions.append(labels)
            qualities.append(quality)
            ss = cluster_size_stats(labels)
            size_stats_list.append(ss)
            separations.append(cluster_intra_inter_separation(W, labels))
            density_stds.append(cluster_density_std(W, labels))
            sil_val = safe_silhouette_from_similarity(
                W, labels, use_silhouette=use_silhouette,
                sample_size=silhouette_sample_size, rng=seed
            )
            if sil_val is not None:
                sils.append(sil_val)

        mean_ARI = pairwise_mean_ari(partitions)
        mean_quality = float(np.mean(qualities))
        mean_sep = float(np.mean(separations))
        mean_density_std = float(np.mean([d for d in density_stds if np.isfinite(d)]) if any(np.isfinite(density_stds)) else np.inf)
        mean_sil = float(np.mean(sils)) if sils else None

        n_clusters_list = [ss["n_clusters"] for ss in size_stats_list]
        singleton_fracs = [ss["singleton_frac"] for ss in size_stats_list]
        tiny_fracs = [ss["tiny_frac"] for ss in size_stats_list]
        size_cvs = [ss["size_cv"] for ss in size_stats_list]

        agg = dict(
            gamma=float(gamma),
            mean_ARI=mean_ARI,
            mean_quality=mean_quality,
            mean_sep=mean_sep,
            mean_density_std=mean_density_std,
            mean_sil=mean_sil,
            mean_n_clusters=float(np.mean(n_clusters_list)),
            p50_n_clusters=float(np.median(n_clusters_list)),
            mean_singleton_frac=float(np.mean(singleton_fracs)),
            mean_tiny_frac=float(np.mean(tiny_fracs)),
            mean_size_cv=float(np.mean(size_cvs)),
            partitions=partitions
        )
        all_results.append(agg)

    # 5) 组合评分
    def minmax(xs):
        xs = np.array(xs, dtype=float)
        if np.all(~np.isfinite(xs)):
            return np.zeros_like(xs), 0.0, 1.0
        xs[~np.isfinite(xs)] = np.nan
        vmin = np.nanmin(xs)
        vmax = np.nanmax(xs)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            return np.zeros_like(xs), vmin, vmax
        return (xs - vmin) / (vmax - vmin), vmin, vmax

    ARIs = [r["mean_ARI"] for r in all_results]
    SEPs = [r["mean_sep"] for r in all_results]
    SILs = [(-1 if r["mean_sil"] is None else r["mean_sil"]) for r in all_results]
    NCLS = [r["mean_n_clusters"] for r in all_results]
    SINGLE = [r["mean_singleton_frac"] for r in all_results]
    TINY = [r["mean_tiny_frac"] for r in all_results]
    SIZECV = [r["mean_size_cv"] for r in all_results]
    DSTD = [r["mean_density_std"] for r in all_results]

    n_ARIs, _, _ = minmax(ARIs)
    n_SEPs, _, _ = minmax(SEPs)
    n_SILs, _, _ = minmax(SILs) if any(s >= -1 for s in SILs) and use_silhouette else (np.zeros_like(ARIs), 0, 1)

    over_penalty = np.clip((np.array(NCLS) - max_clusters) / max(1, max_clusters), 0, 1)
    under_penalty = np.clip((min_clusters - np.array(NCLS)) / max(1, min_clusters), 0, 1)
    small_penalty = 0.5 * np.array(SINGLE) + 0.5 * np.array(TINY)
    sizecv_penalty = np.clip((np.array(SIZECV) - 1.0) / 1.0, 0, 1)

    dstd_arr = np.array(DSTD, dtype=float)
    dstd_arr[~np.isfinite(dstd_arr)] = np.nanmax(dstd_arr[np.isfinite(dstd_arr)]) if np.any(np.isfinite(dstd_arr)) else 1.0
    n_DSTD, _, _ = minmax(dstd_arr)
    dstd_penalty = n_DSTD

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
           + w_pen_dstd * dstd_penalty)
    )

    for i, r in enumerate(all_results):
        r["combined_score"] = float(combined_scores[i])

    # 6) 选出 best γ
    best_idx = int(np.argmax(combined_scores))
    best_gamma = float(all_results[best_idx]["gamma"])
    best_partitions = all_results[best_idx]["partitions"]

    C = coassociation_matrix(best_partitions)
    best_labels = consensus_labels_from_coassoc(
        C, tau=consensus_tau, gamma=consensus_gamma,
        seed=int(rng.integers(0, 1_000_000))
    )
    print(f"Best gamma: {best_gamma}, #Clusters: {len(np.unique(best_labels))}")
    best_labels = merge_tiny_clusters(best_labels, W, min_size=2, min_accept_sim=0.0, penalize_large=True, max_passes=5)
    print(f"After merging tiny clusters, #Clusters: {len(np.unique(best_labels))}")
    return best_gamma, best_labels, all_results

def save_to_file_cluster(files, best_labels):
    file_clusters = {}
    for i, file in enumerate(files):
        if best_labels[i] not in file_clusters:
            file_clusters[best_labels[i]] = file_Cluster(best_labels[i], "", [])
        file_clusters[best_labels[i]].cluster_file_list.append(file)
    return file_clusters.values()

