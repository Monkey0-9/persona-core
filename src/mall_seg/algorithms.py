"""Multi-algorithm clustering suite: K-Means, Ward Hierarchical, DBSCAN, and GMM."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class KMeansSweepPoint:
    k: int
    inertia: float
    silhouette: float
    calinski_harabasz: float = 0.0
    davies_bouldin: float = 0.0


@dataclass(frozen=True)
class GMMSweepPoint:
    k: int
    bic: float
    aic: float


@dataclass(frozen=True)
class AlgorithmMetrics:
    algorithm: str
    n_clusters: int
    silhouette: float | None
    calinski_harabasz: float | None
    davies_bouldin: float | None
    noise_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusteringSuiteResult:
    kmeans_model: KMeans
    kmeans_labels: np.ndarray
    kmeans_sweep: list[KMeansSweepPoint]
    agglomerative_model: AgglomerativeClustering
    agglomerative_labels: np.ndarray
    ward_linkage_matrix: np.ndarray
    dbscan_model: DBSCAN
    dbscan_labels: np.ndarray
    dbscan_eps: float
    dbscan_min_samples: int
    gmm_model: GaussianMixture
    gmm_labels: np.ndarray
    gmm_posteriors: np.ndarray
    gmm_entropy: np.ndarray
    gmm_sweep: list[GMMSweepPoint]
    benchmarks: list[AlgorithmMetrics]
    hdbscan_labels: np.ndarray | None = None
    hdbscan_min_cluster_size: int = 5
    consensus: dict[str, Any] = field(default_factory=dict)


def compute_algorithm_consensus(partitions: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute pairwise Adjusted Rand Index (ARI) and NMI across clustering algorithms."""
    names = list(partitions.keys())
    ari_matrix: dict[str, dict[str, float]] = {}
    nmi_matrix: dict[str, dict[str, float]] = {}
    for name_a in names:
        ari_matrix[name_a] = {}
        nmi_matrix[name_a] = {}
        lbl_a = partitions[name_a]
        for name_b in names:
            lbl_b = partitions[name_b]
            ari_matrix[name_a][name_b] = round(float(adjusted_rand_score(lbl_a, lbl_b)), 4)
            nmi_matrix[name_a][name_b] = round(float(normalized_mutual_info_score(lbl_a, lbl_b)), 4)
    return {"algorithms": names, "ari_matrix": ari_matrix, "nmi_matrix": nmi_matrix}


def compute_normalized_entropy(posteriors: np.ndarray) -> np.ndarray:
    """Compute normalized Shannon entropy in [0.0, 1.0] for GMM posterior probabilities."""
    n_clusters = posteriors.shape[1]
    if n_clusters <= 1:
        return np.zeros(posteriors.shape[0], dtype=np.float64)
    clipped = np.clip(posteriors, 1e-12, 1.0)
    raw_entropy = -np.sum(clipped * np.log(clipped), axis=1)
    normalized = raw_entropy / np.log(float(n_clusters))
    return np.asarray(np.clip(normalized, 0.0, 1.0), dtype=np.float64)


def estimate_dbscan_eps(X_scaled: np.ndarray, min_samples: int = 5) -> float:
    """Estimate DBSCAN epsilon via geometric knee on sorted k-NN distances."""
    n_neighbors = min(max(min_samples, 2), len(X_scaled))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    if len(k_distances) < 3 or float(k_distances[-1] - k_distances[0]) <= 1e-9:
        return max(float(np.median(k_distances)), 0.5)

    x_coords = np.linspace(0.0, 1.0, len(k_distances))
    y_min = float(k_distances[0])
    y_max = float(k_distances[-1])
    y_coords = (k_distances - y_min) / (y_max - y_min)

    # Distance from each point (x, y) to line connecting (0, 0) and (1, 1)
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 1.0])
    line_vec = p2 - p1
    line_norm = float(np.linalg.norm(line_vec))
    points = np.column_stack([x_coords, y_coords])
    diffs = points - p1
    cross = np.abs(diffs[:, 0] * line_vec[1] - diffs[:, 1] * line_vec[0])
    perp_dist = cross / line_norm

    knee_idx = int(np.argmax(perp_dist))
    eps = float(k_distances[knee_idx])
    return round(max(eps, 0.25), 4)


def safe_cluster_metrics(
    X: np.ndarray, labels: np.ndarray
) -> tuple[float | None, float | None, float | None, int]:
    """Compute Silhouette, Calinski-Harabasz, and Davies-Bouldin safely with noise handling."""
    noise_mask = labels == -1
    noise_count = int(np.sum(noise_mask))
    valid_mask = ~noise_mask
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    unique_labels = np.unique(labels_valid)
    if len(unique_labels) < 2 or len(labels_valid) <= len(unique_labels):
        return None, None, None, noise_count

    sil = round(float(silhouette_score(X_valid, labels_valid)), 4)
    ch = round(float(calinski_harabasz_score(X_valid, labels_valid)), 4)
    db = round(float(davies_bouldin_score(X_valid, labels_valid)), 4)
    return sil, ch, db, noise_count


def run_clustering_suite(
    X_scaled: np.ndarray,
    target_k: int = 5,
    k_min: int = 2,
    k_max: int = 10,
    kmeans_n_init: int = 50,
    dbscan_min_samples: int = 5,
    gmm_covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
    seed: int = 42,
) -> ClusteringSuiteResult:
    """Fit and benchmark K-Means, Ward Hierarchical, DBSCAN, and Gaussian Mixture Model."""
    max_feasible_k = min(k_max, max(2, len(X_scaled) - 1))

    # 1. K-Means sweep + final fit
    kmeans_sweep: list[KMeansSweepPoint] = []
    for k in range(k_min, max_feasible_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=kmeans_n_init, random_state=seed)
        lbls = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, lbls))
        ch = float(calinski_harabasz_score(X_scaled, lbls))
        db = float(davies_bouldin_score(X_scaled, lbls))
        kmeans_sweep.append(
            KMeansSweepPoint(
                k=k,
                inertia=round(float(km.inertia_), 4),
                silhouette=round(sil, 4),
                calinski_harabasz=round(ch, 4),
                davies_bouldin=round(db, 4),
            )
        )

    kmeans_model = KMeans(
        n_clusters=target_k, init="k-means++", n_init=kmeans_n_init, random_state=seed
    )
    kmeans_labels = kmeans_model.fit_predict(X_scaled)
    km_sil, km_ch, km_db, km_noise = safe_cluster_metrics(X_scaled, kmeans_labels)

    # 2. Agglomerative Ward Hierarchical Clustering + scipy linkage
    agglomerative_model = AgglomerativeClustering(n_clusters=target_k, linkage="ward")
    agglomerative_labels = agglomerative_model.fit_predict(X_scaled)
    ward_linkage_matrix = np.asarray(linkage(X_scaled, method="ward"), dtype=np.float64)
    agg_sil, agg_ch, agg_db, agg_noise = safe_cluster_metrics(X_scaled, agglomerative_labels)

    # 3. DBSCAN with sorted k-NN distance knee eps estimation
    dbscan_eps = estimate_dbscan_eps(X_scaled, min_samples=dbscan_min_samples)
    dbscan_model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    dbscan_labels = dbscan_model.fit_predict(X_scaled)
    db_sil, db_ch, db_db, db_noise = safe_cluster_metrics(X_scaled, dbscan_labels)
    db_n_clusters = len([lbl for lbl in np.unique(dbscan_labels) if lbl != -1])

    # 3b. HDBSCAN hierarchical density clustering (2026 density baseline)
    hdb_min_cluster_size = max(3, min(10, len(X_scaled) // 16))
    hdb_min_samples = min(dbscan_min_samples, hdb_min_cluster_size)
    if len(X_scaled) >= hdb_min_cluster_size + 2:
        hdb_model = HDBSCAN(
            min_cluster_size=hdb_min_cluster_size,
            min_samples=hdb_min_samples,
            copy=True,
        )
        hdbscan_labels = np.asarray(hdb_model.fit_predict(X_scaled), dtype=int)
    else:
        hdbscan_labels = np.full(len(X_scaled), -1, dtype=int)
    hdb_sil, hdb_ch, hdb_db, hdb_noise = safe_cluster_metrics(X_scaled, hdbscan_labels)
    hdb_n_clusters = len([lbl for lbl in np.unique(hdbscan_labels) if lbl != -1])

    # 4. Gaussian Mixture Model BIC/AIC sweep + final posterior & entropy
    gmm_sweep: list[GMMSweepPoint] = []
    for k in range(k_min, max_feasible_k + 1):
        gmm_k = GaussianMixture(
            n_components=k,
            covariance_type=gmm_covariance_type,
            n_init=10,
            random_state=seed,
        )
        gmm_k.fit(X_scaled)
        gmm_sweep.append(
            GMMSweepPoint(
                k=k,
                bic=round(float(gmm_k.bic(X_scaled)), 4),
                aic=round(float(gmm_k.aic(X_scaled)), 4),
            )
        )

    gmm_model = GaussianMixture(
        n_components=target_k,
        covariance_type=gmm_covariance_type,
        n_init=10,
        random_state=seed,
    )
    gmm_model.fit(X_scaled)
    gmm_labels = gmm_model.predict(X_scaled)
    gmm_posteriors = np.asarray(gmm_model.predict_proba(X_scaled), dtype=np.float64)
    gmm_entropy = compute_normalized_entropy(gmm_posteriors)
    gmm_sil, gmm_ch, gmm_db, gmm_noise = safe_cluster_metrics(X_scaled, gmm_labels)

    benchmarks = [
        AlgorithmMetrics(
            algorithm="K-Means (k-means++)",
            n_clusters=target_k,
            silhouette=km_sil,
            calinski_harabasz=km_ch,
            davies_bouldin=km_db,
            noise_count=km_noise,
            extra={"inertia": round(float(kmeans_model.inertia_), 4), "n_init": kmeans_n_init},
        ),
        AlgorithmMetrics(
            algorithm="Agglomerative (Ward)",
            n_clusters=target_k,
            silhouette=agg_sil,
            calinski_harabasz=agg_ch,
            davies_bouldin=agg_db,
            noise_count=agg_noise,
            extra={"linkage": "ward"},
        ),
        AlgorithmMetrics(
            algorithm="DBSCAN (k-NN knee)",
            n_clusters=db_n_clusters,
            silhouette=db_sil,
            calinski_harabasz=db_ch,
            davies_bouldin=db_db,
            noise_count=db_noise,
            extra={"eps": dbscan_eps, "min_samples": dbscan_min_samples},
        ),
        AlgorithmMetrics(
            algorithm=f"Gaussian Mixture ({gmm_covariance_type})",
            n_clusters=target_k,
            silhouette=gmm_sil,
            calinski_harabasz=gmm_ch,
            davies_bouldin=gmm_db,
            noise_count=gmm_noise,
            extra={
                "bic": round(float(gmm_model.bic(X_scaled)), 4),
                "mean_entropy": round(float(np.mean(gmm_entropy)), 4),
            },
        ),
        AlgorithmMetrics(
            algorithm="HDBSCAN (density-hierarchy)",
            n_clusters=hdb_n_clusters,
            silhouette=hdb_sil,
            calinski_harabasz=hdb_ch,
            davies_bouldin=hdb_db,
            noise_count=hdb_noise,
            extra={"min_cluster_size": hdb_min_cluster_size, "min_samples": hdb_min_samples},
        ),
    ]

    consensus = compute_algorithm_consensus(
        {
            "K-Means++": kmeans_labels,
            "Ward": agglomerative_labels,
            "GMM": gmm_labels,
            "DBSCAN": dbscan_labels,
            "HDBSCAN": hdbscan_labels,
        }
    )

    return ClusteringSuiteResult(
        kmeans_model=kmeans_model,
        kmeans_labels=kmeans_labels,
        kmeans_sweep=kmeans_sweep,
        agglomerative_model=agglomerative_model,
        agglomerative_labels=agglomerative_labels,
        ward_linkage_matrix=ward_linkage_matrix,
        dbscan_model=dbscan_model,
        dbscan_labels=dbscan_labels,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        gmm_model=gmm_model,
        gmm_labels=gmm_labels,
        gmm_posteriors=gmm_posteriors,
        gmm_entropy=gmm_entropy,
        gmm_sweep=gmm_sweep,
        benchmarks=benchmarks,
        hdbscan_labels=hdbscan_labels,
        hdbscan_min_cluster_size=hdb_min_cluster_size,
        consensus=consensus,
    )
