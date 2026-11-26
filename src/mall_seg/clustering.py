"""Clustering suite: K-Means, Agglomerative Ward, DBSCAN, GMM, and k-dist knee tuner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from mall_seg.algorithms import (
    AlgorithmMetrics,
    ClusteringSuiteResult,
    GMMSweepPoint,
    KMeansSweepPoint,
    compute_normalized_entropy,
    estimate_dbscan_eps,
    run_clustering_suite,
)


@dataclass(frozen=True)
class KDistanceCurve:
    """Sorted k-NN distances and estimated knee epsilon for DBSCAN density tuning."""

    sorted_indices: list[int]
    k_distances: list[float]
    knee_index: int
    recommended_eps: float
    min_samples: int


def compute_k_distance_curve(
    X_scaled: np.ndarray,
    min_samples: int = 5,
) -> KDistanceCurve:
    """Compute sorted k-NN distance curve and geometric maximum-curvature knee for DBSCAN eps."""
    n_neighbors = min(max(min_samples, 2), len(X_scaled))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])

    recommended_eps = estimate_dbscan_eps(X_scaled, min_samples=min_samples)
    if len(k_distances) < 3 or float(k_distances[-1] - k_distances[0]) <= 1e-9:
        knee_idx = max(0, len(k_distances) // 2)
    else:
        x_coords = np.linspace(0.0, 1.0, len(k_distances))
        y_coords = (k_distances - float(k_distances[0])) / float(k_distances[-1] - k_distances[0])
        line_vec = np.array([1.0, 1.0])
        points = np.column_stack([x_coords, y_coords])
        cross = np.abs(points[:, 0] * line_vec[1] - points[:, 1] * line_vec[0])
        knee_idx = int(np.argmax(cross / float(np.linalg.norm(line_vec))))

    return KDistanceCurve(
        sorted_indices=list(range(len(k_distances))),
        k_distances=[round(float(v), 4) for v in k_distances],
        knee_index=knee_idx,
        recommended_eps=recommended_eps,
        min_samples=min_samples,
    )


def fit_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
    n_init: int = 50,
    seed: int = 42,
) -> tuple[KMeans, np.ndarray]:
    """Fit deterministic K-Means++ model and return model + labels."""
    model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=n_init, random_state=seed)
    labels = model.fit_predict(X_scaled)
    return model, labels


def fit_agglomerative_ward(
    X_scaled: np.ndarray,
    n_clusters: int = 5,
) -> tuple[AgglomerativeClustering, np.ndarray]:
    """Fit Ward-linkage Agglomerative clustering model and return model + labels."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X_scaled)
    return model, labels


def fit_dbscan_tuned(
    X_scaled: np.ndarray,
    min_samples: int = 5,
    eps: float | None = None,
) -> tuple[DBSCAN, np.ndarray, float]:
    """Fit DBSCAN using provided epsilon or k-dist geometric knee tuner."""
    resolved_eps = (
        eps if eps is not None else estimate_dbscan_eps(X_scaled, min_samples=min_samples)
    )
    model = DBSCAN(eps=resolved_eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    return model, labels, resolved_eps


def fit_gmm(
    X_scaled: np.ndarray,
    n_components: int = 5,
    covariance_type: str = "full",
    seed: int = 42,
) -> tuple[GaussianMixture, np.ndarray, np.ndarray, np.ndarray]:
    """Fit Gaussian Mixture Model and return hard labels, posteriors, and normalized entropy."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,  # type: ignore[arg-type]
        n_init=10,
        random_state=seed,
    )
    model.fit(X_scaled)
    labels = model.predict(X_scaled)
    posteriors = np.asarray(model.predict_proba(X_scaled), dtype=np.float64)
    entropy = compute_normalized_entropy(posteriors)
    return model, labels, posteriors, entropy


__all__ = [
    "AlgorithmMetrics",
    "ClusteringSuiteResult",
    "GMMSweepPoint",
    "KDistanceCurve",
    "KMeansSweepPoint",
    "compute_k_distance_curve",
    "compute_normalized_entropy",
    "estimate_dbscan_eps",
    "fit_agglomerative_ward",
    "fit_dbscan_tuned",
    "fit_gmm",
    "fit_kmeans",
    "run_clustering_suite",
]
