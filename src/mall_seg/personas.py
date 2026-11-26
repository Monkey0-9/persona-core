"""Deterministic persona mapping and live/batch customer decisioning."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import linear_sum_assignment
from sqlalchemy.engine import Engine

from mall_seg.algorithms import compute_normalized_entropy, run_clustering_suite
from mall_seg.contracts import NUMERIC_FEATURE_COLUMNS, AppConfig, strip_customer_id
from mall_seg.db import (
    create_db_engine,
    get_session_factory,
    init_db,
    session_scope,
    upsert_model_version,
    upsert_pipeline_run,
)
from mall_seg.drift import compute_reference_stats, evaluate_single_record_drift
from mall_seg.eda import compute_eda_report
from mall_seg.explainability import fit_surrogate_explainer
from mall_seg.features import fit_transform_features, transform_features
from mall_seg.stability import evaluate_kmeans_stability

logger = logging.getLogger("mall_seg.personas")


@dataclass(frozen=True)
class PersonaArchetype:
    key: str
    name: str
    tagline: str
    recommended_action: str
    target_channel: str
    # Canonical profile in standardized (Age_z, Income_z, Spending_z) coordinates
    target_vector: tuple[float, float, float]


CANONICAL_ARCHETYPES: list[PersonaArchetype] = [
    PersonaArchetype(
        key="affluent_vip",
        name="Affluent VIPs",
        tagline="High-income, high-spending prime luxury shoppers",
        recommended_action=(
            "Enroll in VIP concierge tier with early collection previews, "
            "private styling appointments, and tiered loyalty multipliers."
        ),
        target_channel="Concierge SMS + Private Invite",
        target_vector=(-0.45, 1.10, 1.25),
    ),
    PersonaArchetype(
        key="cautious_affluent",
        name="Cautious Affluent",
        tagline="High-income, selective low-frequency mall visitors",
        recommended_action=(
            "Deploy curated high-AOV quality bundles, brand-heritage storytelling, "
            "and targeted cross-sell incentives on anchor luxury departments."
        ),
        target_channel="Personalized Email + Private Lookbook",
        target_vector=(0.20, 1.15, -1.25),
    ),
    PersonaArchetype(
        key="young_trendsetters",
        name="Young Trendsetters",
        tagline="Younger, lower-income, high-velocity trend adopters",
        recommended_action=(
            "Offer limited-time flash drops, flexible micro-installments, "
            "and social referral perks to protect margin while sustaining frequency."
        ),
        target_channel="Mobile App Push + Social Drops",
        target_vector=(-1.05, -1.10, 1.10),
    ),
    PersonaArchetype(
        key="budget_pragmatists",
        name="Budget Pragmatists",
        tagline="Value-conscious, low-income, conservative spenders",
        recommended_action=(
            "Prioritize multi-buy value packs, seasonal clearance alerts, "
            "and everyday essentials cashback rather than premium upsells."
        ),
        target_channel="Loyalty App Coupons + In-Mall Signage",
        target_vector=(0.50, -1.10, -1.15),
    ),
    PersonaArchetype(
        key="core_mainstream",
        name="Core Mainstream",
        tagline="Balanced middle-income, steady multi-category regulars",
        recommended_action=(
            "Maintain retention cadence via quarterly multi-store rewards, "
            "points-to-perk nudges, and weekend family dining/retail bundles."
        ),
        target_channel="Omnichannel Loyalty Newsletter",
        target_vector=(0.30, -0.10, 0.00),
    ),
]


@dataclass(frozen=True)
class PersonaProfile:
    cluster_id: int
    persona_key: str
    persona_name: str
    tagline: str
    recommended_action: str
    target_channel: str
    customer_count: int
    share: float
    mean_age: float
    mean_income: float
    mean_spending: float
    female_share: float
    male_share: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def map_clusters_to_personas(
    df_with_gender: pd.DataFrame,
    labels: np.ndarray,
) -> dict[int, PersonaProfile]:
    """Deterministically map arbitrary cluster IDs to retail personas via centroid geometry."""
    unique_clusters = sorted(int(c) for c in np.unique(labels) if int(c) >= 0)
    if not unique_clusters:
        raise ValueError("No valid non-noise clusters available for persona mapping.")

    global_means = df_with_gender[NUMERIC_FEATURE_COLUMNS].astype(float).mean()
    global_stds = (
        df_with_gender[NUMERIC_FEATURE_COLUMNS].astype(float).std(ddof=0).replace(0.0, 1.0)
    )

    centroid_vectors: list[np.ndarray] = []
    stats_by_cluster: dict[int, dict[str, float | int]] = {}
    total_rows = max(len(df_with_gender), 1)

    for cid in unique_clusters:
        mask = labels == cid
        sub = df_with_gender.loc[mask]
        m_age = float(sub["Age"].astype(float).mean())
        m_inc = float(sub["Annual Income (k$)"].astype(float).mean())
        m_sp = float(sub["Spending Score (1-100)"].astype(float).mean())
        f_count = int((sub["Gender"] == "Female").sum())
        m_count = int((sub["Gender"] == "Male").sum())
        cnt = int(len(sub))

        stats_by_cluster[cid] = {
            "count": cnt,
            "share": round(cnt / total_rows, 4),
            "mean_age": round(m_age, 2),
            "mean_income": round(m_inc, 2),
            "mean_spending": round(m_sp, 2),
            "female_share": round(f_count / max(cnt, 1), 4),
            "male_share": round(m_count / max(cnt, 1), 4),
        }

        z_vec = np.array(
            [
                (m_age - float(global_means["Age"])) / float(global_stds["Age"]),
                (m_inc - float(global_means["Annual Income (k$)"]))
                / float(global_stds["Annual Income (k$)"]),
                (m_sp - float(global_means["Spending Score (1-100)"]))
                / float(global_stds["Spending Score (1-100)"]),
            ],
            dtype=np.float64,
        )
        centroid_vectors.append(z_vec)

    # Optimal 1-to-1 assignment via Hungarian algorithm on Euclidean centroid-to-archetype distance
    n_c = len(unique_clusters)
    n_a = len(CANONICAL_ARCHETYPES)
    cost_matrix = np.zeros((n_c, n_a), dtype=np.float64)
    for i, z_vec in enumerate(centroid_vectors):
        for j, arch in enumerate(CANONICAL_ARCHETYPES):
            target = np.asarray(arch.target_vector, dtype=np.float64)
            # Weight income & spending slightly higher than age for retail segmentation clarity
            weights = np.array([0.8, 1.2, 1.2], dtype=np.float64)
            cost_matrix[i, j] = float(np.linalg.norm((z_vec - target) * weights))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned: dict[int, PersonaProfile] = {}

    for r, c in zip(row_ind, col_ind, strict=True):
        cid = unique_clusters[int(r)]
        arch = CANONICAL_ARCHETYPES[int(c)]
        st = stats_by_cluster[cid]
        assigned[cid] = PersonaProfile(
            cluster_id=cid,
            persona_key=arch.key,
            persona_name=arch.name,
            tagline=arch.tagline,
            recommended_action=arch.recommended_action,
            target_channel=arch.target_channel,
            customer_count=int(st["count"]),
            share=float(st["share"]),
            mean_age=float(st["mean_age"]),
            mean_income=float(st["mean_income"]),
            mean_spending=float(st["mean_spending"]),
            female_share=float(st["female_share"]),
            male_share=float(st["male_share"]),
        )

    # Fallback if target_k > len(CANONICAL_ARCHETYPES)
    for idx, cid in enumerate(unique_clusters):
        if cid not in assigned:
            best_arch_idx = int(np.argmin(cost_matrix[idx]))
            arch = CANONICAL_ARCHETYPES[best_arch_idx]
            st = stats_by_cluster[cid]
            assigned[cid] = PersonaProfile(
                cluster_id=cid,
                persona_key=f"{arch.key}_{cid}",
                persona_name=f"{arch.name} ({cid})",
                tagline=arch.tagline,
                recommended_action=arch.recommended_action,
                target_channel=arch.target_channel,
                customer_count=int(st["count"]),
                share=float(st["share"]),
                mean_age=float(st["mean_age"]),
                mean_income=float(st["mean_income"]),
                mean_spending=float(st["mean_spending"]),
                female_share=float(st["female_share"]),
                male_share=float(st["male_share"]),
            )

    return assigned


@dataclass(frozen=True)
class ScoredCustomerPrediction:
    customer_ref: str | None
    gender: str
    age: int
    annual_income_k: float
    spending_score: float
    cluster_id: int
    persona_key: str
    persona_name: str
    recommended_action: str
    gmm_confidence: float
    normalized_entropy: float
    is_boundary_customer: bool
    distance_to_centroid: float
    drift_flag: bool
    drifted_features: list[str]
    pca_1: float
    pca_2: float
    posterior_probabilities: dict[str, float]


def align_gmm_to_kmeans(
    kmeans_centers: np.ndarray,
    gmm_means: np.ndarray,
) -> dict[int, int]:
    """Align GMM component indices to K-Means cluster IDs via Hungarian assignment."""
    n_km = kmeans_centers.shape[0]
    n_gm = gmm_means.shape[0]
    cost = np.zeros((n_gm, n_km), dtype=np.float64)
    for g_idx in range(n_gm):
        for k_idx in range(n_km):
            cost[g_idx, k_idx] = float(np.linalg.norm(gmm_means[g_idx] - kmeans_centers[k_idx]))
    row_ind, col_ind = linear_sum_assignment(cost)
    return {int(g): int(k) for g, k in zip(row_ind, col_ind, strict=True)}


def extract_income_spending_ellipses(
    gmm_model: Any,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    kmeans_centers: np.ndarray,
    personas: dict[int, PersonaProfile],
    n_std: float = 1.65,
) -> list[dict[str, Any]]:
    """Compute 2D Gaussian covariance ellipses in raw (Income, Spending) coordinates."""
    gmm_to_km = align_gmm_to_kmeans(kmeans_centers, np.asarray(gmm_model.means_, dtype=np.float64))
    ellipses: list[dict[str, Any]] = []
    # Feature indices in NUMERIC_FEATURE_COLUMNS: 0=Age, 1=Annual Income (k$), 2=Spending Score
    inc_idx, sp_idx = 1, 2
    scale_sub = np.array([float(scaler_scale[inc_idx]), float(scaler_scale[sp_idx])])
    mean_sub = np.array([float(scaler_mean[inc_idx]), float(scaler_mean[sp_idx])])

    for g_idx, km_cid in gmm_to_km.items():
        persona = personas.get(km_cid)
        if persona is None:
            continue
        cov_full = np.asarray(gmm_model.covariances_[g_idx], dtype=np.float64)
        cov_2d_scaled = cov_full[np.ix_([inc_idx, sp_idx], [inc_idx, sp_idx])]
        cov_2d_raw = cov_2d_scaled * np.outer(scale_sub, scale_sub)

        eigvals, eigvecs = np.linalg.eigh(cov_2d_raw)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 1e-6)
        eigvecs = eigvecs[:, order]

        angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
        rx = float(n_std * np.sqrt(eigvals[0]))
        ry = float(n_std * np.sqrt(eigvals[1]))

        gmm_mean_scaled = np.asarray(gmm_model.means_[g_idx], dtype=np.float64)
        cx = float(gmm_mean_scaled[inc_idx] * scale_sub[0] + mean_sub[0])
        cy = float(gmm_mean_scaled[sp_idx] * scale_sub[1] + mean_sub[1])

        ellipses.append(
            {
                "cluster_id": km_cid,
                "persona_name": persona.persona_name,
                "cx": round(cx, 2),
                "cy": round(cy, 2),
                "rx": round(rx, 2),
                "ry": round(ry, 2),
                "angle_deg": round(angle_deg, 2),
            }
        )
    return sorted(ellipses, key=lambda item: int(item["cluster_id"]))


def score_customer_records(
    records: list[dict[str, Any]],
    model_bundle: dict[str, Any],
) -> list[ScoredCustomerPrediction]:
    """Score one or more customer dicts using the persisted model bundle."""
    df_input = pd.DataFrame(
        [
            {
                "Gender": str(rec["gender"]),
                "Age": int(rec["age"]),
                "Annual Income (k$)": float(rec["annual_income_k"]),
                "Spending Score (1-100)": float(rec["spending_score"]),
            }
            for rec in records
        ]
    )

    preprocessor = model_bundle["preprocessor"]
    pca = model_bundle["pca"]
    kmeans_model = model_bundle["kmeans_model"]
    gmm_model = model_bundle["gmm_model"]
    personas: dict[int, PersonaProfile] = model_bundle["personas"]
    entropy_threshold = float(model_bundle["entropy_threshold"])

    gmm_to_km = align_gmm_to_kmeans(
        np.asarray(kmeans_model.cluster_centers_, dtype=np.float64),
        np.asarray(gmm_model.means_, dtype=np.float64),
    )

    X_scaled, X_pca = transform_features(df_input, preprocessor=preprocessor, pca=pca)
    cluster_ids = kmeans_model.predict(X_scaled)
    centers = np.asarray(kmeans_model.cluster_centers_, dtype=np.float64)
    posteriors = np.asarray(gmm_model.predict_proba(X_scaled), dtype=np.float64)
    entropies = compute_normalized_entropy(posteriors)
    confidences = np.max(posteriors, axis=1)
    reference_stats: dict[str, dict[str, float]] = model_bundle.get("reference_stats") or {}

    predictions: list[ScoredCustomerPrediction] = []
    for idx, rec in enumerate(records):
        cid = int(cluster_ids[idx])
        persona = personas[cid]
        ent = round(float(entropies[idx]), 4)
        conf = round(float(confidences[idx]), 4)
        dist = round(float(np.linalg.norm(X_scaled[idx] - centers[cid])), 4)
        drift_flag, drifted_feats = evaluate_single_record_drift(
            {
                "Age": float(rec["age"]),
                "Annual Income (k$)": float(rec["annual_income_k"]),
                "Spending Score (1-100)": float(rec["spending_score"]),
            },
            reference_stats=reference_stats,
        )
        prob_by_persona: dict[str, float] = {}
        for g_idx in range(posteriors.shape[1]):
            km_cid = gmm_to_km.get(g_idx, g_idx)
            p_label = personas[km_cid].persona_name if km_cid in personas else f"Segment {km_cid}"
            prob_by_persona[p_label] = round(float(posteriors[idx, g_idx]), 4)

        predictions.append(
            ScoredCustomerPrediction(
                customer_ref=rec.get("customer_ref"),
                gender=str(rec["gender"]),
                age=int(rec["age"]),
                annual_income_k=float(rec["annual_income_k"]),
                spending_score=float(rec["spending_score"]),
                cluster_id=cid,
                persona_key=persona.persona_key,
                persona_name=persona.persona_name,
                recommended_action=persona.recommended_action,
                gmm_confidence=conf,
                normalized_entropy=ent,
                is_boundary_customer=ent > entropy_threshold,
                distance_to_centroid=dist,
                drift_flag=drift_flag,
                drifted_features=drifted_feats,
                pca_1=round(float(X_pca[idx, 0]), 4),
                pca_2=round(float(X_pca[idx, 1]), 4),
                posterior_probabilities=prob_by_persona,
            )
        )
    return predictions


class ModelSchemaMismatchError(RuntimeError):
    """Raised when a persisted model bundle fails the training-serving schema hash guard."""


def compute_feature_schema_payload(cfg: AppConfig, model_version: str = "v1.0.0") -> dict[str, Any]:
    """Compute canonical feature contract payload and deterministic SHA-256 hash."""
    feature_names = list(NUMERIC_FEATURE_COLUMNS) + (
        ["Gender_Male"] if cfg.privacy.include_gender else []
    )
    payload: dict[str, Any] = {
        "model_version": model_version,
        "include_gender": cfg.privacy.include_gender,
        "feature_names": feature_names,
        "numeric_columns": list(NUMERIC_FEATURE_COLUMNS),
        "pca_components": cfg.features.pca_components,
        "target_k": cfg.clustering.target_k,
    }
    schema_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    payload["feature_schema_hash"] = schema_hash
    return payload


def verify_bundle_schema_compatibility(
    bundle: dict[str, Any],
    *,
    strict_config: AppConfig | None = None,
) -> str:
    """Verify saved bundle feature_schema_hash matches code feature contract & bundle config."""
    required_keys = {
        "feature_schema_hash",
        "preprocessor",
        "pca",
        "kmeans_model",
        "gmm_model",
        "personas",
        "config",
    }
    missing = required_keys - set(bundle.keys())
    if missing:
        raise ModelSchemaMismatchError(f"Model bundle missing required keys: {sorted(missing)}")

    saved_hash = str(bundle["feature_schema_hash"])
    model_version = str(bundle.get("model_version", "v1.0.0"))
    bundle_cfg = AppConfig.model_validate(bundle["config"])
    expected_from_bundle = compute_feature_schema_payload(bundle_cfg, model_version=model_version)
    if not hmac.compare_digest(saved_hash, str(expected_from_bundle["feature_schema_hash"])):
        raise ModelSchemaMismatchError(
            f"Training-serving skew detected: bundle hash {saved_hash} != "
            f"expected {expected_from_bundle['feature_schema_hash']}."
        )

    feat_summary = bundle.get("feature_summary") or {}
    saved_features = feat_summary.get("feature_names")
    if saved_features is not None and list(saved_features) != list(
        expected_from_bundle["feature_names"]
    ):
        raise ModelSchemaMismatchError(
            f"Feature list mismatch: bundle={saved_features} vs "
            f"code={expected_from_bundle['feature_names']}"
        )

    if strict_config is not None:
        expected_runtime = compute_feature_schema_payload(strict_config, model_version=model_version)
        if not hmac.compare_digest(saved_hash, str(expected_runtime["feature_schema_hash"])):
            raise ModelSchemaMismatchError(
                f"Runtime config schema mismatch: bundle={saved_hash} != "
                f"runtime={expected_runtime['feature_schema_hash']}"
            )
    return saved_hash


def compute_dataframe_run_key(raw_df: pd.DataFrame, cfg: AppConfig) -> str:
    """Compute deterministic content-addressed run key from dataframe + config."""
    hasher = hashlib.sha256()
    hasher.update(raw_df.to_csv(index=False).encode("utf-8"))
    hasher.update(cfg.model_dump_json().encode("utf-8"))
    return f"mall-seg-{hasher.hexdigest()[:12]}"


def execute_segmentation_workflow(
    raw_df: pd.DataFrame,
    cfg: AppConfig,
    *,
    dataset_source: str,
    run_key: str | None = None,
    model_version: str = "v1.0.0",
    activate_version: bool = True,
    persist_pipeline_run: bool = True,
    parent_version: str | None = None,
    version_notes: str | None = None,
    engine: Engine | None = None,
    database_url: str | None = None,
    bundle_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    enforce_stability_gate: bool = True,
    log_mlflow: bool = False,
) -> dict[str, Any]:
    """Execute full deterministic segmentation workflow on a validated dataframe."""
    if cfg.clustering.target_k >= len(raw_df):
        raise ValueError(
            f"target_k ({cfg.clustering.target_k}) must be strictly less than "
            f"row count ({len(raw_df)})."
        )

    resolved_run_key = run_key or compute_dataframe_run_key(raw_df, cfg)

    # 1. Exploratory Data Analysis (preserving Gender for demographic breakdown)
    eda_report = compute_eda_report(raw_df)

    # 2. Privacy minimization: strip CustomerID before scaling & distance geometry
    df_no_id = strip_customer_id(raw_df)
    feature_artifacts = fit_transform_features(
        df_no_id=df_no_id,
        include_gender=cfg.privacy.include_gender,
        pca_components=cfg.features.pca_components,
        seed=cfg.clustering.seed,
    )

    # 3. Multi-algorithm clustering suite
    dbscan_min_samples = min(cfg.clustering.dbscan_min_samples, max(2, len(raw_df) // 3))
    suite = run_clustering_suite(
        X_scaled=feature_artifacts.X_scaled,
        target_k=cfg.clustering.target_k,
        k_min=cfg.clustering.k_min,
        k_max=min(cfg.clustering.k_max, len(raw_df) - 1),
        kmeans_n_init=cfg.clustering.kmeans_n_init,
        dbscan_min_samples=dbscan_min_samples,
        gmm_covariance_type=cfg.clustering.gmm_covariance_type,
        seed=cfg.clustering.seed,
    )

    # 4. Bootstrap stability check (Hungarian linear sum assignment)
    stability = evaluate_kmeans_stability(
        X_scaled=feature_artifacts.X_scaled,
        reference_labels=suite.kmeans_labels,
        n_clusters=cfg.clustering.target_k,
        n_bootstrap=cfg.stability.bootstrap_iterations,
        bootstrap_n_init=cfg.stability.bootstrap_n_init,
        min_jaccard_stability=cfg.stability.min_jaccard_stability,
        seed=cfg.clustering.seed,
    )
    logger.info(
        "Bootstrap stability (B=%d): mean_jaccard=%.4f (gate >= %.2f -> %s)",
        stability.n_bootstrap,
        stability.mean_jaccard,
        stability.min_jaccard_threshold,
        "PASS" if stability.passed_gate else "FAIL",
    )
    if enforce_stability_gate and not stability.passed_gate:
        raise RuntimeError(
            f"Stability gate failed: mean Jaccard {stability.mean_jaccard:.4f} "
            f"< {stability.min_jaccard_threshold:.4f}"
        )

    # 5. Deterministic persona mapping (invariant to arbitrary label permutations)
    personas = map_clusters_to_personas(raw_df, suite.kmeans_labels)
    persona_names = {cid: prof.persona_name for cid, prof in personas.items()}

    # 6. Surrogate Decision Tree explainability
    explainability = fit_surrogate_explainer(
        df_features=df_no_id,
        labels=suite.kmeans_labels,
        persona_names=persona_names,
        max_depth=cfg.explainability.max_depth,
        seed=cfg.clustering.seed,
    )

    # 7. Aligned GMM posteriors & normalized entropy per customer
    gmm_posteriors = np.asarray(
        suite.gmm_model.predict_proba(feature_artifacts.X_scaled), dtype=np.float64
    )
    entropies = compute_normalized_entropy(gmm_posteriors)
    confidences = np.max(gmm_posteriors, axis=1)

    scored_rows: list[dict[str, Any]] = []
    for idx in range(len(raw_df)):
        cid = int(suite.kmeans_labels[idx])
        prof = personas[cid]
        ent = round(float(entropies[idx]), 4)
        conf = round(float(confidences[idx]), 4)
        scored_rows.append(
            {
                "customer_ref": f"CUST-{int(raw_df.iloc[idx]['CustomerID']):04d}",
                "gender": str(raw_df.iloc[idx]["Gender"]),
                "age": int(raw_df.iloc[idx]["Age"]),
                "annual_income_k": float(raw_df.iloc[idx]["Annual Income (k$)"]),
                "spending_score": float(raw_df.iloc[idx]["Spending Score (1-100)"]),
                "cluster_id": cid,
                "persona_key": prof.persona_key,
                "persona_name": prof.persona_name,
                "recommended_action": prof.recommended_action,
                "gmm_confidence": conf,
                "normalized_entropy": ent,
                "is_boundary_customer": ent > cfg.clustering.entropy_threshold,
                "pca_1": round(float(feature_artifacts.X_pca[idx, 0]), 4),
                "pca_2": round(float(feature_artifacts.X_pca[idx, 1]), 4),
            }
        )

    kmeans_bench = suite.benchmarks[0]
    kmeans_silhouette = float(kmeans_bench.silhouette or 0.0)
    scaler = feature_artifacts.preprocessor.named_transformers_["num"]
    gmm_ellipses = extract_income_spending_ellipses(
        gmm_model=suite.gmm_model,
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        kmeans_centers=np.asarray(suite.kmeans_model.cluster_centers_, dtype=np.float64),
        personas=personas,
    )
    dendro = dendrogram(
        suite.ward_linkage_matrix,
        truncate_mode="lastp",
        p=min(24, len(raw_df)),
        no_plot=True,
    )
    k_target = cfg.clustering.target_k
    if len(suite.ward_linkage_matrix) >= k_target:
        cut_height = float(
            (
                suite.ward_linkage_matrix[-(k_target - 1), 2]
                + suite.ward_linkage_matrix[-k_target, 2]
            )
            / 2.0
        )
    else:
        cut_height = float(suite.ward_linkage_matrix[-(k_target - 1), 2])

    gmm_to_km = align_gmm_to_kmeans(
        np.asarray(suite.kmeans_model.cluster_centers_, dtype=np.float64),
        np.asarray(suite.gmm_model.means_, dtype=np.float64),
    )

    # Hungarian-align Ward labels to K-Means labels for consistent visual comparison
    km_unique = sorted(int(c) for c in np.unique(suite.kmeans_labels))
    ward_unique = sorted(int(c) for c in np.unique(suite.agglomerative_labels))
    overlap = np.zeros((len(ward_unique), len(km_unique)), dtype=np.float64)
    for wi, wc in enumerate(ward_unique):
        for ki, kc in enumerate(km_unique):
            overlap[wi, ki] = float(
                np.sum((suite.agglomerative_labels == wc) & (suite.kmeans_labels == kc))
            )
    w_row, w_col = linear_sum_assignment(-overlap)
    ward_to_km = {ward_unique[int(r)]: km_unique[int(c)] for r, c in zip(w_row, w_col, strict=True)}

    algorithm_points: list[dict[str, Any]] = []
    for idx, row_dict in enumerate(scored_rows):
        w_raw = int(suite.agglomerative_labels[idx])
        g_raw = int(suite.gmm_labels[idx])
        d_raw = int(suite.dbscan_labels[idx])
        algorithm_points.append(
            {
                "customer_ref": row_dict["customer_ref"],
                "kmeans_cluster_id": row_dict["cluster_id"],
                "ward_cluster_id": ward_to_km.get(w_raw, w_raw),
                "gmm_cluster_id": gmm_to_km.get(g_raw, g_raw),
                "dbscan_cluster_id": d_raw,
            }
        )

    summary_payload: dict[str, Any] = {
        "eda": eda_report.to_dict(),
        "features": feature_artifacts.summary_dict(),
        "benchmarks": [b.to_dict() for b in suite.benchmarks],
        "kmeans_sweep": [asdict(pt) for pt in suite.kmeans_sweep],
        "gmm_sweep": [asdict(pt) for pt in suite.gmm_sweep],
        "stability": stability.to_dict(),
        "personas": [
            prof.to_dict() for prof in sorted(personas.values(), key=lambda p: p.cluster_id)
        ],
        "explainability": explainability.to_dict(),
        "gmm_ellipses": gmm_ellipses,
        "ward_dendrogram": {
            "icoord": dendro["icoord"],
            "dcoord": dendro["dcoord"],
            "ivl": dendro["ivl"],
            "cut_height": round(cut_height, 4),
        },
        "algorithm_points": algorithm_points,
        "dbscan_eps": round(float(suite.dbscan_eps), 4),
        "entropy_threshold": cfg.clustering.entropy_threshold,
        "boundary_customer_count": sum(1 for r in scored_rows if r["is_boundary_customer"]),
    }

    feature_schema_payload = compute_feature_schema_payload(cfg, model_version=model_version)
    feature_schema_hash = str(feature_schema_payload["feature_schema_hash"])
    reference_stats = compute_reference_stats(raw_df, list(NUMERIC_FEATURE_COLUMNS))

    resolved_bundle_path = (
        Path(bundle_path) if bundle_path is not None else Path(cfg.artifacts.model_bundle_path)
    )
    version_dir = resolved_bundle_path.parent / "models" / model_version
    version_bundle_path = version_dir / "model.joblib"
    recorded_artifact_path = (
        str(resolved_bundle_path) if activate_version else str(version_bundle_path)
    )

    # 8. Idempotent DB persistence
    db_engine = engine or create_db_engine(database_url)
    init_db(db_engine)
    session_factory = get_session_factory(db_engine)
    with session_scope(session_factory) as session:
        if persist_pipeline_run:
            upsert_pipeline_run(
                session,
                run_key=resolved_run_key,
                dataset_source=dataset_source,
                row_count=len(raw_df),
                target_k=cfg.clustering.target_k,
                include_gender=cfg.privacy.include_gender,
                silhouette_score=kmeans_silhouette,
                mean_jaccard_stability=stability.mean_jaccard,
                passed_stability_gate=stability.passed_gate,
                surrogate_fidelity=explainability.fidelity_score,
                summary_payload=summary_payload,
                scored_rows=scored_rows,
            )
        upsert_model_version(
            session,
            version=model_version,
            algorithm="K-Means (k-means++) + GMM",
            feature_schema_hash=feature_schema_hash,
            metrics={
                "run_key": resolved_run_key,
                "row_count": len(raw_df),
                "target_k": cfg.clustering.target_k,
                "include_gender": cfg.privacy.include_gender,
                "kmeans_silhouette": kmeans_silhouette,
                "kmeans_calinski_harabasz": float(kmeans_bench.calinski_harabasz or 0.0),
                "kmeans_davies_bouldin": float(kmeans_bench.davies_bouldin or 0.0),
                "mean_jaccard_stability": stability.mean_jaccard,
                "surrogate_fidelity": explainability.fidelity_score,
                "dbscan_eps": round(float(suite.dbscan_eps), 4),
            },
            artifact_path=recorded_artifact_path,
            activate=activate_version,
            status_label="active" if activate_version else "candidate",
            parent_version=parent_version,
            notes=version_notes,
        )

    # 9. Export model bundle & versioned artifact directory artifacts/models/<version>/
    baseline_distributions = {
        col: raw_df[col].astype(float).tolist() for col in NUMERIC_FEATURE_COLUMNS
    }
    model_bundle: dict[str, Any] = {
        "run_key": resolved_run_key,
        "model_version": model_version,
        "feature_schema_hash": feature_schema_hash,
        "reference_stats": reference_stats,
        "dataset_source": dataset_source,
        "config": cfg.model_dump(),
        "preprocessor": feature_artifacts.preprocessor,
        "pca": feature_artifacts.pca,
        "feature_summary": feature_artifacts.summary_dict(),
        "kmeans_model": suite.kmeans_model,
        "gmm_model": suite.gmm_model,
        "agglomerative_labels": suite.agglomerative_labels,
        "ward_linkage_matrix": suite.ward_linkage_matrix,
        "dbscan_labels": suite.dbscan_labels,
        "dbscan_eps": suite.dbscan_eps,
        "personas": personas,
        "entropy_threshold": cfg.clustering.entropy_threshold,
        "baseline_distributions": baseline_distributions,
        "eda": eda_report.to_dict(),
        "benchmarks": [b.to_dict() for b in suite.benchmarks],
        "kmeans_sweep": [asdict(pt) for pt in suite.kmeans_sweep],
        "gmm_sweep": [asdict(pt) for pt in suite.gmm_sweep],
        "stability": stability.to_dict(),
        "explainability": explainability.to_dict(),
        "scored_rows": scored_rows,
    }

    if activate_version:
        resolved_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_bundle, resolved_bundle_path)
        logger.info("Exported active model bundle to %s", resolved_bundle_path)

    version_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, version_bundle_path)
    (version_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_version": model_version,
                "run_key": resolved_run_key,
                "kmeans_silhouette": kmeans_silhouette,
                "kmeans_calinski_harabasz": float(kmeans_bench.calinski_harabasz or 0.0),
                "kmeans_davies_bouldin": float(kmeans_bench.davies_bouldin or 0.0),
                "mean_jaccard_stability": stability.mean_jaccard,
                "surrogate_fidelity": explainability.fidelity_score,
                "benchmarks": [b.to_dict() for b in suite.benchmarks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (version_dir / "feature_schema.json").write_text(
        json.dumps(feature_schema_payload, indent=2),
        encoding="utf-8",
    )
    (version_dir / "reference_stats.json").write_text(
        json.dumps(reference_stats, indent=2),
        encoding="utf-8",
    )

    # 10. Export JSON snapshot when configured
    if snapshot_path is not None:
        resolved_snapshot_path = Path(snapshot_path)
        resolved_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_payload = {
            "run_key": resolved_run_key,
            "dataset_source": dataset_source,
            "row_count": len(raw_df),
            "target_k": cfg.clustering.target_k,
            "include_gender": cfg.privacy.include_gender,
            "silhouette_score": kmeans_silhouette,
            "mean_jaccard_stability": stability.mean_jaccard,
            "passed_stability_gate": stability.passed_gate,
            "surrogate_fidelity": explainability.fidelity_score,
            **summary_payload,
            "points": scored_rows,
            "ward_linkage": suite.ward_linkage_matrix.tolist(),
            "inference_params": {
                "feature_names": list(NUMERIC_FEATURE_COLUMNS),
                "scaler_mean": [float(x) for x in scaler.mean_],
                "scaler_scale": [float(x) for x in scaler.scale_],
                "pca_components": feature_artifacts.pca.components_.tolist(),
                "pca_mean": [float(x) for x in feature_artifacts.pca.mean_],
                "kmeans_centroids": suite.kmeans_model.cluster_centers_.tolist(),
                "gmm_weights": suite.gmm_model.weights_.tolist(),
                "gmm_means": suite.gmm_model.means_.tolist(),
                "gmm_precisions_cholesky": suite.gmm_model.precisions_cholesky_.tolist(),
                "baseline_distributions": baseline_distributions,
            },
        }
        resolved_snapshot_path.write_text(json.dumps(snapshot_payload, indent=2), encoding="utf-8")

    # 11. Log run & artifacts to MLflow when requested
    if log_mlflow:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.artifacts.mlflow_tracking_uri)
        if tracking_uri.startswith("sqlite:///"):
            mlflow_db = Path(tracking_uri.removeprefix("sqlite:///"))
            mlflow_db.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg.artifacts.mlflow_experiment)
        with mlflow.start_run(run_name=resolved_run_key):
            mlflow.log_params(
                {
                    "dataset_source": dataset_source,
                    "row_count": len(raw_df),
                    "target_k": cfg.clustering.target_k,
                    "include_gender": cfg.privacy.include_gender,
                    "seed": cfg.clustering.seed,
                    "entropy_threshold": cfg.clustering.entropy_threshold,
                    "bootstrap_iterations": cfg.stability.bootstrap_iterations,
                    "dbscan_eps": suite.dbscan_eps,
                }
            )
            mlflow.log_metrics(
                {
                    "kmeans_silhouette": kmeans_silhouette,
                    "kmeans_calinski_harabasz": float(kmeans_bench.calinski_harabasz or 0.0),
                    "kmeans_davies_bouldin": float(kmeans_bench.davies_bouldin or 0.0),
                    "mean_jaccard_stability": stability.mean_jaccard,
                    "surrogate_fidelity": explainability.fidelity_score,
                    "pca_cumulative_variance": float(
                        sum(feature_artifacts.explained_variance_ratio)
                    ),
                    "boundary_customer_count": float(summary_payload["boundary_customer_count"]),
                }
            )
            mlflow.log_artifact(str(resolved_bundle_path))

    return {
        "run_key": resolved_run_key,
        "dataset_source": dataset_source,
        "row_count": len(raw_df),
        "kmeans_silhouette": kmeans_silhouette,
        "mean_jaccard_stability": stability.mean_jaccard,
        "passed_stability_gate": stability.passed_gate,
        "surrogate_fidelity": explainability.fidelity_score,
        "boundary_customer_count": summary_payload["boundary_customer_count"],
        "bundle_path": str(resolved_bundle_path),
        "bundle": model_bundle,
        "summary_payload": summary_payload,
    }
