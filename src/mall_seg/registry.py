"""Model registry governance: candidate registration, comparison, promotion & rollback."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from mall_seg.contracts import AppConfig
from mall_seg.db import (
    ModelVersionRecord,
    get_active_model_version,
    get_model_version,
    list_model_versions,
    set_active_model_version,
)
from mall_seg.personas import (
    ModelSchemaMismatchError,
    execute_segmentation_workflow,
    verify_bundle_schema_compatibility,
)
from mall_seg.schema import load_raw_dataset

logger = logging.getLogger("mall_seg.registry")


def resolve_version_bundle_path(
    mv: ModelVersionRecord,
    *,
    default_bundle_path: Path,
) -> Path:
    """Resolve the model artifact path for a given registry version safely."""
    versioned_path = default_bundle_path.parent / "models" / mv.version / "model.joblib"
    if versioned_path.exists():
        return versioned_path
    return Path(mv.artifact_path)


def _compute_centroid_stability_shift(
    candidate_bundle: dict[str, Any],
    baseline_bundle: dict[str, Any],
) -> float | None:
    """Measure mean L2 centroid shift for shared persona keys between two bundles."""
    cand_personas = candidate_bundle.get("personas") or {}
    base_personas = baseline_bundle.get("personas") or {}
    cand_centers = np.asarray(candidate_bundle["kmeans_model"].cluster_centers_, dtype=np.float64)
    base_centers = np.asarray(baseline_bundle["kmeans_model"].cluster_centers_, dtype=np.float64)
    if cand_centers.shape[1] != base_centers.shape[1]:
        return None

    cand_by_key = {p.persona_key: cid for cid, p in cand_personas.items()}
    base_by_key = {p.persona_key: cid for cid, p in base_personas.items()}
    shared_keys = set(cand_by_key.keys()) & set(base_by_key.keys())
    if not shared_keys:
        return None
    distances = [
        float(np.linalg.norm(cand_centers[cand_by_key[k]] - base_centers[base_by_key[k]]))
        for k in shared_keys
    ]
    return round(float(np.mean(distances)), 4)


def serialize_model_version(mv: ModelVersionRecord) -> dict[str, Any]:
    return {
        "id": mv.id,
        "version": mv.version,
        "algorithm": mv.algorithm,
        "feature_schema_hash": mv.feature_schema_hash,
        "artifact_path": mv.artifact_path,
        "is_active": mv.is_active,
        "status": mv.status,
        "parent_version": mv.parent_version,
        "notes": mv.notes,
        "metrics": json.loads(mv.metrics_json),
        "promoted_at": mv.promoted_at.isoformat() if mv.promoted_at else None,
        "created_at": mv.created_at.isoformat(),
    }


def register_candidate_model(
    *,
    version: str,
    cfg: AppConfig,
    engine: Engine,
    bundle_path: Path,
    target_k: int | None = None,
    include_gender: bool | None = None,
    notes: str | None = None,
    parent_version: str | None = None,
) -> dict[str, Any]:
    """Train a candidate model version without disrupting the currently active model."""
    candidate_cfg = cfg.model_copy(
        update={
            "privacy": cfg.privacy.model_copy(
                update={
                    "include_gender": (
                        cfg.privacy.include_gender if include_gender is None else include_gender
                    )
                }
            ),
            "clustering": cfg.clustering.model_copy(
                update={
                    "target_k": target_k or cfg.clustering.target_k,
                    "k_max": max(cfg.clustering.k_max, target_k or cfg.clustering.target_k),
                }
            ),
        }
    )
    raw_df, source_kind, _ = load_raw_dataset(candidate_cfg)
    return execute_segmentation_workflow(
        raw_df=raw_df,
        cfg=candidate_cfg,
        dataset_source=source_kind,
        model_version=version,
        activate_version=False,
        persist_pipeline_run=False,
        parent_version=parent_version,
        version_notes=notes,
        engine=engine,
        bundle_path=bundle_path,
        enforce_stability_gate=True,
        log_mlflow=False,
    )


def compare_model_versions(
    session: Session,
    *,
    candidate_version: str,
    baseline_version: str | None = None,
    default_bundle_path: Path,
) -> dict[str, Any]:
    """Compare candidate metrics & centroid stability against baseline (default: active model)."""
    cand_rec = get_model_version(session, candidate_version)
    if cand_rec is None:
        raise ValueError(f"Model version '{candidate_version}' not found.")

    base_rec = (
        get_model_version(session, baseline_version)
        if baseline_version
        else get_active_model_version(session)
    )
    cand_metrics = json.loads(cand_rec.metrics_json)
    base_metrics = json.loads(base_rec.metrics_json) if base_rec else {}

    centroid_shift: float | None = None
    cand_path = resolve_version_bundle_path(cand_rec, default_bundle_path=default_bundle_path)
    if base_rec is not None:
        base_path = resolve_version_bundle_path(base_rec, default_bundle_path=default_bundle_path)
        if cand_path.exists() and base_path.exists():
            cand_bundle = joblib.load(cand_path)
            base_bundle = joblib.load(base_path)
            centroid_shift = _compute_centroid_stability_shift(cand_bundle, base_bundle)

    def _delta(key: str) -> float | None:
        c_val = cand_metrics.get(key)
        b_val = base_metrics.get(key)
        if isinstance(c_val, (int, float)) and isinstance(b_val, (int, float)):
            return round(float(c_val) - float(b_val), 4)
        return None

    return {
        "candidate": serialize_model_version(cand_rec),
        "baseline": serialize_model_version(base_rec) if base_rec else None,
        "deltas": {
            "kmeans_silhouette": _delta("kmeans_silhouette"),
            "kmeans_calinski_harabasz": _delta("kmeans_calinski_harabasz"),
            "kmeans_davies_bouldin": _delta("kmeans_davies_bouldin"),
            "mean_jaccard_stability": _delta("mean_jaccard_stability"),
            "surrogate_fidelity": _delta("surrogate_fidelity"),
        },
        "centroid_shift_l2": centroid_shift,
    }


def activate_or_rollback_version(
    session: Session,
    *,
    target_version: str | None,
    default_bundle_path: Path,
    rollback: bool = False,
) -> tuple[ModelVersionRecord, dict[str, Any]]:
    """Verify training-serving schema hash, promote target version, and return loaded bundle."""
    current_active = get_active_model_version(session)
    if target_version is None:
        if not rollback:
            raise ValueError("Target version is required for promotion.")
        versions = list_model_versions(session)
        candidates = [
            v for v in versions if current_active is None or v.version != current_active.version
        ]
        # Prefer parent_version when set, otherwise most recent non-active version
        fallback = None
        if current_active and current_active.parent_version:
            fallback = next(
                (v for v in candidates if v.version == current_active.parent_version),
                None,
            )
        target_rec = fallback or (candidates[0] if candidates else None)
        if target_rec is None:
            raise ValueError("No previous model version available for rollback.")
        resolved_target_version = target_rec.version
    else:
        resolved_target_version = target_version
        target_rec = get_model_version(session, resolved_target_version)
        if target_rec is None:
            raise ValueError(f"Model version '{resolved_target_version}' does not exist.")

    bundle_file = resolve_version_bundle_path(target_rec, default_bundle_path=default_bundle_path)
    if not bundle_file.exists():
        raise FileNotFoundError(
            f"Model artifact for '{resolved_target_version}' missing at {bundle_file}"
        )

    bundle = joblib.load(bundle_file)
    try:
        verify_bundle_schema_compatibility(bundle)
    except ModelSchemaMismatchError:
        raise

    # Mirror promoted artifact to default bundle_path so cold restarts stay synced
    if bundle_file.resolve() != default_bundle_path.resolve():
        default_bundle_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, default_bundle_path)

    updated = set_active_model_version(
        session,
        target_version=resolved_target_version,
        mark_previous_status="rolled_back" if rollback else "archived",
    )
    if updated is None:
        raise ValueError(f"Failed to activate model version '{resolved_target_version}'.")
    logger.info(
        "%s active model to version=%s (schema=%s)",
        "Rolled back" if rollback else "Promoted",
        updated.version,
        updated.feature_schema_hash,
    )
    return updated, bundle
