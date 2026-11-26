"""End-to-end CLI entrypoint for Mall Customer Segmentation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mall_seg.config import AppConfig, load_config
from mall_seg.profiling import execute_segmentation_workflow, generate_segment_kpis
from mall_seg.schema import load_raw_dataset
from mall_seg.visualization import render_all_visuals

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mall_seg.run_pipeline")


def _compute_run_key(dataset_path: Path, cfg: AppConfig) -> str:
    hasher = hashlib.sha256()
    hasher.update(dataset_path.read_bytes())
    hasher.update(cfg.model_dump_json().encode("utf-8"))
    return f"mall-seg-{hasher.hexdigest()[:12]}"


def run_end_to_end(
    config_path: str | Path | None = None,
    target_k: int | None = None,
    include_gender: bool | None = None,
    render_visuals: bool = False,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    """Execute validation, preprocessing, clustering, profiling, persistence, and visuals."""
    cfg = load_config(config_path)
    if target_k is not None:
        cfg = cfg.model_copy(
            update={"clustering": cfg.clustering.model_copy(update={"target_k": target_k})}
        )
    if include_gender is not None:
        cfg = cfg.model_copy(
            update={"privacy": cfg.privacy.model_copy(update={"include_gender": include_gender})}
        )

    raw_df, source_kind, dataset_path = load_raw_dataset(cfg)
    run_key = _compute_run_key(dataset_path, cfg)
    logger.info(
        "Validated %s dataset (%d rows) from %s [run_key=%s]",
        source_kind,
        len(raw_df),
        dataset_path,
        run_key,
    )

    workflow_result = execute_segmentation_workflow(
        raw_df=raw_df,
        cfg=cfg,
        dataset_source=source_kind,
        run_key=run_key,
        bundle_path=cfg.artifacts.model_bundle_path,
        snapshot_path=cfg.artifacts.frontend_snapshot_path,
        enforce_stability_gate=True,
        log_mlflow=log_mlflow,
    )

    bundle = workflow_result["bundle"]
    scored_df = workflow_result["summary_payload"]
    _ = scored_df  # validated summary dict

    # Generate rich commercial KPI breakdown via profiling module
    import numpy as np

    labels = np.array([int(r["cluster_id"]) for r in bundle["scored_rows"]], dtype=int)
    boundary = np.array(
        [bool(r["is_boundary_customer"]) for r in bundle["scored_rows"]],
        dtype=bool,
    )
    kpis = [k.to_dict() for k in generate_segment_kpis(raw_df, labels, is_boundary=boundary)]

    visual_paths: list[str] = []
    if render_visuals:
        visual_paths = [str(p) for p in render_all_visuals(bundle=bundle)]

    return {
        "run_key": workflow_result["run_key"],
        "dataset_source": workflow_result["dataset_source"],
        "row_count": workflow_result["row_count"],
        "target_k": cfg.clustering.target_k,
        "include_gender": cfg.privacy.include_gender,
        "kmeans_silhouette": workflow_result["kmeans_silhouette"],
        "mean_jaccard_stability": workflow_result["mean_jaccard_stability"],
        "surrogate_fidelity": workflow_result["surrogate_fidelity"],
        "boundary_customer_count": workflow_result["boundary_customer_count"],
        "bundle_path": workflow_result["bundle_path"],
        "visuals_generated": visual_paths,
        "segment_kpis": kpis,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mall Segmentation Engine CLI runner.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline YAML config.")
    parser.add_argument("--k", type=int, default=None, help="Override target cluster count.")
    parser.add_argument(
        "--include-gender",
        action="store_true",
        default=None,
        help="Opt-in Gender feature encoding during scaling/clustering.",
    )
    parser.add_argument(
        "--render-visuals",
        action="store_true",
        help="Render PNG & 3D HTML visual artifacts after training.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow experiment tracking.",
    )
    args = parser.parse_args()
    output = run_end_to_end(
        config_path=args.config,
        target_k=args.k,
        include_gender=args.include_gender,
        render_visuals=args.render_visuals,
        log_mlflow=not args.no_mlflow,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
