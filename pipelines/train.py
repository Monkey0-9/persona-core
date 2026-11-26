"""End-to-end training pipeline: ingestion, validation, clustering, and persistence."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mall_seg.contracts import AppConfig, load_config, load_raw_dataset
from mall_seg.personas import execute_segmentation_workflow

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mall_seg.train")


def _compute_run_key(dataset_path: Path, cfg: AppConfig) -> str:
    hasher = hashlib.sha256()
    hasher.update(dataset_path.read_bytes())
    hasher.update(cfg.model_dump_json().encode("utf-8"))
    return f"mall-seg-{hasher.hexdigest()[:12]}"


def run_training_pipeline(config_path: str | Path | None = None) -> dict[str, Any]:
    """Execute the full deterministic segmentation training pipeline."""
    cfg = load_config(config_path)

    # 1. Ingestion & Pandera Contract Validation
    raw_df, source_kind, dataset_path = load_raw_dataset(cfg)
    run_key = _compute_run_key(dataset_path, cfg)
    logger.info(
        "Loaded dataset (%s) from %s with %d rows [run_key=%s]",
        source_kind,
        dataset_path,
        len(raw_df),
        run_key,
    )

    result = execute_segmentation_workflow(
        raw_df=raw_df,
        cfg=cfg,
        dataset_source=source_kind,
        run_key=run_key,
        bundle_path=cfg.artifacts.model_bundle_path,
        snapshot_path=cfg.artifacts.frontend_snapshot_path,
        enforce_stability_gate=True,
        log_mlflow=True,
    )

    return {
        "run_key": result["run_key"],
        "dataset_source": result["dataset_source"],
        "row_count": result["row_count"],
        "kmeans_silhouette": result["kmeans_silhouette"],
        "mean_jaccard_stability": result["mean_jaccard_stability"],
        "surrogate_fidelity": result["surrogate_fidelity"],
        "boundary_customer_count": result["boundary_customer_count"],
        "bundle_path": result["bundle_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Mall Customer Segmentation training pipeline."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()
    result = run_training_pipeline(config_path=args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
