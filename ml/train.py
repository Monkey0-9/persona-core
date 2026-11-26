"""Offline versioned ML training script writing artifacts/models/<version>/ bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mall_seg.config import load_config
from mall_seg.personas import execute_segmentation_workflow
from mall_seg.schema import load_raw_dataset


def train_and_export_versioned_model(
    config_path: str | None = None,
    version: str = "v1.0.0",
    log_mlflow: bool = False,
) -> dict[str, Any]:
    """Run deterministic ML training and persist versioned artifacts/models/<version>/."""
    cfg = load_config(config_path)
    raw_df, source_kind, _ = load_raw_dataset(cfg)
    return execute_segmentation_workflow(
        raw_df=raw_df,
        cfg=cfg,
        dataset_source=source_kind,
        model_version=version,
        enforce_stability_gate=True,
        log_mlflow=log_mlflow,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train versioned Mall Segmentation ML bundle.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--version", type=str, default="v1.0.0", help="Semantic artifact version.")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow run tracking.")
    args = parser.parse_args()
    res = train_and_export_versioned_model(
        config_path=args.config,
        version=args.version,
        log_mlflow=args.mlflow,
    )
    summary = {k: v for k, v in res.items() if k not in {"bundle", "summary_payload"}}
    summary["version_dir"] = str(Path("artifacts/models") / args.version)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
