"""Drift monitoring layer & scheduled rolling-window evaluation CLI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from mall_seg.alerts import dispatch_drift_alert
from mall_seg.contracts import NUMERIC_FEATURE_COLUMNS, load_config
from mall_seg.db import (
    create_db_engine,
    get_latest_pipeline_run,
    get_session_factory,
    init_db,
    query_recent_scoring_events,
    record_drift_report,
    session_scope,
)
from mall_seg.drift import (
    BatchDriftReport,
    FeatureDriftResult,
    classify_alert_status,
    compute_ks_test,
    compute_psi,
    compute_reference_stats,
    evaluate_batch_drift,
    evaluate_single_record_drift,
)

logger = logging.getLogger("mall_seg.drift_job")

__all__ = [
    "BatchDriftReport",
    "FeatureDriftResult",
    "classify_alert_status",
    "compute_ks_test",
    "compute_psi",
    "compute_reference_stats",
    "evaluate_batch_drift",
    "evaluate_rolling_window_drift",
    "evaluate_single_record_drift",
]


def evaluate_rolling_window_drift(
    session: Session,
    *,
    bundle: dict[str, Any],
    window_size: int = 100,
    lookback_days: int | None = 7,
    stable_threshold: float = 0.10,
    monitor_threshold: float = 0.25,
    bins: int = 10,
) -> dict[str, Any]:
    """Compute PSI + KS drift on rolling window of recent scoring_events and trigger alerts."""
    events = query_recent_scoring_events(
        session,
        window_size=window_size,
        lookback_days=lookback_days,
    )
    window_label = f"rolling:{len(events)}/{lookback_days or 'all'}d"
    if len(events) < 2:
        return {
            "evaluated": False,
            "reason": "At least 2 recent scoring events are required for rolling drift check.",
            "sample_size": len(events),
            "alert_status": "NORMAL",
            "window_label": window_label,
        }

    current_df = pd.DataFrame(
        [
            {
                "Age": float(ev.age),
                "Annual Income (k$)": float(ev.annual_income_k),
                "Spending Score (1-100)": float(ev.spending_score),
            }
            for ev in events
        ]
    )
    report = evaluate_batch_drift(
        baseline_distributions=bundle["baseline_distributions"],
        current_df=current_df,
        features=list(NUMERIC_FEATURE_COLUMNS),
        bins=bins,
        stable_threshold=stable_threshold,
        monitor_threshold=monitor_threshold,
    ).to_dict()

    alert_status = str(report["alert_status"])
    dispatched = dispatch_drift_alert(
        alert_status=alert_status,
        max_psi=float(report["max_psi"]),
        max_ks_statistic=float(report["max_ks_statistic"]),
        window_label=window_label,
        sample_size=len(events),
        details={"features": report["features"]},
    )

    latest_run = get_latest_pipeline_run(session)
    saved = record_drift_report(
        session,
        run_id=latest_run.id if latest_run else None,
        drift_report=report,
        window_label=window_label,
        alert_dispatched=dispatched,
    )
    return {
        "evaluated": True,
        "report_id": saved.id,
        "window_label": window_label,
        "alert_dispatched": dispatched,
        **report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scheduled rolling-window drift evaluation.")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--lookback-days", type=int, default=7)
    args = parser.parse_args()

    cfg = load_config()
    bundle_file = Path(cfg.artifacts.model_bundle_path)
    if not bundle_file.exists():
        raise SystemExit(f"Model bundle not found at {bundle_file}")
    bundle = joblib.load(bundle_file)

    engine = create_db_engine()
    init_db(engine)
    factory = get_session_factory(engine)
    with session_scope(factory) as session:
        result = evaluate_rolling_window_drift(
            session,
            bundle=bundle,
            window_size=args.window_size,
            lookback_days=args.lookback_days,
            stable_threshold=cfg.drift.stable_threshold,
            monitor_threshold=cfg.drift.monitor_threshold,
            bins=cfg.drift.bins,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
