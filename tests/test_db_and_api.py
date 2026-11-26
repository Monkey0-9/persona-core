"""Tests for SQLAlchemy persistence and FastAPI authenticated endpoints."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from mall_seg.algorithms import compute_normalized_entropy, run_clustering_suite
from mall_seg.api import create_app
from mall_seg.contracts import (
    NUMERIC_FEATURE_COLUMNS,
    AppConfig,
    generate_deterministic_fixture,
    strip_customer_id,
)
from mall_seg.db import (
    DriftReportRecord,
    PipelineRun,
    ScoredCustomer,
    ScoringEvent,
    create_db_engine,
    get_session_factory,
    init_db,
    session_scope,
    upsert_pipeline_run,
)
from mall_seg.eda import compute_eda_report
from mall_seg.explainability import fit_surrogate_explainer
from mall_seg.features import fit_transform_features
from mall_seg.personas import compute_feature_schema_payload, map_clusters_to_personas
from mall_seg.stability import evaluate_kmeans_stability


@pytest.fixture()
def seeded_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, str, str]:
    db_path = tmp_path / "test_mall.db"
    db_url = f"sqlite:///{db_path}"
    bundle_path = tmp_path / "model_bundle.joblib"
    api_key = "test-secret-api-key"
    monkeypatch.setenv("API_KEY", api_key)
    monkeypatch.setenv("DATABASE_URL", db_url)

    fixture = generate_deterministic_fixture(tmp_path / "sample.csv", n_rows=120, seed=42)
    raw_df = pd.read_csv(fixture)
    df_no_id = strip_customer_id(raw_df)
    eda = compute_eda_report(raw_df)
    feats = fit_transform_features(df_no_id, include_gender=False, pca_components=2, seed=42)
    suite = run_clustering_suite(
        feats.X_scaled, target_k=5, k_min=2, k_max=7, kmeans_n_init=20, seed=42
    )
    stability = evaluate_kmeans_stability(
        feats.X_scaled,
        suite.kmeans_labels,
        n_clusters=5,
        n_bootstrap=20,
        bootstrap_n_init=10,
        min_jaccard_stability=0.60,
        seed=42,
    )
    personas = map_clusters_to_personas(raw_df, suite.kmeans_labels)
    expl = fit_surrogate_explainer(
        df_no_id, suite.kmeans_labels, {k: v.persona_name for k, v in personas.items()}
    )

    posteriors = suite.gmm_model.predict_proba(feats.X_scaled)
    entropies = compute_normalized_entropy(posteriors)
    confidences = posteriors.max(axis=1)

    scored_rows = []
    for idx in range(len(raw_df)):
        cid = int(suite.kmeans_labels[idx])
        prof = personas[cid]
        ent = round(float(entropies[idx]), 4)
        scored_rows.append(
            {
                "customer_ref": f"CUST-{int(raw_df.iloc[idx]['CustomerID']):04d}",
                "gender": str(raw_df.iloc[idx]["Gender"]),
                "age": int(raw_df.iloc[idx]["Age"]),
                "annual_income_k": float(raw_df.iloc[idx]["Annual Income (k$)"]),
                "spending_score": float(raw_df.iloc[idx]["Spending Score (1-100)"]),
                "cluster_id": cid,
                "persona_name": prof.persona_name,
                "recommended_action": prof.recommended_action,
                "gmm_confidence": round(float(confidences[idx]), 4),
                "normalized_entropy": ent,
                "is_boundary_customer": ent > 0.45,
                "pca_1": round(float(feats.X_pca[idx, 0]), 4),
                "pca_2": round(float(feats.X_pca[idx, 1]), 4),
            }
        )

    engine = create_db_engine(db_url)
    init_db(engine)
    factory = get_session_factory(engine)
    summary_payload = {
        "eda": eda.to_dict(),
        "benchmarks": [b.to_dict() for b in suite.benchmarks],
        "stability": stability.to_dict(),
        "personas": [p.to_dict() for p in personas.values()],
        "explainability": expl.to_dict(),
    }

    # Test idempotent double-upsert with same run_key
    for _ in range(2):
        with session_scope(factory) as session:
            upsert_pipeline_run(
                session,
                run_key="test-run-001",
                dataset_source="fixture",
                row_count=len(raw_df),
                target_k=5,
                include_gender=False,
                silhouette_score=float(suite.benchmarks[0].silhouette or 0.0),
                mean_jaccard_stability=stability.mean_jaccard,
                passed_stability_gate=stability.passed_gate,
                surrogate_fidelity=expl.fidelity_score,
                summary_payload=summary_payload,
                scored_rows=scored_rows,
            )

    with session_scope(factory) as session:
        assert session.scalar(select(func.count()).select_from(PipelineRun)) == 1
        assert session.scalar(select(func.count()).select_from(ScoredCustomer)) == 120

    cfg = AppConfig()
    schema_payload = compute_feature_schema_payload(cfg, model_version="v1.0.0")
    bundle = {
        "run_key": "test-run-001",
        "model_version": "v1.0.0",
        "feature_schema_hash": schema_payload["feature_schema_hash"],
        "config": cfg.model_dump(),
        "feature_summary": {"feature_names": schema_payload["feature_names"]},
        "preprocessor": feats.preprocessor,
        "pca": feats.pca,
        "kmeans_model": suite.kmeans_model,
        "gmm_model": suite.gmm_model,
        "personas": personas,
        "entropy_threshold": 0.45,
        "baseline_distributions": {
            col: raw_df[col].astype(float).tolist() for col in NUMERIC_FEATURE_COLUMNS
        },
    }
    joblib.dump(bundle, bundle_path)

    app = create_app(database_url=db_url, bundle_path=bundle_path)
    return TestClient(app), api_key, db_url


def test_health_and_auth_enforcement(seeded_env: tuple[TestClient, str, str]) -> None:
    client, api_key, _ = seeded_env

    health = client.get("/health")
    assert health.status_code == 200
    assert health.headers["X-Content-Type-Options"] == "nosniff"
    assert health.headers["X-Frame-Options"] == "DENY"
    body = health.json()
    assert body["status"] == "ok"
    assert body["database_ok"] is True
    assert body["model_loaded"] is True
    assert body["active_run_key"] == "test-run-001"

    # Missing or invalid API key -> 401
    assert client.get("/api/v1/segments").status_code == 401
    assert client.get("/api/v1/segments", headers={"X-API-Key": "wrong"}).status_code == 401
    assert (
        client.post(
            "/api/v1/score",
            json={"gender": "Female", "age": 30, "annual_income_k": 80, "spending_score": 82},
        ).status_code
        == 401
    )

    seg_ok = client.get("/api/v1/segments", headers={"X-API-Key": api_key})
    assert seg_ok.status_code == 200
    assert seg_ok.json()["run_key"] == "test-run-001"


def test_score_batch_and_customers_endpoints(seeded_env: tuple[TestClient, str, str]) -> None:
    client, api_key, db_url = seeded_env
    headers = {"X-API-Key": api_key}

    # Invalid payload -> 422
    bad_resp = client.post(
        "/api/v1/score",
        headers=headers,
        json={"gender": "Female", "age": 10, "annual_income_k": 80, "spending_score": 82},
    )
    assert bad_resp.status_code == 422

    # Valid single score
    single_resp = client.post(
        "/api/v1/score",
        headers=headers,
        json={
            "customer_ref": "LIVE-01",
            "gender": "Female",
            "age": 31,
            "annual_income_k": 90.0,
            "spending_score": 85.0,
        },
    )
    assert single_resp.status_code == 200
    assert "X-Request-ID" in single_resp.headers
    assert "X-Process-Time-Ms" in single_resp.headers
    s_data = single_resp.json()
    assert 0.0 <= s_data["normalized_entropy"] <= 1.0
    assert 0.0 <= s_data["gmm_confidence"] <= 1.0
    assert isinstance(s_data["is_boundary_customer"], bool)
    assert pytest.approx(sum(s_data["posterior_probabilities"].values()), abs=1e-2) == 1.0

    # Counterfactual transition simulator
    cf_resp = client.post(
        "/api/v1/customers/counterfactual",
        headers=headers,
        json={
            "customer_ref": "CF-01",
            "gender": "Female",
            "age": 46,
            "annual_income_k": 24.0,
            "spending_score": 18.0,
            "target_persona_key": "affluent_vip",
        },
    )
    assert cf_resp.status_code == 200
    cf_data = cf_resp.json()
    assert cf_data["target_persona_key"] == "affluent_vip"
    assert "action_summary" in cf_data

    # Valid batch score
    batch_payload = {
        "customers": [
            {"gender": "Male", "age": 24, "annual_income_k": 25.0, "spending_score": 80.0},
            {"gender": "Female", "age": 42, "annual_income_k": 92.0, "spending_score": 16.0},
            {"gender": "Female", "age": 36, "annual_income_k": 69.0, "spending_score": 60.0},
        ]
    }
    batch_resp = client.post("/api/v1/batch-score", headers=headers, json=batch_payload)
    assert batch_resp.status_code == 200
    b_data = batch_resp.json()
    assert b_data["count"] == 3
    assert "max_psi" in b_data["drift_report"]

    # Paginated customers, drift-reports, scoring-events, and visuals guardrails
    cust_resp = client.get("/api/v1/customers?limit=10&offset=0", headers=headers)
    assert cust_resp.status_code == 200
    assert cust_resp.json()["total"] == 123
    assert len(cust_resp.json()["items"]) == 10

    drift_hist = client.get("/api/v1/drift-reports?limit=10&offset=0", headers=headers)
    assert drift_hist.status_code == 200
    assert drift_hist.json()["total"] == 1

    events_hist = client.get("/api/v1/scoring-events?limit=10&offset=0", headers=headers)
    assert events_hist.status_code == 200
    assert events_hist.json()["total"] == 4

    bad_vis = client.get("/api/v1/visuals/../../etc/passwd", headers=headers)
    assert bad_vis.status_code == 404

    # Verify DB persistence of scoring_events and drift_reports
    engine = create_db_engine(db_url)
    factory = get_session_factory(engine)
    with session_scope(factory) as session:
        assert session.scalar(select(func.count()).select_from(ScoringEvent)) == 4
        assert session.scalar(select(func.count()).select_from(DriftReportRecord)) == 1


def test_custom_dataset_train_and_reset(seeded_env: tuple[TestClient, str, str]) -> None:
    client, api_key, _ = seeded_env
    headers = {"X-API-Key": api_key}

    # Unauthenticated request must fail with 401
    assert client.post("/api/v1/train-custom", json={"customers": []}).status_code == 401

    # Construct a 24-row manual dataset across 3 behavioral clusters
    manual_rows = []
    for i in range(8):
        manual_rows.append(
            {
                "customer_id": i + 1,
                "gender": "Female" if i % 2 == 0 else "Male",
                "age": 24 + (i % 4),
                "annual_income_k": 24.0 + i,
                "spending_score": 80.0 + (i % 6),
            }
        )
    for i in range(8):
        manual_rows.append(
            {
                "customer_id": i + 9,
                "gender": "Male" if i % 2 == 0 else "Female",
                "age": 32 + (i % 5),
                "annual_income_k": 90.0 + i,
                "spending_score": 84.0 + (i % 5),
            }
        )
    for i in range(8):
        manual_rows.append(
            {
                "customer_id": i + 17,
                "gender": "Female" if i % 2 == 0 else "Male",
                "age": 46 + (i % 4),
                "annual_income_k": 55.0 + i,
                "spending_score": 20.0 + (i % 7),
            }
        )

    # Duplicate CustomerID must be rejected cleanly with 422
    dup_rows = [dict(r) for r in manual_rows]
    dup_rows[1]["customer_id"] = dup_rows[0]["customer_id"]
    dup_resp = client.post(
        "/api/v1/train-custom",
        headers=headers,
        json={"customers": dup_rows, "target_k": 3, "include_gender": False},
    )
    assert dup_resp.status_code == 422

    # Valid custom dataset training updates active segments & scopes /api/v1/customers
    train_resp = client.post(
        "/api/v1/train-custom",
        headers=headers,
        json={
            "customers": manual_rows,
            "target_k": 3,
            "include_gender": True,
            "entropy_threshold": 0.4,
            "enforce_stability_gate": False,
        },
    )
    assert train_resp.status_code == 200
    t_body = train_resp.json()
    assert t_body["dataset_source"] == "custom"
    assert t_body["row_count"] == 24
    assert t_body["target_k"] == 3
    assert t_body["include_gender"] is True
    assert len(t_body["personas"]) == 3

    cust_after = client.get("/api/v1/customers?limit=100&offset=0", headers=headers)
    assert cust_after.status_code == 200
    assert cust_after.json()["total"] == 24

    # Live scoring works against newly trained custom model
    live_resp = client.post(
        "/api/v1/score",
        headers=headers,
        json={"gender": "Female", "age": 33, "annual_income_k": 92.0, "spending_score": 86.0},
    )
    assert live_resp.status_code == 200
    assert live_resp.json()["persona_name"] in {p["persona_name"] for p in t_body["personas"]}

    # Reset restores default dataset as active run
    reset_resp = client.post("/api/v1/reset-default", headers=headers)
    assert reset_resp.status_code == 200
    r_body = reset_resp.json()
    assert r_body["dataset_source"] in {"primary", "fixture"}
    assert r_body["row_count"] >= 50
    assert client.get("/api/v1/segments", headers=headers).json()["run_key"] == r_body["run_key"]
