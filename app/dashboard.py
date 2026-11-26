"""Interactive executive Streamlit explorer for Mall Customer Segmentation."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from mall_seg.clustering import compute_k_distance_curve, run_clustering_suite
from mall_seg.config import load_config
from mall_seg.evaluation import evaluate_kmeans_stability
from mall_seg.preprocessing import prepare_and_project_features
from mall_seg.profiling import (
    compute_eda_report,
    generate_segment_kpis,
    map_clusters_to_personas,
    score_customer_records,
)
from mall_seg.schema import load_raw_dataset, strip_customer_id, validate_raw_dataframe
from mall_seg.visualization import (
    build_2d_pca_figure,
    build_3d_segment_figure,
    build_k_distance_figure,
)

st.set_page_config(
    page_title="Mall Segmentation Engine — Executive Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _load_initial_state() -> tuple[pd.DataFrame, str, dict[str, Any] | None]:
    cfg = load_config()
    raw_df, source_kind, _ = load_raw_dataset(cfg)
    bundle_path = Path(cfg.artifacts.model_bundle_path)
    bundle: dict[str, Any] | None = None
    if bundle_path.exists():
        loaded = joblib.load(bundle_path)
        if isinstance(loaded, dict):
            bundle = loaded
    return raw_df, source_kind, bundle


st.title("Mall Customer Segmentation Engine")
st.caption(
    "Privacy-first clustering, k-NN DBSCAN knee tuning, GMM entropy gating & actionable KPIs"
)

cfg = load_config()
raw_df, source_kind, initial_bundle = _load_initial_state()

with st.sidebar:
    st.subheader("Pipeline Hyperparameters")
    target_k = st.slider(
        "Target Segments (k)", min_value=2, max_value=10, value=cfg.clustering.target_k
    )
    include_gender = st.checkbox(
        "Include Gender in Clustering",
        value=cfg.privacy.include_gender,
    )
    entropy_threshold = st.slider(
        "GMM Boundary Entropy Gate",
        min_value=0.10,
        max_value=0.90,
        value=float(cfg.clustering.entropy_threshold),
        step=0.05,
    )
    dbscan_min_samples = st.slider(
        "DBSCAN min_samples",
        min_value=3,
        max_value=15,
        value=cfg.clustering.dbscan_min_samples,
    )
    uploaded_csv = st.file_uploader("Upload Custom CSV", type=["csv"])

if uploaded_csv is not None:
    try:
        candidate = pd.read_csv(io.BytesIO(uploaded_csv.getvalue()))
        raw_df = validate_raw_dataframe(candidate, min_rows=15)
        source_kind = "uploaded"
    except Exception as exc:
        st.error(f"Dataset contract error: {exc}")
        st.stop()

df_no_id = strip_customer_id(raw_df)
feats = prepare_and_project_features(
    raw_df,
    include_gender=include_gender,
    pca_components=cfg.features.pca_components,
    seed=cfg.clustering.seed,
)
suite = run_clustering_suite(
    feats.X_scaled,
    target_k=target_k,
    k_min=cfg.clustering.k_min,
    k_max=min(cfg.clustering.k_max, len(raw_df) - 1),
    kmeans_n_init=cfg.clustering.kmeans_n_init,
    dbscan_min_samples=dbscan_min_samples,
    gmm_covariance_type=cfg.clustering.gmm_covariance_type,
    seed=cfg.clustering.seed,
)
stability = evaluate_kmeans_stability(
    feats.X_scaled,
    suite.kmeans_labels,
    n_clusters=target_k,
    n_bootstrap=30,
    bootstrap_n_init=10,
    min_jaccard_stability=cfg.stability.min_jaccard_stability,
    seed=cfg.clustering.seed,
)
personas = map_clusters_to_personas(raw_df, suite.kmeans_labels)
eda_report = compute_eda_report(raw_df)

posteriors = suite.gmm_posteriors
entropies = suite.gmm_entropy
confidences = posteriors.max(axis=1)
is_boundary = entropies > entropy_threshold
kpis = generate_segment_kpis(raw_df, suite.kmeans_labels, is_boundary=is_boundary)

scored_df = pd.DataFrame(
    [
        {
            "customer_ref": f"CUST-{int(raw_df.iloc[i]['CustomerID']):04d}",
            "gender": str(raw_df.iloc[i]["Gender"]),
            "age": int(raw_df.iloc[i]["Age"]),
            "annual_income_k": float(raw_df.iloc[i]["Annual Income (k$)"]),
            "spending_score": float(raw_df.iloc[i]["Spending Score (1-100)"]),
            "cluster_id": int(suite.kmeans_labels[i]),
            "persona_name": personas[int(suite.kmeans_labels[i])].persona_name,
            "recommended_action": personas[int(suite.kmeans_labels[i])].recommended_action,
            "gmm_confidence": round(float(confidences[i]), 4),
            "normalized_entropy": round(float(entropies[i]), 4),
            "is_boundary_customer": bool(is_boundary[i]),
            "pca_1": round(float(feats.X_pca[i, 0]), 4),
            "pca_2": round(float(feats.X_pca[i, 1]), 4),
        }
        for i in range(len(raw_df))
    ]
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Customers", len(raw_df), help=f"Source: {source_kind}")
m2.metric("Target Segments", target_k)
m3.metric("K-Means Silhouette", f"{suite.benchmarks[0].silhouette or 0.0:.3f}")
m4.metric("Bootstrap Jaccard", f"{stability.mean_jaccard:.3f}")
m5.metric("Boundary Holdouts", int(is_boundary.sum()))

tab_overview, tab_geometry, tab_eda, tab_scorer = st.tabs(
    [
        "Persona KPIs & Playbooks",
        "2D / 3D Clusters & k-Dist Knee",
        "EDA & Algorithm Benchmarks",
        "Live Decision Scorer",
    ]
)

with tab_overview:
    st.subheader("Executive Segment KPI Breakdown")
    st.dataframe(
        pd.DataFrame([k.to_dict() for k in kpis]),
        use_container_width=True,
        hide_index=True,
    )

with tab_geometry:
    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(
            build_2d_pca_figure(scored_df, feats.explained_variance_ratio),
            use_container_width=True,
        )
    with c_right:
        st.plotly_chart(
            build_k_distance_figure(feats.X_scaled, min_samples=dbscan_min_samples),
            use_container_width=True,
        )
    st.plotly_chart(
        build_3d_segment_figure(scored_df, template="plotly_white"),
        use_container_width=True,
    )

with tab_eda:
    st.subheader("Algorithm Validity Benchmarks (Silhouette, CH, Davies-Bouldin)")
    st.dataframe(
        pd.DataFrame([b.to_dict() for b in suite.benchmarks]),
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Exploratory Numeric Summary Statistics")
    st.dataframe(
        pd.DataFrame([s.to_dict() for s in eda_report.numeric_summaries]),
        use_container_width=True,
        hide_index=True,
    )
    curve = compute_k_distance_curve(feats.X_scaled, min_samples=dbscan_min_samples)
    st.info(
        f"DBSCAN Auto-Knee Estimate: eps={curve.recommended_eps:.4f} "
        f"at sorted index {curve.knee_index} (min_samples={curve.min_samples})"
    )

with tab_scorer:
    st.subheader("Score New Customer Profile in Real Time")
    with st.form("dashboard_live_scorer"):
        f1, f2, f3, f4 = st.columns(4)
        g_in = f1.selectbox("Gender", options=["Female", "Male"])
        age_in = f2.number_input("Age", min_value=16, max_value=100, value=32)
        inc_in = f3.number_input("Annual Income (k$)", min_value=5.0, max_value=500.0, value=86.0)
        sp_in = f4.number_input(
            "Spending Score (1-100)", min_value=1.0, max_value=100.0, value=82.0
        )
        submitted = st.form_submit_button("Assign Segment & Uncertainty")

    if submitted:
        bundle_runtime = {
            "preprocessor": feats.preprocessor,
            "pca": feats.pca,
            "kmeans_model": suite.kmeans_model,
            "gmm_model": suite.gmm_model,
            "personas": personas,
            "entropy_threshold": entropy_threshold,
            "gmm_cluster_map": None,
        }
        # Re-use direct bundle alignment if loaded with matching configuration
        if initial_bundle and not include_gender and target_k == cfg.clustering.target_k:
            bundle_runtime = initial_bundle
        pred = score_customer_records(
            [
                {
                    "gender": g_in,
                    "age": int(age_in),
                    "annual_income_k": float(inc_in),
                    "spending_score": float(sp_in),
                }
            ],
            bundle_runtime,
        )[0]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Persona", pred.persona_name)
        s2.metric("Confidence", f"{pred.gmm_confidence:.3f}")
        s3.metric("Normalized Entropy", f"{pred.normalized_entropy:.3f}")
        s4.metric("Holdout Flag", "Review" if pred.is_boundary_customer else "Clear")
        st.success(f"Actionable Playbook: {pred.recommended_action}")
