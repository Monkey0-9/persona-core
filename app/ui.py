"""Streamlit Executive & ML Operations UI wired exclusively to the FastAPI backend."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Mall Customer Segmentation Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8010").rstrip("/")
API_KEY = os.getenv("API_KEY", "")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def _api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    resp = requests.get(f"{API_URL}{path}", headers=_headers(), params=params, timeout=10)
    resp.raise_for_status()
    return dict(resp.json())


def _api_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(f"{API_URL}{path}", headers=_headers(), json=payload, timeout=45)
    resp.raise_for_status()
    return dict(resp.json())


st.title("Mall Customer Segmentation & Decisioning Studio")
st.caption(
    "Backend-wired customer intelligence, GMM posterior entropy gating, and PSI drift monitoring"
)

with st.sidebar:
    st.subheader("Backend Connection")
    st.text(f"API: {API_URL}")
    if not API_KEY:
        st.warning("API_KEY is not set in environment. Protected requests will return 401.")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    segments_data = _api_get("/api/v1/segments")
    customers_page = _api_get("/api/v1/customers", params={"limit": 500, "offset": 0})
except Exception as exc:
    st.error(
        f"Unable to connect to FastAPI backend at {API_URL}. "
        f"Ensure uvicorn is running and API_KEY is configured. Details: {exc}"
    )
    st.stop()

if health.get("status") != "ok":
    st.warning(f"API health reported degraded state: {health}")

tab_data, tab_exec, tab_bench, tab_vis, tab_live, tab_batch = st.tabs(
    [
        "Custom / Manual Data Pipeline",
        "Executive KPIs & Playbooks",
        "EDA, Benchmarks & Surrogate Rules",
        "Visualizations",
        "Live Customer Scorer",
        "Batch Scorer, Drift & Audit Logs",
    ]
)

with tab_data:
    st.subheader("Manual Data Entry & Custom CSV Pipeline Execution")
    st.caption(
        "Edit rows interactively, upload a CSV dataset, configure clustering parameters, "
        "and run the full segmentation pipeline on your custom cohort."
    )
    uploaded_custom = st.file_uploader(
        "Import Custom CSV (CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100))",
        type=["csv"],
        key="custom_train_uploader",
    )
    if uploaded_custom is not None:
        base_df = pd.read_csv(io.BytesIO(uploaded_custom.getvalue()))
    else:
        active_items = customers_page.get("items", [])
        if active_items:
            base_df = pd.DataFrame(
                [
                    {
                        "CustomerID": idx + 1,
                        "Gender": row["gender"],
                        "Age": int(row["age"]),
                        "Annual Income (k$)": float(row["annual_income_k"]),
                        "Spending Score (1-100)": float(row["spending_score"]),
                    }
                    for idx, row in enumerate(active_items)
                ]
            )
        else:
            base_df = pd.DataFrame(
                columns=[
                    "CustomerID",
                    "Gender",
                    "Age",
                    "Annual Income (k$)",
                    "Spending Score (1-100)",
                ]
            )

    edited_df = st.data_editor(
        base_df,
        num_rows="dynamic",
        use_container_width=True,
        key="custom_data_editor",
    )

    p1, p2, p3 = st.columns(3)
    custom_k = p1.slider("Target Segments (k)", min_value=2, max_value=10, value=5)
    custom_entropy = p2.slider(
        "Entropy Holdout Threshold", min_value=0.2, max_value=0.8, value=0.45, step=0.05
    )
    custom_gender = p3.checkbox("Include Gender in Distance Geometry", value=False)

    b1, b2 = st.columns(2)
    if b1.button("Run Segmentation Pipeline on Edited Data", type="primary"):
        try:
            records = []
            for idx, row in edited_df.dropna().iterrows():
                cid_raw = row.get("CustomerID", int(idx) + 1)
                records.append(
                    {
                        "customer_id": int(cid_raw),
                        "gender": "Male" if str(row["Gender"]) == "Male" else "Female",
                        "age": int(row["Age"]),
                        "annual_income_k": float(row["Annual Income (k$)"]),
                        "spending_score": float(row["Spending Score (1-100)"]),
                    }
                )
            res = _api_post(
                "/api/v1/train-custom",
                {
                    "customers": records,
                    "target_k": int(custom_k),
                    "include_gender": bool(custom_gender),
                    "entropy_threshold": float(custom_entropy),
                    "enforce_stability_gate": False,
                },
            )
            st.success(
                f"Segmented {res['row_count']} rows into k={res['target_k']} "
                f"(Silhouette={res['silhouette_score']:.3f}, run={res['run_key']})"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Custom training failed: {exc}")

    if b2.button("Reset to Default Dataset"):
        try:
            res = _api_post("/api/v1/reset-default", {})
            st.success(f"Restored default dataset ({res['row_count']} rows).")
            st.rerun()
        except Exception as exc:
            st.error(f"Reset failed: {exc}")

with tab_exec:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dataset Rows", segments_data["row_count"])
    c2.metric("Target Segments (k)", segments_data["target_k"])
    c3.metric("K-Means Silhouette", f"{segments_data['silhouette_score']:.3f}")
    c4.metric("Bootstrap Jaccard", f"{segments_data['mean_jaccard_stability']:.3f}")
    c5.metric("Surrogate Fidelity", f"{segments_data['surrogate_fidelity']:.3f}")

    st.subheader("Actionable Retail Personas & Playbooks")
    personas_df = pd.DataFrame(segments_data["personas"])
    st.dataframe(
        personas_df[
            [
                "cluster_id",
                "persona_name",
                "customer_count",
                "share",
                "mean_age",
                "mean_income",
                "mean_spending",
                "female_share",
                "target_channel",
                "recommended_action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_bench:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Exploratory Summary Statistics")
        st.dataframe(
            pd.DataFrame(segments_data["eda"]["numeric_summaries"]),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Gender Breakdown (Profiling Only)")
        st.json(segments_data["eda"]["gender_breakdown"])
    with col_right:
        st.subheader("Clustering Algorithm Benchmarks")
        st.dataframe(
            pd.DataFrame(segments_data["benchmarks"]),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Bootstrap Stability Gate (B=100)")
        st.json(segments_data["stability"])

    st.subheader("Surrogate Decision Tree IF-THEN Rules")
    st.dataframe(
        pd.DataFrame(segments_data["explainability"]["rules"]),
        use_container_width=True,
        hide_index=True,
    )

with tab_vis:
    customers_df = pd.DataFrame(customers_page["items"])
    if not customers_df.empty:
        st.subheader("Interactive 3D Behavioral Segments")
        fig_3d = px.scatter_3d(
            customers_df,
            x="age",
            y="annual_income_k",
            z="spending_score",
            color="persona_name",
            symbol="is_boundary_customer",
            hover_data=["customer_ref", "gmm_confidence", "normalized_entropy"],
            template="plotly_white",
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.subheader("2D PCA Projection (Sized by GMM Normalized Entropy)")
        fig_pca = px.scatter(
            customers_df,
            x="pca_1",
            y="pca_2",
            color="persona_name",
            size="normalized_entropy",
            symbol="is_boundary_customer",
            hover_data=["customer_ref", "age", "annual_income_k", "spending_score"],
            template="plotly_white",
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    visuals_dir = Path("artifacts/visuals")
    c_img1, c_img2 = st.columns(2)
    if (visuals_dir / "02_kmeans_sweep.png").exists():
        c_img1.image(str(visuals_dir / "02_kmeans_sweep.png"), caption="K-Means & GMM Sweep")
    if (visuals_dir / "03_ward_dendrogram.png").exists():
        c_img2.image(str(visuals_dir / "03_ward_dendrogram.png"), caption="Ward Dendrogram")

with tab_live:
    st.subheader("Real-Time Customer Scoring & Posterior Breakdown")
    with st.form("live_score_form"):
        f1, f2, f3, f4 = st.columns(4)
        gender_in = f1.selectbox("Gender", options=["Female", "Male"])
        age_in = f2.number_input("Age", min_value=16, max_value=100, value=34)
        income_in = f3.number_input(
            "Annual Income (k$)", min_value=5.0, max_value=500.0, value=82.0
        )
        spending_in = f4.number_input(
            "Spending Score (1-100)", min_value=1.0, max_value=100.0, value=78.0
        )
        submitted = st.form_submit_button("Score Customer")

    if submitted:
        try:
            scored = _api_post(
                "/api/v1/score",
                {
                    "gender": gender_in,
                    "age": int(age_in),
                    "annual_income_k": float(income_in),
                    "spending_score": float(spending_in),
                },
            )
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Assigned Persona", scored["persona_name"])
            r2.metric("GMM Confidence", f"{scored['gmm_confidence']:.3f}")
            r3.metric("Normalized Entropy", f"{scored['normalized_entropy']:.3f}")
            boundary_label = "Review Required" if scored["is_boundary_customer"] else "Clear"
            r4.metric("Boundary Holdout", boundary_label)
            st.info(f"Recommended Playbook: {scored['recommended_action']}")

            post_probs = scored.get("posterior_probabilities", {})
            if post_probs:
                prob_df = pd.DataFrame(
                    [{"Persona": k, "Posterior Probability": v} for k, v in post_probs.items()]
                ).sort_values("Posterior Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Persona"))
        except Exception as exc:
            st.error(f"Scoring request failed: {exc}")

    events_page = _api_get("/api/v1/scoring-events", params={"limit": 15, "offset": 0})
    if events_page["items"]:
        st.subheader("Recent Live Scoring Events (Persisted)")
        st.dataframe(pd.DataFrame(events_page["items"]), use_container_width=True, hide_index=True)

with tab_batch:
    st.subheader("Batch Scoring & PSI Drift Monitoring")
    uploaded = st.file_uploader(
        "Upload CSV (columns: gender, age, annual_income_k, spending_score)", type=["csv"]
    )
    if uploaded is not None:
        batch_df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        if st.button("Run Batch Scoring & PSI Check"):
            records = batch_df.rename(
                columns={
                    "Gender": "gender",
                    "Age": "age",
                    "Annual Income (k$)": "annual_income_k",
                    "Spending Score (1-100)": "spending_score",
                }
            )[["gender", "age", "annual_income_k", "spending_score"]].to_dict(orient="records")
            try:
                batch_resp = _api_post("/api/v1/batch-score", {"customers": records})
                st.success(
                    f"Scored {batch_resp['count']} customers | "
                    f"Boundary flags: {batch_resp['boundary_count']} | "
                    f"Overall Drift: {batch_resp['drift_report']['overall_status']} "
                    f"(max PSI={batch_resp['drift_report']['max_psi']:.4f})"
                )
                st.dataframe(
                    pd.DataFrame(batch_resp["drift_report"]["features"]),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as exc:
                st.error(f"Batch scoring failed: {exc}")

    drift_page = _api_get("/api/v1/drift-reports", params={"limit": 10, "offset": 0})
    if drift_page["items"]:
        st.subheader("Historical Drift Audit Reports")
        st.dataframe(
            pd.DataFrame(drift_page["items"]).drop(columns=["features"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Persisted Customer Explorer")
    f_col1, f_col2 = st.columns(2)
    seg_options = ["All", *[p["persona_name"] for p in segments_data["personas"]]]
    selected_seg = f_col1.selectbox("Filter by Segment", options=seg_options)
    boundary_only = f_col2.checkbox("Show Boundary Customers Only", value=False)
    query_params: dict[str, Any] = {"limit": 100, "offset": 0, "boundary_only": boundary_only}
    if selected_seg != "All":
        query_params["segment"] = selected_seg
    filtered_page = _api_get("/api/v1/customers", params=query_params)
    st.caption(
        f"Showing {len(filtered_page['items'])} of {filtered_page['total']} matching records"
    )
    st.dataframe(pd.DataFrame(filtered_page["items"]), use_container_width=True, hide_index=True)
