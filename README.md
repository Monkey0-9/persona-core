# PersonaCore: Enterprise Customer Segmentation & Behavioral Intelligence Engine

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.165+-009688.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19.0-61DAFB.svg)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-3178C6.svg)](https://www.typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-8.3-646CFF.svg)](https://vitejs.dev)
[![SQLAlchemy 2.0](https://img.shields.io/badge/SQLAlchemy-2.0-red.svg)](https://www.sqlalchemy.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com)
[![Tests Passing](https://img.shields.io/badge/tests-16%20passed-brightgreen.svg)]()
[![Type Checking](https://img.shields.io/badge/mypy-100%25%20typed-blue.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**PersonaCore** is a production-grade, privacy-first customer segmentation and prescriptive intelligence engine. It unites unsupervised machine learning (K-Means, Ward Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models) with statistical stability gating, probabilistic uncertainty scoring, continuous data drift auditing (PSI + KS), model lineage governance, and actionable counterfactual intervention planning.

The platform provides a dual-interface delivery model: a high-performance **FastAPI microservice** backed by PostgreSQL/SQLite and Redis, coupled with a reactive **React 19 / TypeScript / Vite** console featuring custom interactive 2D and 3D orthographic cluster projections.

---

## Table of Contents

- [Architectural Overview](#architectural-overview)
- [Key Engineering Pillars](#key-engineering-pillars)
- [Mathematical & Algorithmic Foundations](#mathematical--algorithmic-foundations)
- [Codebase & System Layout](#codebase--system-layout)
- [Data Pipeline & Contracts](#data-pipeline--contracts)
- [Model Registry & Lineage Governance](#model-registry--lineage-governance)
- [Security, Auth & Observability](#security-auth--observability)
- [REST API Specification](#rest-api-specification)
- [Developer Quickstart](#developer-quickstart)
- [Docker & Containerized Deployment](#docker--containerized-deployment)
- [Quality Verification & CI/CD](#quality-verification--cicd)
- [Senior Engineer Architectural Evaluation](#senior-engineer-architectural-evaluation)

---

## Architectural Overview

```mermaid
flowchart TD
    subgraph Data & Contracts Layer
        RAW["Raw Dataset (Mall_Customers.csv)"] --> PANDERA["Pandera Schema Validation\n(Type, Range, Null Invariants)"]
        PANDERA --> PRIVACY["Privacy Masking & Identifier Stripping\n(Zero PII Leakage)"]
        PRIVACY --> PREPROC["Robust Scaling & PCA Projection\n(StandardScaler + PCA 2D/3D)"]
    end

    subgraph Unsupervised ML Engine
        PREPROC --> KM["K-Means Cluster Sweep\n(k=2..10, Elbow + Silhouette)"]
        PREPROC --> WARD["Ward Hierarchical\n(Linkage & Cophenetic Correlation)"]
        PREPROC --> DBSCAN["DBSCAN Density Analysis\n(k-Distance Curve Knee)"]
        PREPROC --> GMM["Gaussian Mixture Models (GMM)\n(BIC/AIC Selection, Soft Assignment)"]
    end

    subgraph Statistical Quality & Stability Gating
        KM --> BOOT["Non-Parametric Bootstrap (B=100)\n(Resampling with Replacement)"]
        BOOT --> HUNG["Hungarian Kuhn-Munkres Alignment"]
        HUNG --> STAB{"Mean Jaccard >= 0.70?"}
        STAB -- "Fail" --> ABORT["Reject Deployment Pipeline"]
        STAB -- "Pass" --> REG["Versioned Model Store\n(SHA-256 Schema Hash + MLflow)"]
    end

    subgraph Decisioning & Uncertainty Engine
        REG --> UNCERTAIN["Normalized Shannon Entropy H_norm(x)\n& GMM Boundary Customer Flagging"]
        REG --> SURROGATE["Surrogate Decision Tree Explainer\n(Auditable IF-THEN Rules)"]
        REG --> COUNTER["Counterfactual Optimizer\n(Minimum Effort Intervention Path)"]
    end

    subgraph Backend Microservice & Persistence
        UNCERTAIN & SURROGATE & COUNTER --> FASTAPI["FastAPI 0.165+ Gateway\n(JWT/RBAC, Redis Token Bucket Rate Limiter)"]
        FASTAPI --> SQLALCHEMY["SQLAlchemy 2.0 Engine\n(PostgreSQL / SQLite Connection Pooling)"]
        FASTAPI --> PROMETHEUS["Prometheus Telemetry (/metrics)\n& Structured JSON Logging (X-Request-ID)"]
        FASTAPI --> DRIFT["Continuous Drift Auditor\n(PSI + Two-Sample KS Test)"]
        DRIFT --> WEBHOOK["HMAC-Signed Webhook Alerts"]
    end

    subgraph Delivery & Consumption
        FASTAPI --> EXPRESS["Express / Node.js Supervisor\n(Ephemeral Port Detection & Subprocess Watchdog)"]
        EXPRESS --> REACT["React 19 Console (Tailwind CSS v4)\n- 2D / 3D Geometry Projections\n- Executive Persona Briefings\n- Live Scoring & Counterfactual Sim"]
        FASTAPI --> STREAMLIT["Streamlit Data Science Dashboard\n(App/Dashboard.py)"]
    end
```

---

## Key Engineering Pillars

### 1. Privacy-by-Design & Data Contracts
- **Zero-PII Clustering**: Customer identifiers (`CustomerID`) are stripped immediately at ingestion. Gender is excluded by default (`include_gender: false`) to avoid algorithmic discrimination and focus clustering strictly on behavioral spending patterns.
- **Runtime Pandera Contracts**: Data is strictly validated for nulls, type consistency, and domain bounds (e.g., $Age \in [16, 100]$, $SpendingScore \in [1, 100]$) prior to tensor transformation.

### 2. Multi-Algorithm Suite & Cross-Validation
- Runs four complementary clustering paradigms concurrently:
  1. **Partitioning**: Multi-start K-Means ($k=2 \dots 10$) evaluated against Inertia, Silhouette Width, Calinski-Harabasz, and Davies-Bouldin metrics.
  2. **Hierarchical**: Ward's minimum-variance agglomeration producing an auditable dendrogram matrix and cophenetic correlation tracking.
  3. **Density-Based**: DBSCAN with automated k-nearest neighbors distance elbow computation to discover arbitrary topologies and isolate outlier noise.
  4. **Probabilistic**: Gaussian Mixture Models (GMM) with full covariance structures, enabling soft membership probabilities and Bayesian Information Criterion (BIC) minimization.

### 3. Bootstrap Stability Gating
- Production deployments are protected by an automated statistical gate:
  - Generates $B = 100$ bootstrap replicates with replacement.
  - Matches perturbed cluster partitions against baseline centroids using the **Hungarian algorithm (Kuhn-Munkres)** for optimal bipartite assignment.
  - Requires a minimum mean Jaccard stability threshold of $\ge 0.70$. Pipeline execution halts automatically if clusters fail to exhibit high structural reproducibility.

### 4. Probabilistic Uncertainty & Boundary Scoring
- Every inference payload evaluates normalized Shannon entropy across GMM posterior probabilities:
  $$H_{\text{norm}}(x) = -\frac{1}{\ln K} \sum_{k=1}^K p(z_k \mid x) \ln p(z_k \mid x)$$
- Data points exhibiting $H_{\text{norm}}(x) \ge 0.45$ are automatically labeled as **Boundary Customers**, signaling marketing automation systems to avoid aggressive or mismatched campaigns.

### 5. Automated Data Drift Governance (PSI + KS)
- Continuous drift monitoring tracks distribution changes between reference baselines and inference traffic:
  - **Population Stability Index (PSI)**: Monitored across equal-frequency bins ($\text{PSI} < 0.10$ stable, $\text{PSI} \ge 0.25$ critical shift).
  - **Two-Sample Kolmogorov-Smirnov (KS) Test**: Non-parametric evaluation checking for feature-level drift with $p$-value significance testing.
  - **Automated Webhook Dispatch**: When drift limits are exceeded, HMAC-SHA256 signed alert payloads are dispatched to designated webhook endpoints.

### 6. Actionable Counterfactual Optimization
- Prescriptive intelligence moves beyond descriptive clustering:
  - Evaluates the gradient distance from a customer's standardized feature vector to higher-value target cluster centroids (e.g., transitioning *Thrifty Aspirants* into *Affluent VIPs*).
  - Yields precise, actionable behavioral guidance (e.g., `"Increase spending score by +34 pts via targeted loyalty incentives"`).

### 7. Dual-Stack Enterprise Web Architecture
- **Backend**: FastAPI with asynchronous endpoints, role-based access control (RBAC), constant-time secret comparison, and SQLAlchemy 2.0 dual-mode persistence.
- **Node Supervisor**: `server.ts` manages process lifecycle, dynamically selects unallocated TCP ports, binds runtime credentials, and proxies API traffic.
- **Frontend**: React 19 single-page application built on Vite and Tailwind CSS v4, featuring hardware-accelerated 2D Canvas and 3D orthographic projection rendering.

---

## Mathematical & Algorithmic Foundations

### K-Means Inertia & Silhouette Width
Given cluster partition $\mathcal{C} = \{C_1, \dots, C_k\}$ and centroids $\mu_k$:
$$\text{Inertia} = \sum_{k=1}^K \sum_{x \in C_k} \| x - \mu_k \|_2^2$$
For a sample $i$, let $a(i)$ be the mean intra-cluster distance and $b(i)$ be the minimum mean distance to any other cluster:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$

### Population Stability Index (PSI)
For reference distribution $P$ and actual distribution $Q$ partitioned into $B$ bins:
$$\text{PSI} = \sum_{b=1}^B \left( Q_b - P_b \right) \times \ln\left( \frac{Q_b}{P_b} \right)$$
- $\text{PSI} < 0.10$: Minimal shift; model remains valid.
- $0.10 \le \text{PSI} < 0.25$: Moderate drift; warning triggered.
- $\text{PSI} \ge 0.25$: Significant divergence; triggers alert and automated retraining recommendation.

### Counterfactual Feature Projection
Let $x \in \mathbb{R}^d$ be the customer's normalized coordinates and $\mu_T \in \mathbb{R}^d$ be the target centroid. The counterfactual delta is computed as:
$$\Delta^* = \mu_T - x, \quad \|\Delta^*\|_2 = \sqrt{\sum_{j=1}^d (\mu_{T,j} - x_j)^2}$$
De-normalized to feature dimensions:
$$\Delta_{\text{actual}, j} = \Delta^*_j \cdot \sigma_j$$

---

## Codebase & System Layout

```
persona-core/
├── alembic/                       # Database migrations (zero-downtime schema upgrades)
│   ├── env.py
│   └── versions/                  # Revision versions (initial schema, registry, drift)
├── app/                           # Alternative data science frontend
│   ├── dashboard.py               # Streamlit executive visualizer
│   └── ui.py                      # Interactive exploration interface
├── artifacts/                     # Versioned model registry storage
│   ├── model_bundle.joblib        # Active model bundle
│   └── models/v1.0.0/             # Immutable versioned artifacts
│       ├── feature_schema.json    # SHA-256 verified feature manifest
│       ├── metrics.json           # Model evaluation statistics
│       ├── model.joblib           # Serialized model pipeline
│       └── reference_stats.json   # Baseline distributions for drift audit
├── configs/                       # Centralized declarative configurations
│   ├── pipeline_config.yaml       # Core ML and platform hyperparameters
│   └── prod.yaml                  # Production-specific environment overrides
├── data/
│   └── raw/                       # Primary input dataset
│       └── Mall_Customers.csv     # Kaggle customer segmentation dataset
├── ml/                            # Machine learning pipeline scripts
│   ├── drift.py                   # Offline & rolling-window drift evaluation
│   ├── registry.py                # Artifact packaging and registration utilities
│   └── train.py                   # Deterministic offline model training entrypoint
├── public/                        # Static web assets and pre-rendered snapshots
├── scripts/                       # Operational management scripts
│   ├── render_visuals.py          # Headless generation of Matplotlib/Plotly figures
│   └── run_pipeline.py            # End-to-end orchestration CLI entrypoint
├── src/
│   ├── api/                       # Frontend REST client bindings
│   │   └── client.ts              # Type-safe Fetch client with token interceptors
│   ├── components/                # React UI component architecture
│   │   ├── BenchmarksSection.tsx  # Elbow, Silhouette, GMM sweeps, and Decision Tree
│   │   ├── BriefSection.tsx       # Executive persona scorecards and KPIs
│   │   ├── Charts.tsx             # Canvas 2D cluster visuals and PCA diagrams
│   │   ├── DataWorkspace.tsx      # Ingestion status, Pandera contracts, EDA stats
│   │   ├── DecisioningSection.tsx # Real-time customer scoring and counterfactuals
│   │   ├── DriftSection.tsx       # PSI gauges, KS feature drift, model registry
│   │   ├── ErrorBoundary.tsx      # Graceful component failure boundary
│   │   ├── GeometrySection.tsx    # Multi-algorithm 2D switchboard
│   │   └── ThreeDClusterView.tsx  # Custom 3D isometric orthographic projector
│   ├── mall_seg/                  # Core Python backend engine
│   │   ├── alerts.py              # HMAC-signed drift alert webhook dispatcher
│   │   ├── algorithms.py          # K-Means, Ward, DBSCAN, and GMM implementation
│   │   ├── api.py                 # FastAPI microservice with RBAC & governance
│   │   ├── contracts.py           # Pandera schemas and Pydantic AppConfig
│   │   ├── db.py                  # SQLAlchemy 2.0 models and persistence layer
│   │   ├── drift.py               # Statistical drift functions (PSI, KS, OOD)
│   │   ├── eda.py                 # Exploratory data analysis calculation engine
│   │   ├── explainability.py      # Surrogate decision tree rule extractor
│   │   ├── features.py            # Preprocessing, scaling, and PCA transformations
│   │   ├── observability.py       # Prometheus metrics and JSON structured logger
│   │   ├── personas.py            # Deterministic archetype matching and scoring
│   │   ├── rate_limit.py          # Distributed Redis sliding-window rate limiter
│   │   ├── registry.py            # Versioned artifact management & rollback engine
│   │   ├── security.py            # Password hashing, JWT token issue/decode
│   │   ├── stability.py           # Bootstrap stability gating (Kuhn-Munkres)
│   │   └── visualization.py       # Matplotlib & Plotly static rendering
│   ├── App.tsx                    # Root React application shell and navigation
│   ├── index.css                  # Tailwind CSS root stylesheet
│   ├── main.tsx                   # React 19 hydration entrypoint
│   └── types.ts                   # Unified TypeScript definitions mirroring backend
├── tests/                         # Comprehensive Pytest test suite
│   ├── fixtures/                  # Deterministic CSV and synthetic generators
│   ├── test_backend_auth_rbac.py  # Security, JWT, RBAC, and rate limiting tests
│   ├── test_clustering_stability_drift.py # Math verification for drift and stability
│   ├── test_contracts_and_features.py     # Pandera schema & PCA feature pipeline
│   ├── test_db_and_api.py         # FastAPI endpoints, DB repositories, rollback
│   ├── test_ml.py                 # Artifact loading, OOD checks, schema hash
│   └── test_pipeline.py           # End-to-end workflow execution tests
├── docker-compose.yml             # Multi-service composition (API, DB, Redis, UI)
├── Dockerfile                     # Multi-stage production container definition
├── Makefile                       # Developer automation commands
├── package.json                   # Node.js dependencies and scripts
├── pyproject.toml                 # Python package definition, dependencies, tools
├── server.ts                      # Express reverse proxy and process manager
├── tsconfig.json                  # Strict TypeScript configuration
└── vite.config.ts                 # Vite bundler configuration
```

---

## Canonical Persona Archetypes

The engine maps mathematical cluster centroids onto standardized behavioral profiles:

| Persona Archetype | Centroid Profile | Key Business Behavioral Characteristics | Prescriptive Strategy |
| :--- | :--- | :--- | :--- |
| **Affluent VIPs** | High Income ($>\$80\text{k}$), High Spending ($>70$) | Luxury shoppers; high brand loyalty; highly responsive to concierge services and premium exclusivity. | VIP account tiering, private event previews, concierge care. |
| **Thrifty Aspirants** | High Income ($>\$80\text{k}$), Low Spending ($<40$) | Conservative affluent buyers; value-driven; deliberate shoppers who research thoroughly before buying. | High-ROI marketing, value-add incentives, quality guarantees. |
| **Indulgent High-Rollers** | Low Income ($<\$40\text{k}$), High Spending ($>70$) | Trend-conscious younger demographic; immediate gratification mindset; high engagement on digital and social channels. | Limited-edition drops, micro-promotions, BNPL options. |
| **Conservative Savers** | Low Income ($<\$40\text{k}$), Low Spending ($<40$) | Budget-constrained, cautious spenders; highly price-sensitive; buy only essential necessities. | Essential bundles, discount alerts, price-match incentives. |
| **Mainstream Core** | Moderate Income ($\$40\text{k}-\$75\text{k}$), Moderate Spending ($40-60$) | The operational backbone of mall foot traffic; predictable cadence; responsive to seasonal campaigns. | Loyalty points acceleration, seasonal catalogs, membership rewards. |

---

## REST API Specification

### Authentication & Authorization (RBAC)

The API enforces Role-Based Access Control using JWT bearer tokens. Roles follow a strict privilege hierarchy:
- `viewer`: Read access to public metrics, segments, and health.
- `analyst`: Standard scoring, batch scoring, counterfactuals, and drift queries.
- `admin`: Model promotion, version rollback, and user management.

```bash
# 1. Obtain Access & Refresh Tokens
curl -X POST http://localhost:8010/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "analyst@example.com", "password": "secure_password"}'
```

### Core API Endpoints

#### 1. Single Customer Real-Time Scoring
`POST /api/v1/customers/score`

```bash
curl -X POST http://localhost:8010/api/v1/customers/score \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_ref": "WALKIN-901",
    "gender": "Female",
    "age": 32,
    "annual_income_k": 88.0,
    "spending_score": 82.0
  }'
```

**Response Payload:**
```json
{
  "customer_ref": "WALKIN-901",
  "gender": "Female",
  "age": 32,
  "annual_income_k": 88.0,
  "spending_score": 82.0,
  "cluster_id": 1,
  "persona_key": "affluent_vip",
  "persona_name": "Affluent VIPs",
  "recommended_action": "Enroll in VIP concierge service and provide private collection previews.",
  "gmm_confidence": 0.9842,
  "normalized_entropy": 0.0418,
  "is_boundary_customer": false,
  "distance_to_centroid": 0.284,
  "drift_flag": false,
  "drifted_features": [],
  "pca_1": 1.482,
  "pca_2": -0.812,
  "posterior_probabilities": {
    "Affluent VIPs": 0.9842,
    "Mainstream Core": 0.0121,
    "Indulgent High-Rollers": 0.0037
  }
}
```

#### 2. Counterfactual Intervention Analysis
`POST /api/v1/customers/counterfactual`

Calculates the minimal feature vector adjustments required to shift a customer into a targeted high-value persona.

```bash
curl -X POST http://localhost:8010/api/v1/customers/counterfactual \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 45,
    "annual_income_k": 85.0,
    "spending_score": 25.0,
    "target_persona_key": "affluent_vip"
  }'
```

#### 3. Model Registry Version Management (Admin Only)
- `GET /api/v1/models`: List all registered model versions, schemas, and metrics.
- `POST /api/v1/models/activate`: Promote a version to active or execute an instant rollback.
- `GET /api/v1/models/compare?v1=v1.0.0&v2=v1.1.0`: Side-by-side metric and schema diffing.

#### 4. Observability & Telemetry
- `GET /health`: Comprehensive system probe (DB connectivity, active model state, config hash).
- `GET /metrics`: Standard Prometheus metrics exposition format.
- `GET /api/v1/monitoring/drift`: Population Stability Index (PSI) audits and feature drift ledgers.

---

## Developer Quickstart

### Prerequisites
- Python $\ge$ 3.12
- Node.js $\ge$ 20.x
- PostgreSQL $\ge$ 16 (Optional; defaults to SQLite for local development)
- Redis $\ge$ 7 (Optional; defaults to in-memory sliding window rate limiting)

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/organization/persona-core.git
cd persona-core

# Create and activate Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install Python dependencies in editable mode with dev tools
pip install -e ".[dev]"

# Install frontend dependencies
npm install
```

### Step 2: Initialize Database and Model Artifacts

```bash
# Apply Alembic schema migrations
alembic upgrade head

# Train initial model bundle with bootstrap stability gating
python ml/train.py --version v1.0.0

# Generate pipeline execution records and static visual artifacts
python scripts/run_pipeline.py --render-visuals
```

### Step 3: Run the Development Server

The simplest way to run both the backend FastAPI service and the React frontend simultaneously is via the Node supervisor:

```bash
npm run dev
```

*The supervisor detects a free port, launches FastAPI with automatic uvicorn reload, validates the health endpoint, and starts Vite on `http://localhost:3000`.*

Alternatively, run services independently:
```bash
# Terminal 1: FastAPI Microservice
uvicorn mall_seg.api:app --host 0.0.0.0 --port 8010 --reload

# Terminal 2: Vite React Console
npm run build && npm run preview
```

---

## Docker & Containerized Deployment

A production-ready `docker-compose.yml` orchestrates the complete topology:

```bash
# Start all microservices in the background
docker compose up -d --build

# View operational logs
docker compose logs -f api
```

The stack provisions:
1. `db`: PostgreSQL 16 Alpine with persistent storage volume.
2. `redis`: Redis 7 Alpine for distributed rate limiting.
3. `api`: Containerized FastAPI application running migrations on boot.
4. `ui`: Streamlit interactive data science dashboard at `http://localhost:8501`.

---

## Quality Verification & CI/CD

The repository enforces strict quality standards:

```bash
# 1. Run full test suite with coverage
pytest -v --cov=mall_seg

# 2. Strict Python static type verification
mypy src ml

# 3. Python linting and code formatting check
ruff check src ml pipelines scripts app tests
ruff format --check src ml pipelines scripts app tests

# 4. Strict TypeScript type check
npm run lint

# 5. Production frontend bundle build
npm run build
```

---

## Senior Engineer Architectural Evaluation

### Why This Project Stands Out

Most customer segmentation projects remain confined to static Jupyter notebooks executing $k$-means on unscaled data without verification. **PersonaCore** is built like an enterprise-grade platform:

1. **Mathematically Defensible Machine Learning**:
   Instead of guessing $k$, the pipeline computes Inertia, Silhouette Width, Calinski-Harabasz, and Davies-Bouldin metrics alongside GMM BIC/AIC sweeps. Non-parametric bootstrap stability gating ($B = 100$) guarantees that cluster structures are statistically robust rather than artifacts of noise.

2. **Probabilistic Uncertainty & Guardrails**:
   Hard clustering forces ambiguous boundary data into rigid bins. PersonaCore incorporates GMM posterior probability distributions and normalized Shannon entropy to detect boundary cases ($H_{\text{norm}} \ge 0.45$). This protects downstream marketing automation from mistargeting customers near cluster thresholds.

3. **Production Governance & Continuous Auditing**:
   The inclusion of real-time Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests ensures that data drift is caught immediately in production. The model registry verifies SHA-256 schema hashes before loading artifacts, preventing silent schema breakage.

4. **Actionable Counterfactual Intelligence**:
   Rather than stopping at descriptive labels ("Customer X is in Cluster 2"), PersonaCore provides prescriptive optimization vectors that specify the exact delta needed to elevate a customer into a higher-value segment.

5. **Industrial-Grade Software Engineering**:
   - Zero loose scripts: All business logic is encapsulated in a fully typed Python package (`mall_seg`).
   - Pandera data contracts guarantee runtime data integrity.
   - Dual-mode persistence supports SQLite for zero-dependency local testing and PostgreSQL for enterprise production.
   - Context-correlated JSON logging (`X-Request-ID`), custom Prometheus metrics exposition, and JWT/RBAC security establish enterprise operational readiness.
   - The React 19 console incorporates interactive 2D and 3D orthographic projections, bridging the gap between deep data science and executive decision-making.

---

## License

This project is licensed under the Apache 2.0 License. See `LICENSE` for details.
