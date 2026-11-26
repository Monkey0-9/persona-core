"""FastAPI application with JWT/RBAC, model registry governance, batch scoring & observability."""

from __future__ import annotations

import hmac
import io
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import pandera.errors
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from mall_seg.alerts import dispatch_drift_alert
from mall_seg.contracts import (
    NUMERIC_FEATURE_COLUMNS,
    AppConfig,
    load_config,
    load_raw_dataset,
    validate_raw_dataframe,
)
from mall_seg.db import (
    PipelineRun,
    check_db_connectivity,
    count_users,
    create_db_engine,
    create_user,
    get_active_model_version,
    get_latest_pipeline_run,
    get_session_factory,
    get_user_by_email,
    get_user_by_id,
    get_valid_refresh_token,
    init_db,
    list_model_versions,
    query_drift_reports,
    query_scored_customers,
    query_scoring_events,
    record_batch_scores_and_drift,
    record_scoring_event,
    revoke_refresh_token,
    store_refresh_token,
)
from mall_seg.drift import evaluate_batch_drift
from mall_seg.observability import (
    configure_structured_logging,
    metrics_registry,
    request_id_ctx,
)
from mall_seg.personas import (
    ModelSchemaMismatchError,
    compute_counterfactual_path,
    execute_segmentation_workflow,
    score_customer_records,
    verify_bundle_schema_compatibility,
)
from mall_seg.rate_limit import check_redis_connectivity, enforce_rate_limit
from mall_seg.registry import (
    activate_or_rollback_version,
    compare_model_versions,
    register_candidate_model,
    serialize_model_version,
)
from mall_seg.security import (
    RoleName,
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from ml.drift import evaluate_rolling_window_drift

load_dotenv()
configure_structured_logging()
logger = logging.getLogger("mall_seg.api")

_SAFE_REQ_ID_RE = re.compile(r"^[A-Za-z0-9._-]{4,64}$")
_SEMVER_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9]+)?$")


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CustomerScoreRequest(StrictBaseModel):
    customer_ref: str | None = Field(default=None, max_length=64)
    gender: Literal["Male", "Female"]
    age: int = Field(ge=16, le=100)
    annual_income_k: float = Field(ge=5.0, le=500.0)
    spending_score: float = Field(ge=1.0, le=100.0)


class CounterfactualRequest(StrictBaseModel):
    customer_ref: str | None = Field(default=None, max_length=64)
    gender: Literal["Male", "Female"]
    age: int = Field(ge=16, le=100)
    annual_income_k: float = Field(ge=5.0, le=500.0)
    spending_score: float = Field(ge=1.0, le=100.0)
    target_persona_key: str | None = Field(default=None, max_length=64)


class CustomerScoreResponse(BaseModel):
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
    distance_to_centroid: float = 0.0
    drift_flag: bool = False
    drifted_features: list[str] = Field(default_factory=list)
    pca_1: float
    pca_2: float
    posterior_probabilities: dict[str, float] = Field(default_factory=dict)


class BatchScoreRequest(StrictBaseModel):
    customers: list[CustomerScoreRequest] = Field(min_length=1, max_length=1000)


class BatchCustomersLenientRequest(StrictBaseModel):
    batch_id: str | None = Field(default=None, max_length=64)
    customers: list[dict[str, Any]] = Field(min_length=1, max_length=2000)


class BatchRowError(BaseModel):
    row_index: int
    customer_ref: str | None = None
    errors: list[str]


class BatchCustomersResponse(BaseModel):
    batch_id: str
    count: int
    accepted_count: int
    rejected_count: int
    boundary_count: int
    row_errors: list[BatchRowError]
    predictions: list[CustomerScoreResponse]
    drift_report: dict[str, Any]


class BatchScoreResponse(BaseModel):
    count: int
    boundary_count: int
    predictions: list[CustomerScoreResponse]
    drift_report: dict[str, Any]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database_ok: bool
    model_loaded: bool
    active_run_key: str | None


class ActiveModelResponse(BaseModel):
    version: str
    algorithm: str
    feature_schema_hash: str
    artifact_path: str
    metrics: dict[str, Any]
    reference_stats: dict[str, dict[str, float]]
    created_at: str


class RegisterCandidateRequest(StrictBaseModel):
    version: str = Field(min_length=3, max_length=32)
    target_k: int = Field(default=5, ge=2, le=10)
    include_gender: bool = False
    notes: str | None = Field(default=None, max_length=500)


class RollbackModelRequest(StrictBaseModel):
    target_version: str | None = Field(default=None, max_length=32)


class RollingDriftEvaluateRequest(StrictBaseModel):
    window_size: int = Field(default=100, ge=2, le=5000)
    lookback_days: int | None = Field(default=7, ge=1, le=365)


class CustomersPageResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[dict[str, Any]]


class CustomCustomerRow(StrictBaseModel):
    customer_id: int = Field(gt=0)
    gender: Literal["Male", "Female"]
    age: int = Field(ge=16, le=100)
    annual_income_k: float = Field(ge=5.0, le=500.0)
    spending_score: float = Field(ge=1.0, le=100.0)


class CustomTrainRequest(StrictBaseModel):
    customers: list[CustomCustomerRow] = Field(min_length=15, max_length=5000)
    target_k: int = Field(default=5, ge=2, le=10)
    include_gender: bool = False
    entropy_threshold: float = Field(default=0.45, ge=0.05, le=0.95)
    enforce_stability_gate: bool = False


class LoginRequest(StrictBaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class BootstrapAdminRequest(StrictBaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)


class CreateUserRequest(StrictBaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, max_length=128)
    role: RoleName = "analyst"


class RefreshRequest(StrictBaseModel):
    refresh_token: str = Field(min_length=16, max_length=512)


class AuthUserResponse(BaseModel):
    id: int
    email: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: AuthUserResponse


class AuthContext(BaseModel):
    auth_mode: Literal["jwt", "api_key"]
    user_id: int | None = None
    email: str | None = None
    role: str = "admin"


def _load_model_bundle(bundle_path: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(bundle_path)
    if not path.exists():
        return None, None
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict):
            return None, "Model bundle artifact is not a valid dictionary."
        verify_bundle_schema_compatibility(bundle)
        return bundle, None
    except ModelSchemaMismatchError as exc:
        logger.error("Training-serving skew guard blocked bundle %s: %s", path, exc)
        return None, str(exc)
    except Exception as exc:
        logger.error("Failed to load model bundle %s: %s", path, exc)
        return None, str(exc)


def create_app(
    config: AppConfig | None = None,
    database_url: str | None = None,
    bundle_path: str | Path | None = None,
) -> FastAPI:
    cfg = config or load_config()
    engine = create_db_engine(database_url)
    init_db(engine)
    session_factory = get_session_factory(engine)
    resolved_bundle_path = Path(bundle_path or cfg.artifacts.model_bundle_path)
    is_prod = os.getenv("ENV", "development").lower() == "production"

    def _bootstrap_if_empty() -> None:
        if os.getenv("MALL_SEG_AUTO_BOOTSTRAP", "1") == "0":
            return
        if application.state.schema_mismatch_error is not None:
            return
        has_bundle = application.state.model_bundle is not None
        has_run = False
        session = session_factory()
        try:
            has_run = get_latest_pipeline_run(session) is not None
        finally:
            session.close()
        if has_bundle and has_run:
            return
        logger.info("Database or model bundle empty; auto-bootstrapping default segmentation run.")
        raw_df, source_kind, _ = load_raw_dataset(cfg)
        result = execute_segmentation_workflow(
            raw_df=raw_df,
            cfg=cfg,
            dataset_source=source_kind,
            engine=engine,
            bundle_path=resolved_bundle_path,
            snapshot_path=cfg.artifacts.frontend_snapshot_path,
            enforce_stability_gate=True,
            log_mlflow=False,
        )
        application.state.model_bundle = result["bundle"]
        application.state.schema_mismatch_error = None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.config = cfg
        application.state.engine = engine
        application.state.session_factory = session_factory
        application.state.bundle_path = resolved_bundle_path
        loaded_bundle, skew_err = _load_model_bundle(resolved_bundle_path)
        application.state.model_bundle = loaded_bundle
        application.state.schema_mismatch_error = skew_err
        _bootstrap_if_empty()
        yield

    application = FastAPI(
        title="Mall Customer Segmentation API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/openapi.json",
    )

    application.state.config = cfg
    application.state.engine = engine
    application.state.session_factory = session_factory
    application.state.bundle_path = resolved_bundle_path
    init_bundle, init_skew = _load_model_bundle(resolved_bundle_path)
    application.state.model_bundle = init_bundle
    application.state.schema_mismatch_error = init_skew

    cors_raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:8501",
    )
    allowed_origins = [origin.strip() for origin in cors_raw.split(",") if origin.strip()]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )

    @application.middleware("http")
    async def add_security_and_telemetry(request: Request, call_next: Any) -> Any:
        incoming_id = (request.headers.get("X-Request-ID") or "").strip()
        request_id = (
            incoming_id if _SAFE_REQ_ID_RE.match(incoming_id) else f"req-{uuid.uuid4().hex[:12]}"
        )
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            elapsed_sec = time.perf_counter() - started
            elapsed_ms = elapsed_sec * 1000.0
            metrics_registry.observe_http_request(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_seconds=elapsed_sec,
            )
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )
            request_id_ctx.reset(token)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if request.url.path.startswith("/api/v1/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(Exception)
    async def clean_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error on %s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "request_id": request_id_ctx.get()},
        )

    def get_db() -> Any:
        session = application.state.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def require_roles(
        *allowed_roles: RoleName,
    ) -> Callable[..., AuthContext]:
        allowed_set = set(allowed_roles) if allowed_roles else {"admin", "analyst", "viewer"}

        def _dependency(
            authorization: str | None = Header(default=None, alias="Authorization"),
            x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        ) -> AuthContext:
            if authorization and authorization.lower().startswith("bearer "):
                token = authorization.split(" ", 1)[1].strip()
                claims = decode_access_token(token)
                if claims is None:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired Bearer token.",
                    )
                role = str(claims.get("role", "viewer"))
                if role not in allowed_set:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Role '{role}' is not permitted for this operation.",
                    )
                sub = claims.get("sub")
                return AuthContext(
                    auth_mode="jwt",
                    user_id=int(sub) if sub is not None else None,
                    email=str(claims.get("email", "")),
                    role=role,
                )

            expected_key = os.getenv("API_KEY", "")
            if expected_key and x_api_key and hmac.compare_digest(x_api_key, expected_key):
                return AuthContext(auth_mode="api_key", user_id=None, email=None, role="admin")

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authentication credentials.",
            )

        return _dependency

    def _require_bundle() -> dict[str, Any]:
        bundle: dict[str, Any] | None = application.state.model_bundle
        if bundle is None and application.state.schema_mismatch_error is None:
            bundle, skew_err = _load_model_bundle(application.state.bundle_path)
            application.state.model_bundle = bundle
            application.state.schema_mismatch_error = skew_err
        if application.state.schema_mismatch_error is not None:
            skew_err_msg = application.state.schema_mismatch_error
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model schema skew guard blocked inference: {skew_err_msg}",
            )
        if bundle is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model bundle is not loaded. Run the training pipeline first.",
            )
        return bundle

    def _issue_session_tokens(db: Session, user_id: int, email: str, role: str) -> TokenResponse:
        access_ttl = 1800
        access_token = create_access_token(
            user_id=user_id, email=email, role=role, expires_in_seconds=access_ttl
        )
        raw_refresh = generate_refresh_token()
        store_refresh_token(
            db,
            user_id=user_id,
            token_hash=hash_refresh_token(raw_refresh),
            expires_at=datetime.now(UTC) + timedelta(days=7),
        )
        return TokenResponse(
            access_token=access_token,
            refresh_token=raw_refresh,
            expires_in=access_ttl,
            user=AuthUserResponse(id=user_id, email=email, role=role),
        )

    @application.get("/health/live")
    def health_live() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/health/ready", response_model=HealthResponse)
    def health_ready(response: Response, db: Session = Depends(get_db)) -> HealthResponse:
        res = health(db)
        if res.status != "ok":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return res

    @application.get("/health", response_model=HealthResponse)
    def health(db: Session = Depends(get_db)) -> HealthResponse:
        db_ok = check_db_connectivity(application.state.engine)
        bundle = application.state.model_bundle
        if bundle is None and application.state.schema_mismatch_error is None:
            bundle, skew_err = _load_model_bundle(application.state.bundle_path)
            application.state.model_bundle = bundle
            application.state.schema_mismatch_error = skew_err
        latest_run = get_latest_pipeline_run(db) if db_ok else None
        model_loaded = bundle is not None and application.state.schema_mismatch_error is None
        overall: Literal["ok", "degraded"] = "ok" if (db_ok and model_loaded) else "degraded"
        return HealthResponse(
            status=overall,
            database_ok=db_ok,
            model_loaded=model_loaded,
            active_run_key=latest_run.run_key if latest_run else None,
        )

    @application.get("/metrics")
    def prometheus_metrics(db: Session = Depends(get_db)) -> PlainTextResponse:
        db_ok = check_db_connectivity(application.state.engine)
        redis_ok = check_redis_connectivity()
        bundle = application.state.model_bundle
        active_mv = get_active_model_version(db) if db_ok else None
        text_body = metrics_registry.render_prometheus(
            db_up=db_ok,
            redis_up=redis_ok,
            model_loaded=bundle is not None and application.state.schema_mismatch_error is None,
            active_version=active_mv.version if active_mv else None,
        )
        return PlainTextResponse(
            content=text_body,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @application.get("/api/v1/observability/summary")
    def observability_summary(
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> dict[str, Any]:
        db_ok = check_db_connectivity(application.state.engine)
        redis_ok = check_redis_connectivity()
        active_mv = get_active_model_version(db) if db_ok else None
        return metrics_registry.summary(
            db_up=db_ok,
            redis_up=redis_ok,
            model_loaded=(
                application.state.model_bundle is not None
                and application.state.schema_mismatch_error is None
            ),
            active_version=active_mv.version if active_mv else None,
        )

    @application.get("/api/v1/auth/status")
    def auth_status(db: Session = Depends(get_db)) -> dict[str, Any]:
        return {"has_users": count_users(db) > 0}

    @application.post("/api/v1/auth/bootstrap-admin", response_model=TokenResponse)
    def bootstrap_admin(
        payload: BootstrapAdminRequest,
        request: Request,
        db: Session = Depends(get_db),
    ) -> TokenResponse:
        enforce_rate_limit(request, scope="auth_bootstrap", max_requests=10, window_seconds=60)
        if count_users(db) > 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Admin account already initialized. Please sign in.",
            )
        user = create_user(
            db,
            email=payload.email,
            password_hash=hash_password(payload.password),
            role="admin",
        )
        return _issue_session_tokens(db, user.id, user.email, user.role)

    @application.post("/api/v1/auth/login", response_model=TokenResponse)
    def login(
        payload: LoginRequest,
        request: Request,
        db: Session = Depends(get_db),
    ) -> TokenResponse:
        enforce_rate_limit(request, scope="auth_login", max_requests=20, window_seconds=60)
        user = get_user_by_email(db, payload.email)
        if user is None or not verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
            )
        return _issue_session_tokens(db, user.id, user.email, user.role)

    @application.post("/api/v1/auth/refresh", response_model=TokenResponse)
    def refresh_session(
        payload: RefreshRequest,
        request: Request,
        db: Session = Depends(get_db),
    ) -> TokenResponse:
        enforce_rate_limit(request, scope="auth_refresh", max_requests=30, window_seconds=60)
        tok_hash = hash_refresh_token(payload.refresh_token)
        rec = get_valid_refresh_token(db, tok_hash)
        if rec is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token.",
            )
        revoke_refresh_token(db, tok_hash)
        user = get_user_by_id(db, rec.user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
        return _issue_session_tokens(db, user.id, user.email, user.role)

    @application.post("/api/v1/auth/logout")
    def logout(payload: RefreshRequest, db: Session = Depends(get_db)) -> dict[str, str]:
        revoke_refresh_token(db, hash_refresh_token(payload.refresh_token))
        return {"status": "logged_out"}

    @application.post(
        "/api/v1/auth/users",
        response_model=AuthUserResponse,
    )
    def create_user_account(
        payload: CreateUserRequest,
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin")),
    ) -> AuthUserResponse:
        if get_user_by_email(db, payload.email) is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered.",
            )
        user = create_user(
            db,
            email=payload.email,
            password_hash=hash_password(payload.password),
            role=payload.role,
        )
        return AuthUserResponse(id=user.id, email=user.email, role=user.role)

    @application.get("/api/v1/auth/me")
    def auth_me(
        auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> AuthContext:
        return auth

    async def _score_single_impl(
        payload: CustomerScoreRequest,
        request: Request,
        db: Session,
        auth: AuthContext,
    ) -> CustomerScoreResponse:
        enforce_rate_limit(request, scope="score", max_requests=120, window_seconds=60)
        bundle = _require_bundle()
        predictions = await run_in_threadpool(
            score_customer_records,
            [payload.model_dump()],
            bundle,
        )
        pred_dict = asdict(predictions[0])
        latest_run = get_latest_pipeline_run(db)
        active_mv = get_active_model_version(db)
        record_scoring_event(
            db,
            run_id=latest_run.id if latest_run else None,
            prediction=pred_dict,
            user_id=auth.user_id,
            model_version_id=active_mv.id if active_mv else None,
        )
        metrics_registry.inc_persona_score(str(pred_dict["persona_name"]))
        logger.info(
            "customer_scored",
            extra={
                "persona": pred_dict["persona_name"],
                "distance_to_centroid": pred_dict["distance_to_centroid"],
                "drift_flag": pred_dict["drift_flag"],
                "user_id": auth.user_id,
            },
        )
        return CustomerScoreResponse(**pred_dict)

    @application.post(
        "/api/v1/score",
        response_model=CustomerScoreResponse,
    )
    async def score_single(
        payload: CustomerScoreRequest,
        request: Request,
        db: Session = Depends(get_db),
        auth: AuthContext = Depends(require_roles("admin", "analyst")),
    ) -> CustomerScoreResponse:
        return await _score_single_impl(payload, request, db, auth)

    @application.post(
        "/api/v1/customers/score",
        response_model=CustomerScoreResponse,
    )
    async def score_single_customer_route(
        payload: CustomerScoreRequest,
        request: Request,
        db: Session = Depends(get_db),
        auth: AuthContext = Depends(require_roles("admin", "analyst")),
    ) -> CustomerScoreResponse:
        return await _score_single_impl(payload, request, db, auth)

    @application.post("/api/v1/customers/counterfactual")
    async def counterfactual_customer_route(
        payload: CounterfactualRequest,
        request: Request,
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> dict[str, Any]:
        enforce_rate_limit(request, scope="score", max_requests=120, window_seconds=60)
        bundle = _require_bundle()
        return await run_in_threadpool(
            compute_counterfactual_path,
            payload.model_dump(exclude={"target_persona_key"}),
            bundle,
            payload.target_persona_key,
        )

    async def _run_batch_scoring_with_row_validation(
        *,
        raw_rows: list[dict[str, Any]],
        batch_id: str | None,
        db: Session,
        auth: AuthContext,
    ) -> BatchCustomersResponse:
        bundle = _require_bundle()
        resolved_batch_id = batch_id or f"batch-{uuid.uuid4().hex[:12]}"
        valid_records: list[dict[str, Any]] = []
        row_errors: list[BatchRowError] = []

        for idx, raw_row in enumerate(raw_rows):
            try:
                validated = CustomerScoreRequest.model_validate(raw_row)
                valid_records.append(validated.model_dump())
            except ValidationError as exc:
                errs = [
                    f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}" for e in exc.errors()
                ]
                ref_val = raw_row.get("customer_ref") if isinstance(raw_row, dict) else None
                row_errors.append(
                    BatchRowError(
                        row_index=idx,
                        customer_ref=str(ref_val) if ref_val is not None else None,
                        errors=errs,
                    )
                )

        if not valid_records:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": "No valid customer rows in batch.",
                    "row_errors": [r.model_dump() for r in row_errors[:25]],
                },
            )

        predictions = await run_in_threadpool(score_customer_records, valid_records, bundle)
        pred_dicts = [asdict(p) for p in predictions]
        for p in pred_dicts:
            metrics_registry.inc_persona_score(str(p["persona_name"]))

        current_df = pd.DataFrame(
            [
                {
                    "Age": r["age"],
                    "Annual Income (k$)": r["annual_income_k"],
                    "Spending Score (1-100)": r["spending_score"],
                }
                for r in valid_records
            ]
        )
        drift_report = evaluate_batch_drift(
            baseline_distributions=bundle["baseline_distributions"],
            current_df=current_df,
            features=list(NUMERIC_FEATURE_COLUMNS),
            bins=cfg.drift.bins,
            stable_threshold=cfg.drift.stable_threshold,
            monitor_threshold=cfg.drift.monitor_threshold,
        ).to_dict()

        alert_status = str(drift_report.get("alert_status", "NORMAL"))
        dispatched = False
        if alert_status in {"WARNING", "CRITICAL"}:
            metrics_registry.inc_drift_alert(alert_status)
            dispatched = dispatch_drift_alert(
                alert_status=alert_status,
                max_psi=float(drift_report["max_psi"]),
                max_ks_statistic=float(drift_report["max_ks_statistic"]),
                window_label=f"batch:{resolved_batch_id}",
                sample_size=len(pred_dicts),
                details={"batch_id": resolved_batch_id},
            )

        latest_run = get_latest_pipeline_run(db)
        if latest_run is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No pipeline run persisted in database.",
            )
        active_mv = get_active_model_version(db)
        record_batch_scores_and_drift(
            db,
            run_id=latest_run.id,
            scored_rows=pred_dicts,
            drift_report=drift_report,
            batch_id=resolved_batch_id,
            user_id=auth.user_id,
            model_version_id=active_mv.id if active_mv else None,
            alert_dispatched=dispatched,
        )
        logger.info(
            "batch_scored",
            extra={
                "batch_id": resolved_batch_id,
                "accepted_count": len(pred_dicts),
                "rejected_count": len(row_errors),
                "alert_status": alert_status,
            },
        )
        boundary_count = sum(1 for item in pred_dicts if item["is_boundary_customer"])
        return BatchCustomersResponse(
            batch_id=resolved_batch_id,
            count=len(pred_dicts),
            accepted_count=len(pred_dicts),
            rejected_count=len(row_errors),
            boundary_count=boundary_count,
            row_errors=row_errors,
            predictions=[CustomerScoreResponse(**p) for p in pred_dicts],
            drift_report=drift_report,
        )

    @application.post(
        "/api/v1/batch-score",
        response_model=BatchScoreResponse,
    )
    async def score_batch(
        payload: BatchScoreRequest,
        request: Request,
        db: Session = Depends(get_db),
        auth: AuthContext = Depends(require_roles("admin", "analyst")),
    ) -> BatchScoreResponse:
        enforce_rate_limit(request, scope="batch_score", max_requests=30, window_seconds=60)
        res = await _run_batch_scoring_with_row_validation(
            raw_rows=[item.model_dump() for item in payload.customers],
            batch_id=None,
            db=db,
            auth=auth,
        )
        return BatchScoreResponse(
            count=res.count,
            boundary_count=res.boundary_count,
            predictions=res.predictions,
            drift_report=res.drift_report,
        )

    @application.post(
        "/api/v1/customers/batch-score",
        response_model=BatchCustomersResponse,
    )
    async def score_customers_batch(
        payload: BatchCustomersLenientRequest,
        request: Request,
        db: Session = Depends(get_db),
        auth: AuthContext = Depends(require_roles("admin", "analyst")),
    ) -> BatchCustomersResponse:
        enforce_rate_limit(request, scope="batch_score", max_requests=30, window_seconds=60)
        return await _run_batch_scoring_with_row_validation(
            raw_rows=payload.customers,
            batch_id=payload.batch_id,
            db=db,
            auth=auth,
        )

    @application.post(
        "/api/v1/customers/batch-score/upload",
        response_model=BatchCustomersResponse,
    )
    async def upload_batch_score_file(
        request: Request,
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        auth: AuthContext = Depends(require_roles("admin", "analyst")),
    ) -> BatchCustomersResponse:
        enforce_rate_limit(request, scope="batch_score", max_requests=20, window_seconds=60)
        filename = (file.filename or "").lower()
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Uploaded batch file exceeds 10 MB limit.",
            )
        try:
            if filename.endswith(".parquet"):
                df = pd.read_parquet(io.BytesIO(content))
            elif filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported file format. Upload a .csv or .parquet file.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Failed to parse uploaded file: {exc}",
            ) from exc

        normalized_rows: list[dict[str, Any]] = []
        for raw_rec in df.to_dict(orient="records"):
            row = dict(raw_rec)
            ref_raw = row.get("customer_ref") or row.get("CustomerID")
            normalized_rows.append(
                {
                    "customer_ref": (None if pd.isna(ref_raw) else str(ref_raw)),
                    "gender": row.get("gender", row.get("Gender")),
                    "age": row.get("age", row.get("Age")),
                    "annual_income_k": row.get("annual_income_k", row.get("Annual Income (k$)")),
                    "spending_score": row.get("spending_score", row.get("Spending Score (1-100)")),
                }
            )
        return await _run_batch_scoring_with_row_validation(
            raw_rows=normalized_rows,
            batch_id=None,
            db=db,
            auth=auth,
        )

    def _serialize_pipeline_run(run_row: PipelineRun) -> dict[str, Any]:
        summary = json.loads(run_row.summary_json)
        return {
            "run_key": run_row.run_key,
            "dataset_source": run_row.dataset_source,
            "row_count": run_row.row_count,
            "target_k": run_row.target_k,
            "include_gender": run_row.include_gender,
            "silhouette_score": run_row.silhouette_score,
            "mean_jaccard_stability": run_row.mean_jaccard_stability,
            "passed_stability_gate": run_row.passed_stability_gate,
            "surrogate_fidelity": run_row.surrogate_fidelity,
            "created_at": run_row.created_at.isoformat(),
            **summary,
        }

    @application.get("/api/v1/segments")
    def get_segments(
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> dict[str, Any]:
        latest_run = get_latest_pipeline_run(db)
        if latest_run is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No pipeline run found.",
            )
        return _serialize_pipeline_run(latest_run)

    @application.get("/api/v1/models/active", response_model=ActiveModelResponse)
    def get_active_model(
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> ActiveModelResponse:
        bundle = _require_bundle()
        mv = get_active_model_version(db)
        if mv is not None:
            return ActiveModelResponse(
                version=mv.version,
                algorithm=mv.algorithm,
                feature_schema_hash=mv.feature_schema_hash,
                artifact_path=mv.artifact_path,
                metrics=json.loads(mv.metrics_json),
                reference_stats=bundle.get("reference_stats") or {},
                created_at=mv.created_at.isoformat(),
            )
        return ActiveModelResponse(
            version=str(bundle.get("model_version", "v1.0.0")),
            algorithm="K-Means (k-means++) + GMM",
            feature_schema_hash=str(bundle.get("feature_schema_hash", "legacy")),
            artifact_path=str(application.state.bundle_path),
            metrics={"run_key": str(bundle.get("run_key", "unknown"))},
            reference_stats=bundle.get("reference_stats") or {},
            created_at=datetime.now(UTC).isoformat(),
        )

    @application.get("/api/v1/models")
    def list_registered_models(
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> dict[str, Any]:
        versions = [serialize_model_version(v) for v in list_model_versions(db)]
        return {"total": len(versions), "items": versions}

    @application.post("/api/v1/models/candidates")
    def register_candidate(
        payload: RegisterCandidateRequest,
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin")),
    ) -> dict[str, Any]:
        if not _SEMVER_RE.match(payload.version):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Version must match semantic tag format such as v1.1.0.",
            )
        active_mv = get_active_model_version(db)
        try:
            res = register_candidate_model(
                version=payload.version,
                cfg=cfg,
                engine=application.state.engine,
                bundle_path=Path(application.state.bundle_path),
                target_k=payload.target_k,
                include_gender=payload.include_gender,
                notes=payload.notes,
                parent_version=active_mv.version if active_mv else None,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        cmp_res = compare_model_versions(
            db,
            candidate_version=payload.version,
            default_bundle_path=Path(application.state.bundle_path),
        )
        return {
            "version": payload.version,
            "run_key": res["run_key"],
            "comparison": cmp_res,
        }

    @application.get("/api/v1/models/{version}/compare")
    def compare_candidate_model(
        version: str,
        baseline: str | None = Query(default=None),
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> dict[str, Any]:
        try:
            return compare_model_versions(
                db,
                candidate_version=version,
                baseline_version=baseline,
                default_bundle_path=Path(application.state.bundle_path),
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    @application.post("/api/v1/models/{version}/promote")
    def promote_candidate_model(
        version: str,
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin")),
    ) -> dict[str, Any]:
        try:
            mv, bundle = activate_or_rollback_version(
                db,
                target_version=version,
                default_bundle_path=Path(application.state.bundle_path),
                rollback=False,
            )
        except ModelSchemaMismatchError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        application.state.model_bundle = bundle
        application.state.schema_mismatch_error = None
        return serialize_model_version(mv)

    @application.post("/api/v1/models/rollback")
    def rollback_active_model(
        payload: RollbackModelRequest | None = None,
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin")),
    ) -> dict[str, Any]:
        target = payload.target_version if payload else None
        try:
            mv, bundle = activate_or_rollback_version(
                db,
                target_version=target,
                default_bundle_path=Path(application.state.bundle_path),
                rollback=True,
            )
        except ModelSchemaMismatchError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        application.state.model_bundle = bundle
        application.state.schema_mismatch_error = None
        return serialize_model_version(mv)

    @application.get("/api/v1/monitoring/drift")
    def get_monitoring_drift(
        limit: int = Query(default=10, ge=1, le=50),
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> dict[str, Any]:
        bundle = _require_bundle()
        rows, total = query_drift_reports(db, limit=limit, offset=0)
        latest = rows[0] if rows else None
        return {
            "total_reports": total,
            "latest_status": latest.overall_status if latest else "stable",
            "latest_alert_status": latest.alert_status if latest else "NORMAL",
            "latest_max_psi": latest.max_psi if latest else 0.0,
            "latest_max_ks": latest.max_ks_statistic if latest else 0.0,
            "reference_stats": bundle.get("reference_stats") or {},
            "recent_reports": [
                {
                    "id": r.id,
                    "batch_size": r.batch_size,
                    "overall_status": r.overall_status,
                    "alert_status": r.alert_status,
                    "max_psi": r.max_psi,
                    "max_ks_statistic": r.max_ks_statistic,
                    "window_label": r.window_label,
                    "alert_dispatched": r.alert_dispatched,
                    "features": json.loads(r.feature_psi_json),
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ],
        }

    @application.post("/api/v1/monitoring/drift/evaluate")
    def trigger_rolling_drift_evaluation(
        payload: RollingDriftEvaluateRequest | None = None,
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst")),
    ) -> dict[str, Any]:
        bundle = _require_bundle()
        req_cfg = payload or RollingDriftEvaluateRequest()
        result = evaluate_rolling_window_drift(
            db,
            bundle=bundle,
            window_size=req_cfg.window_size,
            lookback_days=req_cfg.lookback_days,
            stable_threshold=cfg.drift.stable_threshold,
            monitor_threshold=cfg.drift.monitor_threshold,
            bins=cfg.drift.bins,
        )
        if result.get("evaluated") and result.get("alert_status") in {"WARNING", "CRITICAL"}:
            metrics_registry.inc_drift_alert(str(result["alert_status"]))
        return result

    @application.post("/api/v1/train-custom")
    def train_custom_dataset(
        payload: CustomTrainRequest,
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst")),
    ) -> dict[str, Any]:
        if payload.target_k >= len(payload.customers):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="target_k must be strictly smaller than the number of customer rows.",
            )

        raw_df = pd.DataFrame(
            [
                {
                    "CustomerID": row.customer_id,
                    "Gender": row.gender,
                    "Age": row.age,
                    "Annual Income (k$)": row.annual_income_k,
                    "Spending Score (1-100)": row.spending_score,
                }
                for row in payload.customers
            ]
        )
        try:
            validated_df = validate_raw_dataframe(raw_df, min_rows=15)
        except (pandera.errors.SchemaErrors, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Custom dataset contract violation: {exc}",
            ) from exc

        custom_cfg = cfg.model_copy(
            update={
                "privacy": cfg.privacy.model_copy(
                    update={"include_gender": payload.include_gender}
                ),
                "clustering": cfg.clustering.model_copy(
                    update={
                        "target_k": payload.target_k,
                        "k_max": max(cfg.clustering.k_max, payload.target_k),
                        "entropy_threshold": payload.entropy_threshold,
                    }
                ),
            }
        )
        try:
            result = execute_segmentation_workflow(
                raw_df=validated_df,
                cfg=custom_cfg,
                dataset_source="custom",
                engine=application.state.engine,
                bundle_path=application.state.bundle_path,
                enforce_stability_gate=payload.enforce_stability_gate,
                log_mlflow=False,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc

        application.state.model_bundle = result["bundle"]
        application.state.schema_mismatch_error = None
        latest_run = get_latest_pipeline_run(db)
        if latest_run is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Pipeline run persistence failed.",
            )
        return _serialize_pipeline_run(latest_run)

    @application.post("/api/v1/reset-default")
    def reset_default_dataset(
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst")),
    ) -> dict[str, Any]:
        raw_df, source_kind, _ = load_raw_dataset(cfg)
        result = execute_segmentation_workflow(
            raw_df=raw_df,
            cfg=cfg,
            dataset_source=source_kind,
            engine=application.state.engine,
            bundle_path=application.state.bundle_path,
            enforce_stability_gate=True,
            log_mlflow=False,
        )
        application.state.model_bundle = result["bundle"]
        application.state.schema_mismatch_error = None
        latest_run = get_latest_pipeline_run(db)
        if latest_run is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Default pipeline run restoration failed.",
            )
        return _serialize_pipeline_run(latest_run)

    @application.get("/api/v1/customers", response_model=CustomersPageResponse)
    def list_customers(
        segment: str | None = Query(default=None),
        boundary_only: bool = Query(default=False),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> CustomersPageResponse:
        rows, total = query_scored_customers(
            db,
            segment=segment,
            boundary_only=boundary_only,
            limit=limit,
            offset=offset,
        )
        items = [
            {
                "id": r.id,
                "customer_ref": r.customer_ref,
                "gender": r.gender,
                "age": r.age,
                "annual_income_k": r.annual_income_k,
                "spending_score": r.spending_score,
                "cluster_id": r.cluster_id,
                "persona_name": r.persona_name,
                "recommended_action": r.recommended_action,
                "gmm_confidence": r.gmm_confidence,
                "normalized_entropy": r.normalized_entropy,
                "is_boundary_customer": r.is_boundary_customer,
                "pca_1": r.pca_1,
                "pca_2": r.pca_2,
                "source": r.source,
                "batch_id": r.batch_id,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
        return CustomersPageResponse(total=total, limit=limit, offset=offset, items=items)

    @application.get("/api/v1/drift-reports")
    def list_drift_reports(
        limit: int = Query(default=25, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> dict[str, Any]:
        rows, total = query_drift_reports(db, limit=limit, offset=offset)
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [
                {
                    "id": r.id,
                    "run_id": r.run_id,
                    "batch_size": r.batch_size,
                    "overall_status": r.overall_status,
                    "alert_status": r.alert_status,
                    "max_psi": r.max_psi,
                    "max_ks_statistic": r.max_ks_statistic,
                    "window_label": r.window_label,
                    "alert_dispatched": r.alert_dispatched,
                    "features": json.loads(r.feature_psi_json),
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ],
        }

    @application.get("/api/v1/scoring-events")
    def list_scoring_events(
        limit: int = Query(default=25, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        db: Session = Depends(get_db),
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> dict[str, Any]:
        rows, total = query_scoring_events(db, limit=limit, offset=offset)
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": [
                {
                    "id": r.id,
                    "batch_id": r.batch_id,
                    "gender": r.gender,
                    "age": r.age,
                    "annual_income_k": r.annual_income_k,
                    "spending_score": r.spending_score,
                    "cluster_id": r.cluster_id,
                    "persona_name": r.persona_name,
                    "recommended_action": r.recommended_action,
                    "gmm_confidence": r.gmm_confidence,
                    "normalized_entropy": r.normalized_entropy,
                    "is_boundary_customer": r.is_boundary_customer,
                    "distance_to_centroid": r.distance_to_centroid,
                    "drift_flag": r.drift_flag,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ],
        }

    allowed_charts = {
        "01_eda_distributions.png": "image/png",
        "02_kmeans_sweep.png": "image/png",
        "03_ward_dendrogram.png": "image/png",
        "04_pca_clusters.png": "image/png",
        "05_segments_3d.html": "text/html; charset=utf-8",
        "06_age_income_spending_matrix.png": "image/png",
        "07_algorithm_consensus_ari.png": "image/png",
    }

    @application.get("/api/v1/visuals/{chart_name}")
    def get_visual_artifact(
        chart_name: str,
        _auth: AuthContext = Depends(require_roles("admin", "analyst", "viewer")),
    ) -> FileResponse:
        media_type = allowed_charts.get(chart_name)
        if media_type is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Requested visual artifact is not recognized.",
            )
        base_dir = Path(cfg.artifacts.visuals_dir).resolve()
        target = (base_dir / chart_name).resolve()
        if not target.is_relative_to(base_dir) or not target.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Visual artifact file not generated yet.",
            )
        return FileResponse(target, media_type=media_type)

    return application


app = create_app()
