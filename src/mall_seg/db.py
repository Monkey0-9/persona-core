"""SQLAlchemy 2.0 persistence models and repository helpers (SQLite & PostgreSQL)."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class UserAccount(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(24), nullable=False, default="analyst")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class RefreshTokenRecord(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_schema_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    parent_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    dataset_source: Mapped[str] = mapped_column(String(32), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    target_k: Mapped[int] = mapped_column(Integer, nullable=False)
    include_gender: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    silhouette_score: Mapped[float] = mapped_column(Float, nullable=False)
    mean_jaccard_stability: Mapped[float] = mapped_column(Float, nullable=False)
    passed_stability_gate: Mapped[bool] = mapped_column(Boolean, nullable=False)
    surrogate_fidelity: Mapped[float] = mapped_column(Float, nullable=False)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class ScoredCustomer(Base):
    __tablename__ = "scored_customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    customer_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    annual_income_k: Mapped[float] = mapped_column(Float, nullable=False)
    spending_score: Mapped[float] = mapped_column(Float, nullable=False)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    persona_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    recommended_action: Mapped[str] = mapped_column(Text, nullable=False)
    gmm_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    normalized_entropy: Mapped[float] = mapped_column(Float, nullable=False)
    is_boundary_customer: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    pca_1: Mapped[float] = mapped_column(Float, nullable=False)
    pca_2: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(24), nullable=False, default="training")
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class ScoringEvent(Base):
    __tablename__ = "scoring_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    model_version_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    gender: Mapped[str] = mapped_column(String(16), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    annual_income_k: Mapped[float] = mapped_column(Float, nullable=False)
    spending_score: Mapped[float] = mapped_column(Float, nullable=False)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    persona_name: Mapped[str] = mapped_column(String(64), nullable=False)
    recommended_action: Mapped[str] = mapped_column(Text, nullable=False)
    gmm_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    normalized_entropy: Mapped[float] = mapped_column(Float, nullable=False)
    is_boundary_customer: Mapped[bool] = mapped_column(Boolean, nullable=False)
    distance_to_centroid: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    drift_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class DriftReportRecord(Base):
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("pipeline_runs.id", ondelete="SET NULL"), nullable=True
    )
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    overall_status: Mapped[str] = mapped_column(String(24), nullable=False)
    alert_status: Mapped[str] = mapped_column(String(24), nullable=False, default="NORMAL")
    max_psi: Mapped[float] = mapped_column(Float, nullable=False)
    max_ks_statistic: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    window_label: Mapped[str] = mapped_column(String(64), nullable=False, default="batch")
    alert_dispatched: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    feature_psi_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


def resolve_database_url(database_url: str | None = None) -> str:
    return database_url or os.getenv("DATABASE_URL") or "sqlite:///artifacts/mall_seg.db"


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create SQLAlchemy engine supporting SQLite (with dir creation) and PostgreSQL."""
    url = resolve_database_url(database_url)
    if url.startswith("sqlite:///"):
        db_file = Path(url.removeprefix("sqlite:///"))
        if db_file.parent and str(db_file.parent) != ".":
            db_file.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(url, connect_args={"check_same_thread": False}, future=True)
    return create_engine(url, pool_pre_ping=True, future=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)
    # Apply safe additive column upgrades for pre-existing local SQLite databases
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    additive_ddls: list[str] = []

    if "model_versions" in tables:
        mv_cols = {col["name"] for col in inspector.get_columns("model_versions")}
        if "status" not in mv_cols:
            additive_ddls.append(
                "ALTER TABLE model_versions ADD COLUMN status VARCHAR(32) DEFAULT 'active'"
            )
        if "parent_version" not in mv_cols:
            additive_ddls.append(
                "ALTER TABLE model_versions ADD COLUMN parent_version VARCHAR(64)"
            )
        if "notes" not in mv_cols:
            additive_ddls.append("ALTER TABLE model_versions ADD COLUMN notes TEXT")
        if "promoted_at" not in mv_cols:
            additive_ddls.append("ALTER TABLE model_versions ADD COLUMN promoted_at DATETIME")

    if "scored_customers" in tables:
        sc_cols = {col["name"] for col in inspector.get_columns("scored_customers")}
        if "batch_id" not in sc_cols:
            additive_ddls.append("ALTER TABLE scored_customers ADD COLUMN batch_id VARCHAR(64)")

    if "scoring_events" in tables:
        existing_cols = {col["name"] for col in inspector.get_columns("scoring_events")}
        if "user_id" not in existing_cols:
            additive_ddls.append("ALTER TABLE scoring_events ADD COLUMN user_id INTEGER")
        if "model_version_id" not in existing_cols:
            additive_ddls.append("ALTER TABLE scoring_events ADD COLUMN model_version_id INTEGER")
        if "batch_id" not in existing_cols:
            additive_ddls.append("ALTER TABLE scoring_events ADD COLUMN batch_id VARCHAR(64)")
        if "distance_to_centroid" not in existing_cols:
            additive_ddls.append(
                "ALTER TABLE scoring_events ADD COLUMN distance_to_centroid FLOAT DEFAULT 0.0"
            )
        if "drift_flag" not in existing_cols:
            additive_ddls.append(
                "ALTER TABLE scoring_events ADD COLUMN drift_flag BOOLEAN DEFAULT 0"
            )

    if "drift_reports" in tables:
        dr_cols = {col["name"] for col in inspector.get_columns("drift_reports")}
        if "max_ks_statistic" not in dr_cols:
            additive_ddls.append(
                "ALTER TABLE drift_reports ADD COLUMN max_ks_statistic FLOAT DEFAULT 0.0"
            )
        if "window_label" not in dr_cols:
            additive_ddls.append(
                "ALTER TABLE drift_reports ADD COLUMN window_label VARCHAR(64) DEFAULT 'batch'"
            )
        if "alert_status" not in dr_cols:
            additive_ddls.append(
                "ALTER TABLE drift_reports ADD COLUMN alert_status VARCHAR(24) DEFAULT 'NORMAL'"
            )
        if "alert_dispatched" not in dr_cols:
            additive_ddls.append(
                "ALTER TABLE drift_reports ADD COLUMN alert_dispatched BOOLEAN DEFAULT 0"
            )

    if additive_ddls:
        with engine.begin() as conn:
            for ddl in additive_ddls:
                conn.execute(text(ddl))


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_db_connectivity(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def count_users(session: Session) -> int:
    return int(session.scalar(select(func.count()).select_from(UserAccount)) or 0)


def get_user_by_email(session: Session, email: str) -> UserAccount | None:
    return session.scalar(select(UserAccount).where(UserAccount.email == email.lower().strip()))


def get_user_by_id(session: Session, user_id: int) -> UserAccount | None:
    return session.scalar(select(UserAccount).where(UserAccount.id == user_id))


def create_user(
    session: Session,
    *,
    email: str,
    password_hash: str,
    role: str = "analyst",
) -> UserAccount:
    user = UserAccount(
        email=email.lower().strip(),
        password_hash=password_hash,
        role=role,
    )
    session.add(user)
    session.flush()
    return user


def store_refresh_token(
    session: Session,
    *,
    user_id: int,
    token_hash: str,
    expires_at: datetime,
) -> RefreshTokenRecord:
    rec = RefreshTokenRecord(user_id=user_id, token_hash=token_hash, expires_at=expires_at)
    session.add(rec)
    session.flush()
    return rec


def get_valid_refresh_token(session: Session, token_hash: str) -> RefreshTokenRecord | None:
    rec = session.scalar(
        select(RefreshTokenRecord).where(
            RefreshTokenRecord.token_hash == token_hash,
            RefreshTokenRecord.revoked.is_(False),
        )
    )
    if rec is None:
        return None
    expires = rec.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    if expires <= datetime.now(UTC):
        return None
    return rec


def revoke_refresh_token(session: Session, token_hash: str) -> None:
    rec = session.scalar(
        select(RefreshTokenRecord).where(RefreshTokenRecord.token_hash == token_hash)
    )
    if rec is not None:
        rec.revoked = True
        session.flush()


def upsert_model_version(
    session: Session,
    *,
    version: str,
    algorithm: str,
    feature_schema_hash: str,
    metrics: dict[str, Any],
    artifact_path: str,
    activate: bool = True,
    status_label: str | None = None,
    parent_version: str | None = None,
    notes: str | None = None,
) -> ModelVersionRecord:
    now = datetime.now(UTC)
    if activate:
        existing_active = session.scalars(
            select(ModelVersionRecord).where(ModelVersionRecord.is_active.is_(True))
        ).all()
        for item in existing_active:
            if item.version != version:
                item.is_active = False
                if item.status == "active":
                    item.status = "archived"

    resolved_status = status_label or ("active" if activate else "candidate")
    rec = session.scalar(
        select(ModelVersionRecord).where(ModelVersionRecord.version == version)
    )
    serialized = json.dumps(metrics)
    if rec is None:
        rec = ModelVersionRecord(
            version=version,
            algorithm=algorithm,
            feature_schema_hash=feature_schema_hash,
            metrics_json=serialized,
            artifact_path=artifact_path,
            is_active=activate,
            status=resolved_status,
            parent_version=parent_version,
            notes=notes,
            promoted_at=now if activate else None,
        )
        session.add(rec)
    else:
        rec.algorithm = algorithm
        rec.feature_schema_hash = feature_schema_hash
        rec.metrics_json = serialized
        rec.artifact_path = artifact_path
        rec.is_active = activate
        rec.status = resolved_status
        if parent_version is not None:
            rec.parent_version = parent_version
        if notes is not None:
            rec.notes = notes
        if activate:
            rec.promoted_at = now
        rec.created_at = now
    session.flush()
    return rec


def get_active_model_version(session: Session) -> ModelVersionRecord | None:
    return session.scalar(
        select(ModelVersionRecord)
        .where(ModelVersionRecord.is_active.is_(True))
        .order_by(ModelVersionRecord.created_at.desc(), ModelVersionRecord.id.desc())
        .limit(1)
    )


def get_model_version(session: Session, version: str) -> ModelVersionRecord | None:
    return session.scalar(
        select(ModelVersionRecord).where(ModelVersionRecord.version == version)
    )


def list_model_versions(session: Session) -> list[ModelVersionRecord]:
    return list(
        session.scalars(
            select(ModelVersionRecord).order_by(
                ModelVersionRecord.is_active.desc(),
                ModelVersionRecord.created_at.desc(),
                ModelVersionRecord.id.desc(),
            )
        ).all()
    )


def set_active_model_version(
    session: Session,
    *,
    target_version: str,
    mark_previous_status: str = "archived",
) -> ModelVersionRecord | None:
    target = get_model_version(session, target_version)
    if target is None:
        return None
    now = datetime.now(UTC)
    prev_active = session.scalars(
        select(ModelVersionRecord).where(ModelVersionRecord.is_active.is_(True))
    ).all()
    for item in prev_active:
        if item.version != target_version:
            item.is_active = False
            item.status = mark_previous_status
    target.is_active = True
    target.status = "active"
    target.promoted_at = now
    session.flush()
    return target


def upsert_pipeline_run(
    session: Session,
    *,
    run_key: str,
    dataset_source: str,
    row_count: int,
    target_k: int,
    include_gender: bool,
    silhouette_score: float,
    mean_jaccard_stability: float,
    passed_stability_gate: bool,
    surrogate_fidelity: float,
    summary_payload: dict[str, Any],
    scored_rows: list[dict[str, Any]],
) -> PipelineRun:
    """Idempotently persist pipeline run metadata and its training customer assignments."""
    existing = session.scalar(select(PipelineRun).where(PipelineRun.run_key == run_key))
    serialized_summary = json.dumps(summary_payload)

    if existing is None:
        run_row = PipelineRun(
            run_key=run_key,
            dataset_source=dataset_source,
            row_count=row_count,
            target_k=target_k,
            include_gender=include_gender,
            silhouette_score=silhouette_score,
            mean_jaccard_stability=mean_jaccard_stability,
            passed_stability_gate=passed_stability_gate,
            surrogate_fidelity=surrogate_fidelity,
            summary_json=serialized_summary,
        )
        session.add(run_row)
        session.flush()
    else:
        run_row = existing
        run_row.dataset_source = dataset_source
        run_row.row_count = row_count
        run_row.target_k = target_k
        run_row.include_gender = include_gender
        run_row.silhouette_score = silhouette_score
        run_row.mean_jaccard_stability = mean_jaccard_stability
        run_row.passed_stability_gate = passed_stability_gate
        run_row.surrogate_fidelity = surrogate_fidelity
        run_row.summary_json = serialized_summary
        run_row.created_at = datetime.now(UTC)
        session.execute(
            delete(ScoredCustomer).where(
                ScoredCustomer.run_id == run_row.id,
                ScoredCustomer.source == "training",
            )
        )
        session.flush()

    customer_objects = [
        ScoredCustomer(
            run_id=run_row.id,
            customer_ref=row.get("customer_ref"),
            gender=str(row["gender"]),
            age=int(row["age"]),
            annual_income_k=float(row["annual_income_k"]),
            spending_score=float(row["spending_score"]),
            cluster_id=int(row["cluster_id"]),
            persona_name=str(row["persona_name"]),
            recommended_action=str(row["recommended_action"]),
            gmm_confidence=float(row["gmm_confidence"]),
            normalized_entropy=float(row["normalized_entropy"]),
            is_boundary_customer=bool(row["is_boundary_customer"]),
            pca_1=float(row["pca_1"]),
            pca_2=float(row["pca_2"]),
            source="training",
        )
        for row in scored_rows
    ]
    session.add_all(customer_objects)
    session.flush()
    return run_row


def record_scoring_event(
    session: Session,
    *,
    run_id: int | None,
    prediction: dict[str, Any],
    user_id: int | None = None,
    model_version_id: int | None = None,
    batch_id: str | None = None,
) -> ScoringEvent:
    event = ScoringEvent(
        run_id=run_id,
        user_id=user_id,
        model_version_id=model_version_id,
        batch_id=batch_id,
        gender=str(prediction["gender"]),
        age=int(prediction["age"]),
        annual_income_k=float(prediction["annual_income_k"]),
        spending_score=float(prediction["spending_score"]),
        cluster_id=int(prediction["cluster_id"]),
        persona_name=str(prediction["persona_name"]),
        recommended_action=str(prediction["recommended_action"]),
        gmm_confidence=float(prediction["gmm_confidence"]),
        normalized_entropy=float(prediction["normalized_entropy"]),
        is_boundary_customer=bool(prediction["is_boundary_customer"]),
        distance_to_centroid=float(prediction.get("distance_to_centroid", 0.0)),
        drift_flag=bool(prediction.get("drift_flag", False)),
    )
    session.add(event)
    session.flush()
    return event


def record_drift_report(
    session: Session,
    *,
    run_id: int | None,
    drift_report: dict[str, Any],
    window_label: str = "batch",
    alert_dispatched: bool = False,
) -> DriftReportRecord:
    drift_rec = DriftReportRecord(
        run_id=run_id,
        batch_size=int(drift_report["sample_size"]),
        overall_status=str(drift_report["overall_status"]),
        alert_status=str(drift_report.get("alert_status", "NORMAL")),
        max_psi=float(drift_report["max_psi"]),
        max_ks_statistic=float(drift_report.get("max_ks_statistic", 0.0)),
        window_label=window_label,
        alert_dispatched=alert_dispatched,
        feature_psi_json=json.dumps(drift_report["features"]),
    )
    session.add(drift_rec)
    session.flush()
    return drift_rec


def record_batch_scores_and_drift(
    session: Session,
    *,
    run_id: int,
    scored_rows: list[dict[str, Any]],
    drift_report: dict[str, Any],
    batch_id: str | None = None,
    user_id: int | None = None,
    model_version_id: int | None = None,
    alert_dispatched: bool = False,
) -> DriftReportRecord:
    batch_objects = [
        ScoredCustomer(
            run_id=run_id,
            customer_ref=row.get("customer_ref"),
            gender=str(row["gender"]),
            age=int(row["age"]),
            annual_income_k=float(row["annual_income_k"]),
            spending_score=float(row["spending_score"]),
            cluster_id=int(row["cluster_id"]),
            persona_name=str(row["persona_name"]),
            recommended_action=str(row["recommended_action"]),
            gmm_confidence=float(row["gmm_confidence"]),
            normalized_entropy=float(row["normalized_entropy"]),
            is_boundary_customer=bool(row["is_boundary_customer"]),
            pca_1=float(row["pca_1"]),
            pca_2=float(row["pca_2"]),
            source="batch",
            batch_id=batch_id,
        )
        for row in scored_rows
    ]
    session.add_all(batch_objects)

    event_objects = [
        ScoringEvent(
            run_id=run_id,
            user_id=user_id,
            model_version_id=model_version_id,
            batch_id=batch_id,
            gender=str(row["gender"]),
            age=int(row["age"]),
            annual_income_k=float(row["annual_income_k"]),
            spending_score=float(row["spending_score"]),
            cluster_id=int(row["cluster_id"]),
            persona_name=str(row["persona_name"]),
            recommended_action=str(row["recommended_action"]),
            gmm_confidence=float(row["gmm_confidence"]),
            normalized_entropy=float(row["normalized_entropy"]),
            is_boundary_customer=bool(row["is_boundary_customer"]),
            distance_to_centroid=float(row.get("distance_to_centroid", 0.0)),
            drift_flag=bool(row.get("drift_flag", False)),
        )
        for row in scored_rows
    ]
    session.add_all(event_objects)

    return record_drift_report(
        session,
        run_id=run_id,
        drift_report=drift_report,
        window_label=f"batch:{batch_id}" if batch_id else "batch",
        alert_dispatched=alert_dispatched,
    )


def get_latest_pipeline_run(session: Session) -> PipelineRun | None:
    stmt = (
        select(PipelineRun)
        .order_by(PipelineRun.created_at.desc(), PipelineRun.id.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def query_scored_customers(
    session: Session,
    *,
    run_id: int | None = None,
    segment: str | None = None,
    boundary_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[ScoredCustomer], int]:
    """Query paginated scored customers for the active run with optional filters."""
    resolved_run_id = run_id
    if resolved_run_id is None:
        latest_run = get_latest_pipeline_run(session)
        if latest_run is not None:
            resolved_run_id = latest_run.id

    filters = []
    if resolved_run_id is not None:
        filters.append(ScoredCustomer.run_id == resolved_run_id)
    if segment:
        filters.append(ScoredCustomer.persona_name == segment)
    if boundary_only:
        filters.append(ScoredCustomer.is_boundary_customer.is_(True))

    count_stmt = select(func.count()).select_from(ScoredCustomer)
    data_stmt = select(ScoredCustomer).order_by(ScoredCustomer.id.desc())
    for flt in filters:
        count_stmt = count_stmt.where(flt)
        data_stmt = data_stmt.where(flt)

    total = int(session.scalar(count_stmt) or 0)
    rows = list(session.scalars(data_stmt.offset(offset).limit(limit)).all())
    return rows, total


def query_drift_reports(
    session: Session,
    *,
    limit: int = 25,
    offset: int = 0,
) -> tuple[list[DriftReportRecord], int]:
    """Query paginated batch drift reports ordered newest-first."""
    total = int(session.scalar(select(func.count()).select_from(DriftReportRecord)) or 0)
    rows = list(
        session.scalars(
            select(DriftReportRecord)
            .order_by(DriftReportRecord.id.desc())
            .offset(offset)
            .limit(limit)
        ).all()
    )
    return rows, total


def query_scoring_events(
    session: Session,
    *,
    limit: int = 25,
    offset: int = 0,
) -> tuple[list[ScoringEvent], int]:
    """Query paginated single-customer live scoring events ordered newest-first."""
    total = int(session.scalar(select(func.count()).select_from(ScoringEvent)) or 0)
    rows = list(
        session.scalars(
            select(ScoringEvent)
            .order_by(ScoringEvent.id.desc())
            .offset(offset)
            .limit(limit)
        ).all()
    )
    return rows, total


def query_recent_scoring_events(
    session: Session,
    *,
    window_size: int = 100,
    lookback_days: int | None = None,
) -> list[ScoringEvent]:
    """Fetch recent scoring events for rolling-window drift evaluation."""
    stmt = select(ScoringEvent).order_by(ScoringEvent.id.desc())
    if lookback_days is not None and lookback_days > 0:
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        stmt = stmt.where(ScoringEvent.created_at >= cutoff)
    return list(session.scalars(stmt.limit(window_size)).all())
