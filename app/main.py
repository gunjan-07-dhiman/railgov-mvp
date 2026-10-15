"""
app/main.py
FastAPI Production Gateway for RailGov MVP.
Exposes passenger prediction endpoints with data quality guardrails and governance controls:
- POST /predict: Pre-inference quality checks, SHAP explanation, and append-only audit logging.
- POST /governance/check-batch: Batch validation, drift analysis, and performance tracking.
- POST /governance/run-supervisor: Autonomous supervisor execution.
- GET /governance/models: Model registry catalog.
- POST /governance/rollback: Version rollback.
- GET /governance/audit: Filtered audit trail.
- GET /health: Health check and active version telemetry.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
import pandas as pd
from pydantic import BaseModel, Field

from app.config import DATA_DIR, TARGET_COLUMN
from app.governance.audit import get_events, init_audit_db, log_event
from app.governance.data_quality import validate_batch, validate_record
from app.governance.drift import detect_drift
from app.governance.monitor import evaluate_performance
from app.governance.registry import get_active, list_versions, register, rollback
from app.governance.retrain import retrain_if_needed
from app.predictor import load_model, predict, train
from app.supervisor import run_supervisor_cycle


# ---------------------------------------------------------
# Application Lifespan Handler
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup sequence:
    1. Initialize append-only SQLite audit database.
    2. Check model registry: if no active model exists, automatically train v1 on baseline.csv.
    """
    init_audit_db()
    active = get_active()
    if not active:
        baseline_path = DATA_DIR / "baseline.csv"
        if baseline_path.exists():
            baseline_df = pd.read_csv(baseline_path)
            train_res = train(baseline_df, version="v1")
            register(
                version="v1",
                metrics=train_res["metrics"],
                training_data_desc="Automatic bootstrap on startup from baseline.csv",
                status="active",
            )
            log_event(
                event_type="MODEL_PROMOTED",
                actor="system_startup",
                details={
                    "version": "v1",
                    "metrics": train_res["metrics"],
                    "reason": "Initial bootstrap",
                },
                model_version="v1",
            )
    yield


app = FastAPI(
    title="RailGov MVP API",
    description="Autonomous AI Governance Layer for Railway Waitlist Prediction",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------

class TicketPredictionRequest(BaseModel):
    booking_id: Optional[int] = Field(None, json_schema_extra={"example": 100025})
    travel_class: str = Field(..., description="SL, 3A, 2A, 1A, or CC", json_schema_extra={"example": "3A"})
    booking_status: str = Field(..., description="WL or RAC", json_schema_extra={"example": "WL"})
    wl_position: int = Field(..., description="Waitlist / RAC position", json_schema_extra={"example": 35})
    days_before_journey: int = Field(..., description="Days before journey (0-120)", json_schema_extra={"example": 24})
    day_of_week: int = Field(..., description="Day of week (0-6)", json_schema_extra={"example": 3})
    month: int = Field(..., description="Month (1-12)", json_schema_extra={"example": 10})
    is_festival_season: int = Field(..., description="1 if festival season else 0", json_schema_extra={"example": 1})
    route_popularity: float = Field(..., description="Route popularity (0.0-1.0)", json_schema_extra={"example": 0.75})
    train_type: str = Field(..., description="Express, Superfast, or Rajdhani", json_schema_extra={"example": "Superfast"})


class FactorItem(BaseModel):
    feature: str
    value: Any
    shap_value: float
    strength: str
    direction: str
    explanation: str


class PredictionResponse(BaseModel):
    probability: float
    predicted_status: str
    confidence: str
    top_factors: List[FactorItem]
    model_version: str


class BatchCheckRequest(BaseModel):
    records: List[Dict[str, Any]]


class SupervisorRunRequest(BaseModel):
    records: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------
# Prediction Gateway Endpoint
# ---------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(request: TicketPredictionRequest):
    """
    Predict ticket confirmation with data-quality guardrail and SHAP explainability.
    Blocks invalid input with HTTP 422 and logs to the audit database.
    """
    record_dict = request.model_dump()

    # 1. Data Quality Guardrail
    is_valid, issues = validate_record(record_dict)
    if not is_valid:
        log_event(
            event_type="DATA_QUALITY_BLOCK",
            actor="api_gateway",
            details={"input": record_dict, "issues": issues},
            model_version=None,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Data quality validation failed", "issues": issues},
        )

    # 2. Retrieve Active Model
    active = get_active()
    if not active:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No active model found in registry.",
        )

    active_version = active["version"]
    bundle, _ = load_model(active_version)

    # 3. Predict & Explain
    prediction_result = predict(record_dict, bundle)
    prediction_result["model_version"] = active_version

    # 4. Append-Only Audit Trail
    log_event(
        event_type="PREDICTION",
        actor="api_gateway",
        details={
            "booking_id": record_dict.get("booking_id"),
            "probability": prediction_result["probability"],
            "predicted_status": prediction_result["predicted_status"],
            "confidence": prediction_result["confidence"],
            "top_factor": prediction_result["top_factors"][0]["explanation"]
            if prediction_result["top_factors"]
            else None,
        },
        model_version=active_version,
    )

    return prediction_result


# ---------------------------------------------------------
# Governance Endpoints
# ---------------------------------------------------------

@app.post("/governance/check-batch")
def check_batch(payload: BatchCheckRequest):
    """
    Run data quality, drift detection, and performance evaluation on an ingested batch.
    """
    if not payload.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Records list cannot be empty.",
        )

    batch_df = pd.DataFrame(payload.records)

    # 1. Data quality check
    is_valid, issues = validate_batch(batch_df)

    # 2. Statistical drift
    baseline_path = DATA_DIR / "baseline.csv"
    if not baseline_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Baseline reference dataset missing.",
        )
    baseline_df = pd.read_csv(baseline_path)
    drift_res = detect_drift(baseline_df, batch_df)

    # 3. Performance check (if labels present)
    perf_res = None
    if TARGET_COLUMN in batch_df.columns:
        active = get_active()
        if active:
            bundle, meta = load_model(active["version"])
            baseline_auc = meta.get("metrics", {}).get("auc", 0.85)
            perf_res = evaluate_performance(bundle, batch_df, baseline_auc=baseline_auc)

    return {
        "sample_count": len(batch_df),
        "data_quality": {"valid": is_valid, "issues": issues},
        "drift": drift_res,
        "performance": perf_res,
    }


@app.post("/governance/run-supervisor")
def run_supervisor(payload: Optional[SupervisorRunRequest] = None):
    """
    Trigger an autonomous supervisor governance cycle.
    If no records are provided, processes the drifted.csv batch as a demo.
    """
    if payload and payload.records:
        batch_df = pd.DataFrame(payload.records)
    else:
        drifted_path = DATA_DIR / "drifted.csv"
        if not drifted_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Default drifted batch not found.",
            )
        batch_df = pd.read_csv(drifted_path)

    report = run_supervisor_cycle(batch_df)
    return report


@app.get("/governance/models")
def get_model_registry():
    """Retrieve full model version catalog from registry.json."""
    active = get_active()
    versions = list_versions()
    return {
        "active_version": active["version"] if active else None,
        "total_versions": len(versions),
        "versions": versions,
    }


@app.post("/governance/rollback")
def rollback_model():
    """Roll back active model to the previously active version."""
    try:
        updated_reg = rollback()
        log_event(
            event_type="ROLLBACK",
            actor="api_admin",
            details={
                "new_active_version": updated_reg["active_version"],
                "previous_version": updated_reg["previous_active_version"],
            },
            model_version=updated_reg["active_version"],
        )
        return {
            "status": "success",
            "message": f"Successfully rolled back to {updated_reg['active_version']}",
            "registry": updated_reg,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@app.get("/governance/audit")
def get_audit_trail(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(100, ge=1, le=500, description="Max records to return"),
):
    """Query immutable audit events from SQLite."""
    events = get_events(event_type=event_type, limit=limit)
    return {
        "count": len(events),
        "filter": event_type,
        "events": events,
    }


@app.get("/health")
def health_check():
    """System health check and active model telemetry."""
    active = get_active()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_model_version": active["version"] if active else None,
        "audit_database": "connected",
    }
