"""
app/main.py
FastAPI Production Gateway for RailGov MVP.
Exposes passenger prediction endpoints and multi-model governance controls.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
import pandas as pd
from pydantic import BaseModel, Field

from app.base import get_service, list_services
import app.modules.ticket_predictor  # Ensure ticket predictor registers
from app.governance.audit import get_events, init_audit_db, log_event
from app.governance.drift import detect_drift
from app.governance.registry import get_active, list_models, list_versions, register, rollback
from app.supervisor import run_supervisor_cycle


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize audit DB and bootstrap v1 active model for ticket predictor if missing."""
    init_audit_db()
    service = get_service("ticket_predictor")
    active = get_active("ticket_predictor")
    if not active:
        baseline_df = service.load_baseline_df()
        train_res = service.train(baseline_df, version="v1")
        register(
            version="v1",
            metrics=train_res["metrics"],
            training_data_desc="Automatic bootstrap on startup from baseline.csv",
            status="active",
            model_name="ticket_predictor",
        )
        log_event(
            event_type="MODEL_PROMOTED",
            actor="system_startup",
            details={"version": "v1", "metrics": train_res["metrics"], "reason": "Initial bootstrap"},
            model_version="v1",
            model_name="ticket_predictor",
        )
    yield


app = FastAPI(
    title="RailGov MVP API",
    description="Autonomous Multi-Model AI Governance Layer for Indian Railways",
    version="1.0.0",
    lifespan=lifespan,
)


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


@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(request: TicketPredictionRequest):
    """Predict ticket confirmation with data-quality guardrail and SHAP explainability."""
    service = get_service("ticket_predictor")
    record_dict = request.model_dump()

    # 1. Pre-inference guardrail
    is_valid, issues = service.validate_record(record_dict)
    if not is_valid:
        log_event(
            event_type="DATA_QUALITY_BLOCK",
            actor="api_gateway",
            details={"input": record_dict, "issues": issues},
            model_name=service.model_name,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Data quality validation failed", "issues": issues},
        )

    # 2. Retrieve active model
    active = get_active(service.model_name)
    if not active:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No active model found in registry for {service.model_name}.",
        )

    active_version = active["version"]
    bundle, _ = service.load_model(active_version)

    # 3. Predict & Explain
    prediction_result = service.predict(record_dict, bundle)
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
        model_name=service.model_name,
    )

    return prediction_result


@app.post("/governance/check-batch")
def check_batch(
    payload: BatchCheckRequest,
    model_name: str = Query("ticket_predictor", description="Target model name"),
):
    """Run data quality, drift detection, and performance evaluation on an ingested batch."""
    if not payload.records:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Records list cannot be empty.")

    service = get_service(model_name)
    batch_df = pd.DataFrame(payload.records)

    is_valid, issues = service.validate_batch(batch_df)
    baseline_df = service.load_baseline_df()
    drift_res = detect_drift(
        baseline_df,
        batch_df,
        feature_columns=service.feature_columns,
        categorical_features=service.categorical_features,
        numeric_features=service.numeric_features,
    )

    perf_res = None
    if service.target_column in batch_df.columns:
        active = get_active(model_name)
        if active:
            bundle, meta = service.load_model(active["version"])
            baseline_auc = meta.get("metrics", {}).get("auc", 0.85)
            perf_res = service.evaluate_performance(bundle, batch_df, baseline_metric=baseline_auc)

    return {
        "model_name": model_name,
        "sample_count": len(batch_df),
        "data_quality": {"valid": is_valid, "issues": issues},
        "drift": drift_res,
        "performance": perf_res,
    }


@app.post("/governance/run-supervisor")
def run_supervisor(
    payload: Optional[SupervisorRunRequest] = None,
    model_name: str = Query("ticket_predictor", description="Target model name"),
):
    """Trigger an autonomous supervisor governance cycle for a model."""
    service = get_service(model_name)
    if payload and payload.records:
        batch_df = pd.DataFrame(payload.records)
    else:
        from app.config import DATA_DIR
        drifted_path = DATA_DIR / "drifted.csv"
        if not drifted_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Default batch data missing.")
        batch_df = pd.read_csv(drifted_path)

    report = run_supervisor_cycle(batch_df, service=service)
    return report


@app.get("/governance/models")
def get_model_registry(model_name: str = Query("ticket_predictor", description="Target model name")):
    """Retrieve full model version catalog from registry.json."""
    active = get_active(model_name)
    versions = list_versions(model_name)
    return {
        "model_name": model_name,
        "active_version": active["version"] if active else None,
        "total_versions": len(versions),
        "versions": versions,
        "registered_models": list_models(),
    }


@app.post("/governance/rollback")
def rollback_model(model_name: str = Query("ticket_predictor", description="Target model name")):
    """Roll back active model to the previously active version."""
    try:
        updated_catalog = rollback(model_name)
        log_event(
            event_type="ROLLBACK",
            actor="api_admin",
            details={
                "new_active_version": updated_catalog["active_version"],
                "previous_version": updated_catalog["previous_active_version"],
            },
            model_version=updated_catalog["active_version"],
            model_name=model_name,
        )
        return {
            "status": "success",
            "message": f"Successfully rolled back {model_name} to {updated_catalog['active_version']}",
            "catalog": updated_catalog,
        }
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.get("/governance/audit")
def get_audit_trail(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    limit: int = Query(100, ge=1, le=500, description="Max records to return"),
):
    """Query immutable audit events from SQLite."""
    events = get_events(event_type=event_type, model_name=model_name, limit=limit)
    return {"count": len(events), "filter": event_type, "model_name": model_name, "events": events}


@app.get("/health")
def health_check():
    """System health check and active model telemetry."""
    active = get_active("ticket_predictor")
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_model_version": active["version"] if active else None,
        "active_ticket_model_version": active["version"] if active else None,
        "registered_services": list_services(),
        "audit_database": "connected",
    }
