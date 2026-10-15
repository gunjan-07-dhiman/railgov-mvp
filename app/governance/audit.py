"""
app/governance/audit.py
Append-only Audit Log for RailGov MVP backed by SQLite.
Enforces non-repudiation using database-level SQLite triggers that ABORT on UPDATE and DELETE.
Supports multi-model tracking via model_name column.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
from app.config import AUDIT_DB_PATH

# Validated event types
EVENT_TYPES = {
    "PREDICTION",
    "DATA_QUALITY_BLOCK",
    "DRIFT_ALERT",
    "PERFORMANCE_ALERT",
    "RETRAIN_STARTED",
    "MODEL_PROMOTED",
    "MODEL_REJECTED",
    "ROLLBACK",
    "SUPERVISOR_DECISION",
}


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite database connection with row factory enabled."""
    target_path = Path(db_path) if db_path else AUDIT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_audit_db(db_path: Optional[Path] = None) -> None:
    """Initialize audit_log table, triggers, and ensure model_name column exists."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    model_version TEXT,
                    model_name TEXT DEFAULT 'ticket_predictor'
                );
                """
            )

            # Check if model_name column exists in existing table, add if missing
            cols = [row["name"] for row in conn.execute("PRAGMA table_info(audit_log)").fetchall()]
            if "model_name" not in cols:
                conn.execute("ALTER TABLE audit_log ADD COLUMN model_name TEXT DEFAULT 'ticket_predictor'")

            # Enforce append-only with SQLite triggers
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS prevent_audit_update
                BEFORE UPDATE ON audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'Audit log is strictly append-only: UPDATE operations are forbidden');
                END;
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS prevent_audit_delete
                BEFORE DELETE ON audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'Audit log is strictly append-only: DELETE operations are forbidden');
                END;
                """
            )
    finally:
        conn.close()


def log_event(
    event_type: str,
    actor: str,
    details: Dict[str, Any],
    model_version: Optional[str] = None,
    model_name: str = "ticket_predictor",
    db_path: Optional[Path] = None,
) -> int:
    """Append an immutable event to the audit trail."""
    init_audit_db(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    details_str = json.dumps(details, default=str)

    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO audit_log (timestamp, event_type, actor, details_json, model_version, model_name)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timestamp, event_type, actor, details_str, model_version, model_name),
            )
            return int(cursor.lastrowid)
    finally:
        conn.close()


def get_events(
    event_type: Optional[str] = None,
    model_name: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve audit events in reverse chronological order with optional filtering."""
    init_audit_db(db_path)
    conn = get_connection(db_path)
    try:
        query = "SELECT id, timestamp, event_type, actor, details_json, model_version, model_name FROM audit_log"
        params: List[Any] = []
        conditions: List[str] = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            try:
                parsed = json.loads(r["details_json"])
            except json.JSONDecodeError:
                parsed = r["details_json"]

            results.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "event_type": r["event_type"],
                "actor": r["actor"],
                "details": parsed,
                "model_version": r["model_version"],
                "model_name": r["model_name"] if "model_name" in r.keys() else "ticket_predictor",
            })
        return results
    finally:
        conn.close()
