"""
app/governance/audit.py
Append-only Audit Log for RailGov MVP backed by SQLite.
Enforces non-repudiation using database-level SQLite triggers that ABORT on UPDATE and DELETE.
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
    """
    Initialize audit_log table and strict append-only triggers.
    Any attempt to execute an UPDATE or DELETE statement is aborted by the database engine.
    """
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
                    model_version TEXT
                );
                """
            )

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
    db_path: Optional[Path] = None,
) -> int:
    """
    Append an immutable event to the audit trail.

    Args:
        event_type: Type of governance event (e.g. PREDICTION, DRIFT_ALERT).
        actor: Identity or subsystem generating the event (e.g. "api_gateway", "supervisor").
        details: Payload dictionary stored as JSON.
        model_version: Active model version associated with the event.
        db_path: Optional custom path (useful in isolated tests).

    Returns:
        The generated integer event ID.
    """
    init_audit_db(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()
    details_str = json.dumps(details, default=str)

    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO audit_log (timestamp, event_type, actor, details_json, model_version)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, event_type, actor, details_str, model_version),
            )
            return int(cursor.lastrowid)
    finally:
        conn.close()


def get_events(
    event_type: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve audit events in reverse chronological order.

    Args:
        event_type: Optional filter for a specific event type.
        limit: Maximum number of records to return.
        db_path: Optional custom path.

    Returns:
        List of audit event dictionaries.
    """
    init_audit_db(db_path)
    conn = get_connection(db_path)
    try:
        if event_type:
            cursor = conn.execute(
                """
                SELECT id, timestamp, event_type, actor, details_json, model_version
                FROM audit_log
                WHERE event_type = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (event_type, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, timestamp, event_type, actor, details_json, model_version
                FROM audit_log
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cursor.fetchall()
        results = []
        for r in rows:
            try:
                parsed_details = json.loads(r["details_json"])
            except json.JSONDecodeError:
                parsed_details = r["details_json"]

            results.append(
                {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "event_type": r["event_type"],
                    "actor": r["actor"],
                    "details": parsed_details,
                    "model_version": r["model_version"],
                }
            )
        return results
    finally:
        conn.close()
