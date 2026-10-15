"""
app/governance/registry.py
Model Registry for RailGov MVP.
Maintains a versioned JSON catalog of models with promotion, rejection,
and rollback capabilities at models/registry.json.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import REGISTRY_PATH


def _get_registry_path(custom_path: Optional[Path] = None) -> Path:
    return Path(custom_path) if custom_path else REGISTRY_PATH


def load_registry(custom_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load model registry from JSON; initialize default structure if absent."""
    reg_path = _get_registry_path(custom_path)
    if not reg_path.exists():
        default_registry = {
            "active_version": None,
            "previous_active_version": None,
            "versions": [],
        }
        save_registry(default_registry, custom_path=custom_path)
        return default_registry

    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"active_version": None, "previous_active_version": None, "versions": []}


def save_registry(data: Dict[str, Any], custom_path: Optional[Path] = None) -> None:
    """Save model registry to disk atomically."""
    reg_path = _get_registry_path(custom_path)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def register(
    version: str,
    metrics: Dict[str, Any],
    training_data_desc: str,
    status: str = "archived",
    custom_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Register a newly trained model version in the registry.

    Args:
        version: Version identifier (e.g. 'v1', 'v2').
        metrics: Evaluation metrics dictionary.
        training_data_desc: Description of datasets utilized for training.
        status: Lifecycle status ('active', 'archived', 'rejected').
        custom_path: Optional custom registry path.

    Returns:
        Registered version dictionary.
    """
    reg = load_registry(custom_path)
    entry = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "training_data_desc": training_data_desc,
        "status": status,
    }

    # If updating an existing version entry
    existing_idx = next(
        (i for i, v in enumerate(reg["versions"]) if v["version"] == version), None
    )
    if existing_idx is not None:
        reg["versions"][existing_idx] = entry
    else:
        reg["versions"].append(entry)

    if status == "active":
        if reg["active_version"] and reg["active_version"] != version:
            reg["previous_active_version"] = reg["active_version"]
            # Set older active version to archived
            for v in reg["versions"]:
                if v["version"] == reg["previous_active_version"]:
                    v["status"] = "archived"
        reg["active_version"] = version

    save_registry(reg, custom_path=custom_path)
    return entry


def get_active(custom_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return the currently active model record, or None if no active model exists."""
    reg = load_registry(custom_path)
    active_ver = reg.get("active_version")
    if not active_ver:
        return None
    for v in reg.get("versions", []):
        if v["version"] == active_ver:
            return v
    return None


def promote(version: str, custom_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Promote a model version to active status and archive the previously active version.

    Args:
        version: Target version to activate.

    Returns:
        Updated registry dictionary.
    """
    reg = load_registry(custom_path)
    target = next((v for v in reg["versions"] if v["version"] == version), None)
    if not target:
        raise ValueError(f"Version '{version}' not found in registry.")

    current_active = reg.get("active_version")
    if current_active != version:
        reg["previous_active_version"] = current_active
        for v in reg["versions"]:
            if v["version"] == current_active:
                v["status"] = "archived"

    target["status"] = "active"
    reg["active_version"] = version

    save_registry(reg, custom_path=custom_path)
    return reg


def rollback(custom_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Rollback to the previous active model version.

    Returns:
        Updated registry dictionary with rollback applied.
    """
    reg = load_registry(custom_path)
    prev_ver = reg.get("previous_active_version")
    if not prev_ver:
        raise RuntimeError("No previous active model version recorded to roll back to.")

    current_active = reg.get("active_version")

    # Set current active to archived
    for v in reg["versions"]:
        if v["version"] == current_active:
            v["status"] = "archived"
        if v["version"] == prev_ver:
            v["status"] = "active"

    reg["active_version"] = prev_ver
    # Clear or swap previous active
    reg["previous_active_version"] = current_active

    save_registry(reg, custom_path=custom_path)
    return reg


def list_versions(custom_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return list of all registered models."""
    reg = load_registry(custom_path)
    return reg.get("versions", [])
