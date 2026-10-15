"""
app/governance/registry.py
Multi-Model Registry for RailGov MVP.
Maintains versioned JSON catalogs of models with promotion, rejection,
and rollback capabilities at models/registry.json.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import REGISTRY_PATH


def _get_registry_path(custom_path: Optional[Path] = None) -> Path:
    return Path(custom_path) if custom_path else REGISTRY_PATH


def _normalize_registry(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure multi-model dictionary structure, converting legacy flat JSON if needed."""
    if "models" not in data:
        return {"models": {"ticket_predictor": data}}
    return data


def load_registry(custom_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load model registry from JSON; initialize default multi-model structure if absent."""
    reg_path = _get_registry_path(custom_path)
    if not reg_path.exists():
        default_registry = {"models": {}}
        save_registry(default_registry, custom_path=custom_path)
        return default_registry

    try:
        with open(reg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return _normalize_registry(data)
    except json.JSONDecodeError:
        return {"models": {}}


def save_registry(data: Dict[str, Any], custom_path: Optional[Path] = None) -> None:
    """Save model registry to disk atomically."""
    reg_path = _get_registry_path(custom_path)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _get_model_catalog(reg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Get or initialize catalog entry for a specific model."""
    if "models" not in reg:
        reg["models"] = {}
    if model_name not in reg["models"]:
        reg["models"][model_name] = {
            "active_version": None,
            "previous_active_version": None,
            "versions": [],
        }
    return reg["models"][model_name]


def register(
    version: str,
    metrics: Dict[str, Any],
    training_data_desc: str,
    status: str = "archived",
    model_name: str = "ticket_predictor",
    custom_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Register a newly trained model version in the catalog."""
    reg = load_registry(custom_path)
    catalog = _get_model_catalog(reg, model_name)

    entry = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "training_data_desc": training_data_desc,
        "status": status,
    }

    # Update or append version entry
    idx = next((i for i, v in enumerate(catalog["versions"]) if v["version"] == version), None)
    if idx is not None:
        catalog["versions"][idx] = entry
    else:
        catalog["versions"].append(entry)

    if status == "active":
        if catalog["active_version"] and catalog["active_version"] != version:
            catalog["previous_active_version"] = catalog["active_version"]
            for v in catalog["versions"]:
                if v["version"] == catalog["previous_active_version"]:
                    v["status"] = "archived"
        catalog["active_version"] = version

    save_registry(reg, custom_path=custom_path)
    return entry


def get_active(
    model_name: str = "ticket_predictor", custom_path: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    """Return the active model record for the specified model_name."""
    reg = load_registry(custom_path)
    catalog = _get_model_catalog(reg, model_name)
    active_ver = catalog.get("active_version")
    if not active_ver:
        return None
    for v in catalog.get("versions", []):
        if v["version"] == active_ver:
            return v
    return None


def promote(
    version: str,
    model_name: str = "ticket_predictor",
    custom_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Promote a model version to active status and archive the previous active."""
    reg = load_registry(custom_path)
    catalog = _get_model_catalog(reg, model_name)

    target = next((v for v in catalog["versions"] if v["version"] == version), None)
    if not target:
        raise ValueError(f"Version '{version}' not found for model '{model_name}'.")

    current_active = catalog.get("active_version")
    if current_active != version:
        catalog["previous_active_version"] = current_active
        for v in catalog["versions"]:
            if v["version"] == current_active:
                v["status"] = "archived"

    target["status"] = "active"
    catalog["active_version"] = version

    save_registry(reg, custom_path=custom_path)
    return catalog


def rollback(
    model_name: str = "ticket_predictor", custom_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Roll back to the previous active model version."""
    reg = load_registry(custom_path)
    catalog = _get_model_catalog(reg, model_name)

    prev_ver = catalog.get("previous_active_version")
    if not prev_ver:
        raise RuntimeError(f"No previous active model version recorded for '{model_name}'.")

    current_active = catalog.get("active_version")
    for v in catalog["versions"]:
        if v["version"] == current_active:
            v["status"] = "archived"
        if v["version"] == prev_ver:
            v["status"] = "active"

    catalog["active_version"] = prev_ver
    catalog["previous_active_version"] = current_active

    save_registry(reg, custom_path=custom_path)
    return catalog


def list_versions(
    model_name: str = "ticket_predictor", custom_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Return list of all registered versions for a model."""
    reg = load_registry(custom_path)
    catalog = _get_model_catalog(reg, model_name)
    return catalog.get("versions", [])


def list_models(custom_path: Optional[Path] = None) -> List[str]:
    """Return all model names present in the registry."""
    reg = load_registry(custom_path)
    return list(reg.get("models", {}).keys())
