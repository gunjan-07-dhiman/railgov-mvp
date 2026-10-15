"""
app/base.py
Abstract base class and registry for RailGov model services.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class ModelService(ABC):
    """Abstract interface defining required behaviors for any governed model."""

    model_name: str
    feature_columns: List[str]
    categorical_features: List[str]
    numeric_features: List[str]
    target_column: str

    @abstractmethod
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single inference record."""
        pass

    @abstractmethod
    def validate_batch(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate an ingested batch DataFrame."""
        pass

    @abstractmethod
    def train(self, df: pd.DataFrame, version: str) -> Dict[str, Any]:
        """Train a candidate model version and return metrics/paths."""
        pass

    @abstractmethod
    def load_model(self, version: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model bundle and metadata from storage."""
        pass

    @abstractmethod
    def predict(self, record: Dict[str, Any], bundle: Any) -> Dict[str, Any]:
        """Generate prediction and explanation payload."""
        pass

    @abstractmethod
    def evaluate_performance(
        self, bundle: Any, df: pd.DataFrame, baseline_metric: Optional[float] = None
    ) -> Dict[str, Any]:
        """Evaluate holdout performance against baseline thresholds."""
        pass

    @abstractmethod
    def load_baseline_df(self) -> pd.DataFrame:
        """Load reference baseline dataset for drift comparison."""
        pass


_SERVICES: Dict[str, ModelService] = {}


def register_service(service: ModelService) -> None:
    """Register a model service instance in the global catalog."""
    _SERVICES[service.model_name] = service


def get_service(model_name: str = "ticket_predictor") -> ModelService:
    """Retrieve a registered model service by name."""
    if model_name not in _SERVICES:
        raise KeyError(f"Model service '{model_name}' is not registered.")
    return _SERVICES[model_name]


def list_services() -> List[str]:
    """Return all registered model service names."""
    return list(_SERVICES.keys())
