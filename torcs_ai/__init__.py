"""TORCS AI package metadata and lazy compatibility exports.

Importing :mod:`torcs_ai` must be safe: it must not instantiate a model,
start a simulator, open a socket, train, or write a checkpoint.  Legacy
training and visualization APIs remain available through lazy attributes so
existing callers can migrate without reintroducing import-time side effects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "2.1.0a1"
__author__ = "Anas Babari"
__description__ = "Research tooling for native TORCS racing agents"

_LAZY_EXPORTS = {
    "Client": (".client", "Client"),
    "MLRacingAI": (".ml_models", "MLRacingAI"),
    "RacingVisualizer": (".visualization", "RacingVisualizer"),
    "automated_training_pipeline": (".training", "automated_training_pipeline"),
    "continuous_learning_mode": (".training", "continuous_learning_mode"),
    "perfection_training_pipeline": (".training", "perfection_training_pipeline"),
    "elite_training_curriculum": (".training", "elite_training_curriculum"),
    "intensive_training_session": (".training", "intensive_training_session"),
    "start_torcs_server": (".utils", "start_torcs_server"),
    "analyze_ml_models": (".utils", "analyze_ml_models"),
    "generate_racing_insights": (".utils", "generate_racing_insights"),
}

__all__ = ["__version__", "__author__", "__description__", *_LAZY_EXPORTS]


def __getattr__(name: str) -> Any:
    """Load a legacy export only when the caller explicitly requests it."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
