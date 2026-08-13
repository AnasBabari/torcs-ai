"""Lazy compatibility handles for the legacy training API.

Historically importing this module constructed a model and could immediately
train it or write a checkpoint.  The handles below preserve the old attribute
names while deferring construction until a caller explicitly uses them.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class LazyInstance(Generic[T]):
    """Create one object on first use, never during module import."""

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._instance: Optional[T] = None

    def _get(self) -> T:
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)

    def __repr__(self) -> str:
        state = "initialized" if self._instance is not None else "deferred"
        return f"<LazyInstance {state}>"


def _new_model() -> Any:
    from .ml_models import MLRacingAI

    return MLRacingAI()


def _new_visualizer() -> Any:
    from .visualization import RacingVisualizer

    return RacingVisualizer()


ml_racing_ai: LazyInstance[Any] = LazyInstance(_new_model)
visualizer: LazyInstance[Any] = LazyInstance(_new_visualizer)
