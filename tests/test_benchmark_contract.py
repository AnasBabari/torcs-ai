"""Pure contract tests for the native benchmark matrix surface."""

from scripts.benchmark_native import _runtime_slug


def test_runtime_slug_is_stable_for_track_labels() -> None:
    assert _runtime_slug("road/alpine-1") == "road-alpine-1"
    assert _runtime_slug("oval/michigan") == "oval-michigan"


def test_runtime_slug_cannot_escape_runtime_root() -> None:
    slug = _runtime_slug("../outside/track")
    assert "/" not in slug
    assert "\\" not in slug
    assert ".." not in slug


def test_runtime_slug_has_default_for_empty_track() -> None:
    assert _runtime_slug(None) == "default"
