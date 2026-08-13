"""Safety shield invariants."""

from torcs_ai.controllers import apply_safety_shield


def test_shield_projects_edge_and_heading_recovery() -> None:
    result = apply_safety_shield(
        {"steer": 0.0, "accel": 1.0, "brake": 0.0, "gear": 3.0},
        {"trackPos": 1.2, "angle": 0.8, "speedX": 30.0},
    )
    assert result.intervened
    assert "edge_recovery" in result.reasons
    assert "heading_recovery" in result.reasons
    assert result.controls["accel"] == 0.0
    assert result.controls["brake"] > 0.0


def test_shield_never_allows_accel_and_brake() -> None:
    result = apply_safety_shield(
        {"steer": 0.0, "accel": 0.7, "brake": 0.6, "gear": 2.0},
        {"trackPos": 0.0, "angle": 0.0, "speedX": 20.0},
    )
    assert not (result.controls["accel"] > 0 and result.controls["brake"] > 0)
