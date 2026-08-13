"""Actuator smoothing and longitudinal-exclusion contracts."""

import pytest

from torcs_ai.controllers import apply_slew_limiter


def test_slew_limiter_bounds_steering_change() -> None:
    controls, limited = apply_slew_limiter(
        {"steer": 1.0, "accel": 0.0, "brake": 0.0, "gear": 3.0},
        {"steer": 0.0, "accel": 0.0, "brake": 0.0, "gear": 2.0},
    )
    assert limited
    assert controls["steer"] == pytest.approx(0.15)
    assert controls["gear"] == 3.0


def test_brake_request_cuts_residual_accel() -> None:
    controls, _ = apply_slew_limiter(
        {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": 3.0},
        {"steer": 0.0, "accel": 0.8, "brake": 0.0, "gear": 3.0},
    )
    assert controls["accel"] == 0.0
    assert controls["brake"] == pytest.approx(0.2)
