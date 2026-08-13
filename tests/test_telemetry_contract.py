"""Tests for the versioned competitive telemetry encoder."""

import numpy as np
import pytest

from torcs_ai.envs.telemetry import (
    OBSERVATION_SIZE,
    TelemetryObservationEncoder,
    TelemetryValidationError,
)


def sensors(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "angle": 0.0,
        "trackPos": 0.0,
        "speedX": 50.0,
        "speedY": 0.0,
        "speedZ": 0.0,
        "rpm": 3_000.0,
        "gear": 2,
        "fuel": 75.0,
        "damage": 0.0,
        "wheelSpinVel": [10.0, 10.0, 10.0, 10.0],
        "track": [100.0] * 19,
        "opponents": [200.0] * 36,
        "racePos": 1,
        "laps": 3,
        "lap": 0,
        "distRaced": 0.0,
        "curLapTime": 0.0,
        "lastLapTime": 0.0,
        "racePhase": 0,
    }
    value.update(overrides)
    return value


def test_encoder_returns_finite_fixed_shape() -> None:
    observation = TelemetryObservationEncoder(competitor_count=6).encode(sensors())
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()


def test_encoder_preserves_negative_sensor_sentinel() -> None:
    observation = TelemetryObservationEncoder().encode(
        sensors(track=[-1.0] + [100.0] * 18, opponents=[-1.0] + [200.0] * 35)
    )
    assert observation[13] == -1.0
    assert observation[32] == -1.0


def test_encoder_uses_damage_delta_and_opponent_closing_rate() -> None:
    encoder = TelemetryObservationEncoder()
    encoder.update(sensors(damage=100.0, opponents=[150.0] * 36), (0.1, 0.2, 0.0))
    observation = encoder.encode(sensors(damage=150.0, opponents=[100.0] * 36))
    assert observation[8] == pytest.approx(0.005)
    assert observation[72] == pytest.approx(0.25)


def test_encoder_rejects_wrong_sensor_lengths() -> None:
    with pytest.raises(TelemetryValidationError, match="track"):
        TelemetryObservationEncoder().encode(sensors(track=[1.0]))


def test_encoder_rejects_non_finite_values() -> None:
    with pytest.raises(TelemetryValidationError, match="speedX"):
        TelemetryObservationEncoder().encode(sensors(speedX=float("nan")))
