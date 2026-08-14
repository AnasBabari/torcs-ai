"""Unit tests for ObservationSchema explicit segments, unpack, and validation."""

from __future__ import annotations

import numpy as np
import pytest

from torcs_ai.envs.telemetry import (
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    ObservationSchema,
    TelemetryObservationEncoder,
    TelemetryValidationError,
)


def test_schema_segments_sum_to_total_size() -> None:
    total_size = sum(segment.size for segment in ObservationSchema.ALL_SEGMENTS)
    assert total_size == OBSERVATION_SIZE == 118
    assert (
        ObservationSchema.VERSION
        == OBSERVATION_SCHEMA_VERSION
        == "competitive-telemetry-v1"
    )


def test_segment_indices_are_contiguous() -> None:
    current_index = 0
    for segment in ObservationSchema.ALL_SEGMENTS:
        assert segment.start_index == current_index
        assert segment.end_index == current_index + segment.size
        current_index = segment.end_index
    assert current_index == OBSERVATION_SIZE


def test_observation_unpack() -> None:
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    obs[0:9] = 0.5  # ego
    obs[9:13] = 0.2  # wheel_spin
    obs[13:32] = 0.8  # track_rays
    obs[32:68] = -1.0  # opponents
    obs[68:71] = 0.0  # controls
    obs[71:72] = 0.1  # normalized_position
    obs[72:108] = 0.0  # closing_rates
    obs[108:111] = 1.0  # traffic_clearance
    obs[111:118] = 0.3  # race_context

    unpacked = ObservationSchema.unpack(obs)
    assert set(unpacked.keys()) == {
        "ego",
        "wheel_spin",
        "track_rays",
        "opponents",
        "controls",
        "normalized_position",
        "closing_rates",
        "traffic_clearance",
        "race_context",
    }
    assert unpacked["ego"].shape == (9,)
    assert unpacked["wheel_spin"].shape == (4,)
    assert unpacked["track_rays"].shape == (19,)
    assert unpacked["opponents"].shape == (36,)
    assert unpacked["controls"].shape == (3,)
    assert unpacked["normalized_position"].shape == (1,)
    assert unpacked["closing_rates"].shape == (36,)
    assert unpacked["traffic_clearance"].shape == (3,)
    assert unpacked["race_context"].shape == (7,)


def test_observation_validation() -> None:
    valid_obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    ObservationSchema.validate(valid_obs)

    # Wrong shape
    with pytest.raises(TelemetryValidationError, match="shape"):
        ObservationSchema.validate(np.zeros(100, dtype=np.float32))

    # Non-finite values
    invalid_nan = valid_obs.copy()
    invalid_nan[10] = np.nan
    with pytest.raises(TelemetryValidationError, match="non-finite"):
        ObservationSchema.validate(invalid_nan)

    # Out of range values
    invalid_range = valid_obs.copy()
    invalid_range[5] = 2.5
    with pytest.raises(TelemetryValidationError, match="range"):
        ObservationSchema.validate(invalid_range)
