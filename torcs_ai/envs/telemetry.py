"""Versioned, explicit telemetry observation schema and encoder for competitive racing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

OBSERVATION_SCHEMA_VERSION = "competitive-telemetry-v1"
OBSERVATION_SIZE = 118
TRACK_SENSOR_COUNT = 19
OPPONENT_SENSOR_COUNT = 36
WHEEL_SENSOR_COUNT = 4


@dataclass(frozen=True)
class ObservationSegment:
    """Explicit metadata describing one named slice of the telemetry observation vector."""

    name: str
    start_index: int
    end_index: int
    size: int
    min_value: float
    max_value: float
    description: str


class ObservationSchema:
    """Canonical schema specification and segment registry for competitive-telemetry-v1."""

    VERSION: str = OBSERVATION_SCHEMA_VERSION
    SIZE: int = OBSERVATION_SIZE

    # Named segment slice definitions:
    EGO = ObservationSegment(
        name="ego",
        start_index=0,
        end_index=9,
        size=9,
        min_value=-1.0,
        max_value=1.0,
        description="Ego telemetry (angle, trackPos, speedX, speedY, speedZ, rpm, gear, fuel, damage_delta)",
    )
    WHEEL_SPIN = ObservationSegment(
        name="wheel_spin",
        start_index=9,
        end_index=13,
        size=4,
        min_value=-1.0,
        max_value=1.0,
        description="4 wheel angular spin velocities normalized by 300 rad/s",
    )
    TRACK_RAYS = ObservationSegment(
        name="track_rays",
        start_index=13,
        end_index=32,
        size=19,
        min_value=-1.0,
        max_value=1.0,
        description="19 track rangefinder rays spanning -90 deg to +90 deg (normalized by 200m, -1 sentinel off track)",
    )
    OPPONENTS = ObservationSegment(
        name="opponents",
        start_index=32,
        end_index=68,
        size=36,
        min_value=-1.0,
        max_value=1.0,
        description="36 opponent proximity rangefinders covering 360 deg (normalized by 200m, -1 sentinel undetected)",
    )
    CONTROLS = ObservationSegment(
        name="controls",
        start_index=68,
        end_index=71,
        size=3,
        min_value=-1.0,
        max_value=1.0,
        description="Previous applied controls (steer [-1, 1], accel [0, 1], brake [0, 1])",
    )
    NORMALIZED_POSITION = ObservationSegment(
        name="normalized_position",
        start_index=71,
        end_index=72,
        size=1,
        min_value=0.0,
        max_value=1.0,
        description="Current race position normalized to [0, 1] (0 = 1st place / leader)",
    )
    CLOSING_RATES = ObservationSegment(
        name="closing_rates",
        start_index=72,
        end_index=108,
        size=36,
        min_value=-1.0,
        max_value=1.0,
        description="36 opponent closing rates (rate of range decrease between steps, normalized to [-1, 1])",
    )
    TRAFFIC_CLEARANCE = ObservationSegment(
        name="traffic_clearance",
        start_index=108,
        end_index=111,
        size=3,
        min_value=0.0,
        max_value=1.0,
        description="Minimum clearance in left, front, and right sectors (1.0 = clear path)",
    )
    RACE_CONTEXT = ObservationSegment(
        name="race_context",
        start_index=111,
        end_index=118,
        size=7,
        min_value=0.0,
        max_value=1.0,
        description="Race progress context (distance progress, lap fraction, remaining fraction, curLapTime, lastLapTime, competitor count, race phase)",
    )

    ALL_SEGMENTS: tuple[ObservationSegment, ...] = (
        EGO,
        WHEEL_SPIN,
        TRACK_RAYS,
        OPPONENTS,
        CONTROLS,
        NORMALIZED_POSITION,
        CLOSING_RATES,
        TRAFFIC_CLEARANCE,
        RACE_CONTEXT,
    )

    @classmethod
    def unpack(cls, observation: np.ndarray) -> dict[str, np.ndarray]:
        """Unpack a flat 118-value observation vector into named dictionary segments."""
        if observation.shape != (cls.SIZE,):
            raise TelemetryValidationError(
                f"expected observation shape ({cls.SIZE},), got {observation.shape}"
            )
        return {
            segment.name: observation[segment.start_index : segment.end_index]
            for segment in cls.ALL_SEGMENTS
        }

    @classmethod
    def validate(cls, observation: np.ndarray) -> None:
        """Validate observation vector shape, dtype, finiteness, and bounds."""
        if observation.shape != (cls.SIZE,):
            raise TelemetryValidationError(
                f"observation shape must be ({cls.SIZE},), got {observation.shape}"
            )
        if not np.issubdtype(observation.dtype, np.floating):
            raise TelemetryValidationError(
                f"observation dtype must be float, got {observation.dtype}"
            )
        if not np.isfinite(observation).all():
            raise TelemetryValidationError("observation contains non-finite values")
        if (observation < -1.001).any() or (observation > 1.001).any():
            raise TelemetryValidationError(
                "observation values exceed normalized range [-1, 1]"
            )


class TelemetryValidationError(ValueError):
    """Raised when required TORCS telemetry is absent or malformed."""


def _scalar(sensors: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = sensors.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TelemetryValidationError(f"sensor {key!r} is not numeric") from exc
    if not np.isfinite(result):
        raise TelemetryValidationError(f"sensor {key!r} is not finite")
    return result


def _array(sensors: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    value = sensors.get(key)
    if value is None:
        raise TelemetryValidationError(f"missing sensor {key!r}")
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TelemetryValidationError(f"sensor {key!r} is not numeric") from exc
    if result.ndim != 1 or result.size != size:
        raise TelemetryValidationError(
            f"sensor {key!r} must have {size} values, got shape {result.shape}"
        )
    if not np.isfinite(result).all():
        raise TelemetryValidationError(f"sensor {key!r} contains non-finite values")
    return result


def _normalize_range(values: np.ndarray) -> np.ndarray:
    normalized = np.clip(values, 0.0, 200.0) / 200.0
    return np.where(values < 0.0, -1.0, normalized)


def _normalize_clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high) / max(abs(low), abs(high)))


def _closing_rates(current: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is None:
        return np.zeros_like(current)
    return np.asarray(
        np.clip(previous - current, -200.0, 200.0) / 200.0, dtype=np.float32
    )


def _traffic_clearance(opponents: np.ndarray) -> np.ndarray:
    """Return left/front/right clearance while preserving empty-sensor safety."""
    values: list[float] = []
    for section in np.array_split(opponents, 3):
        valid = section[section >= 0.0]
        values.append(float(np.min(valid)) / 200.0 if valid.size else 1.0)
    return np.asarray(values, dtype=np.float32)


@dataclass
class TelemetryObservationEncoder:
    """Encode the canonical 118-field observation without network side effects."""

    competitor_count: int = 1
    track_length: float = 10_000.0
    previous_sensors: Mapping[str, Any] | None = None
    previous_controls: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def encode(self, sensors: Mapping[str, Any]) -> np.ndarray:
        track = _array(sensors, "track", TRACK_SENSOR_COUNT)
        opponents = _array(sensors, "opponents", OPPONENT_SENSOR_COUNT)
        wheel_spin = _array(sensors, "wheelSpinVel", WHEEL_SENSOR_COUNT)
        previous_opponents = None
        previous_damage = 0.0
        if self.previous_sensors is not None:
            previous_opponents = _array(
                self.previous_sensors, "opponents", OPPONENT_SENSOR_COUNT
            )
            previous_damage = _scalar(self.previous_sensors, "damage")

        damage_delta = max(0.0, _scalar(sensors, "damage") - previous_damage)
        race_position = max(1.0, _scalar(sensors, "racePos", 1.0))
        competitor_count = max(1, int(self.competitor_count))
        normalized_position = (race_position - 1.0) / max(competitor_count - 1, 1)
        laps = max(1.0, _scalar(sensors, "laps", 1.0))
        completed_laps = max(0.0, _scalar(sensors, "lap", 0.0))
        lap_fraction = float(np.clip(completed_laps / laps, 0.0, 1.0))
        distance_progress = float(
            np.clip(
                _scalar(sensors, "distRaced") / max(self.track_length, 1.0), 0.0, 1.0
            )
        )
        context = np.asarray(
            [
                distance_progress,
                lap_fraction,
                1.0 - lap_fraction,
                np.clip(_scalar(sensors, "curLapTime") / 3600.0, 0.0, 1.0),
                np.clip(_scalar(sensors, "lastLapTime") / 3600.0, 0.0, 1.0),
                competitor_count / max(competitor_count, 1),
                np.clip(_scalar(sensors, "racePhase") / 3.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        ego = np.asarray(
            [
                np.clip(_scalar(sensors, "angle") / np.pi, -1.0, 1.0),
                np.clip(_scalar(sensors, "trackPos") / 2.0, -1.0, 1.0),
                np.clip(_scalar(sensors, "speedX"), -50.0, 350.0) / 350.0,
                np.clip(_scalar(sensors, "speedY"), -100.0, 100.0) / 100.0,
                np.clip(_scalar(sensors, "speedZ"), -100.0, 100.0) / 100.0,
                np.clip(_scalar(sensors, "rpm"), 0.0, 10_000.0) / 10_000.0,
                np.clip(_scalar(sensors, "gear"), -1.0, 6.0) / 6.0,
                np.clip(_scalar(sensors, "fuel"), 0.0, 100.0) / 100.0,
                np.clip(damage_delta, 0.0, 10_000.0) / 10_000.0,
            ],
            dtype=np.float32,
        )
        controls = np.asarray(self.previous_controls, dtype=np.float32)
        result = np.concatenate(
            [
                ego,
                np.clip(wheel_spin, -300.0, 300.0) / 300.0,
                _normalize_range(track),
                _normalize_range(opponents),
                controls,
                np.asarray([np.clip(normalized_position, 0.0, 1.0)], dtype=np.float32),
                _closing_rates(opponents, previous_opponents),
                _traffic_clearance(opponents),
                context,
            ]
        ).astype(np.float32, copy=False)
        if result.shape != (OBSERVATION_SIZE,) or not np.isfinite(result).all():
            raise TelemetryValidationError(
                f"encoded observation must be finite with shape ({OBSERVATION_SIZE},)"
            )
        return result

    def update(
        self, sensors: Mapping[str, Any], applied_controls: Sequence[float]
    ) -> np.ndarray:
        """Encode and advance the previous-state context after a control step."""
        if len(applied_controls) != 3:
            raise TelemetryValidationError("applied_controls must contain three values")
        encoded = self.encode(sensors)
        self.previous_sensors = dict(sensors)
        self.previous_controls = (
            float(np.clip(applied_controls[0], -1.0, 1.0)),
            float(np.clip(applied_controls[1], -1.0, 1.0)),
            float(np.clip(applied_controls[2], -1.0, 1.0)),
        )
        return encoded
