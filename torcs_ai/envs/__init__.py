"""Environment schemas and Gymnasium adapters."""

from .multi_track import (
    ALL_APPROVED_TRACKS,
    HELD_OUT_TEST_TRACKS,
    TRAINING_TRACKS,
    VALIDATION_TRACKS,
    MultiTrackRacingEnv,
    TrackRoleError,
    validate_track_training_selection,
)
from .racing import RacingEnv, default_low_level_controller
from .telemetry import (
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    ObservationSchema,
    ObservationSegment,
    TelemetryObservationEncoder,
    TelemetryValidationError,
)
from .torcs_transport import ScrTransportConfig, TorcsScrTransport

__all__ = [
    "OBSERVATION_SCHEMA_VERSION",
    "OBSERVATION_SIZE",
    "ObservationSchema",
    "ObservationSegment",
    "TelemetryObservationEncoder",
    "TelemetryValidationError",
    "RacingEnv",
    "default_low_level_controller",
    "MultiTrackRacingEnv",
    "ScrTransportConfig",
    "TorcsScrTransport",
    "TRAINING_TRACKS",
    "VALIDATION_TRACKS",
    "HELD_OUT_TEST_TRACKS",
    "ALL_APPROVED_TRACKS",
    "TrackRoleError",
    "validate_track_training_selection",
]
