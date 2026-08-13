"""Deterministic tactical demonstrator used as a training/evaluation baseline."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .actions import TacticalAction


# The forward-ray estimator is intentionally conservative on Forza.  Its
# corners close earlier than the other bundled benchmark tracks, so using the
# generic ceiling causes a late brake entry and an unrecoverable spin.  Keep
# this table small and explicit: unknown tracks retain the validated default
# rather than inheriting an unsafe guess.
_TRACK_SPEED_LIMIT_SCALES = {
    "road/forza": 0.72,
}
_SHARP_TURN_BRAKING_TRACKS = {"road/forza"}


def track_speed_limit_scale(track: str | None) -> float:
    """Return the audited forward-ray speed scale for an approved track."""

    if track is None:
        return 1.0
    key = track.strip().lower().replace("\\", "/")
    return float(_TRACK_SPEED_LIMIT_SCALES.get(key, 1.0))


def track_sharp_turn_braking(track: str | None) -> bool:
    """Return whether an approved track needs a short-horizon brake floor."""

    if track is None:
        return False
    key = track.strip().lower().replace("\\", "/")
    return key in _SHARP_TURN_BRAKING_TRACKS


def _curvature_signal(sensors: Mapping[str, Any]) -> float:
    track = np.asarray(sensors.get("track", []), dtype=np.float32)
    if track.ndim != 1:
        return 0.0
    valid = track[np.isfinite(track) & (track >= 0.0)]
    if valid.size < 5:
        return 0.0
    return float(np.clip(np.mean(np.abs(np.diff(valid))) / 80.0, 0.0, 1.0))


def track_speed_limit(
    sensors: Mapping[str, Any],
    *,
    speed_limit_scale: float = 1.0,
    sharp_turn_braking: bool = False,
) -> float:
    """Estimate a conservative speed ceiling from the forward track rays."""

    track = np.asarray(sensors.get("track", []), dtype=np.float32)
    if track.ndim != 1 or track.size < 11:
        return 300.0
    center = track[track.size // 2 - 2 : track.size // 2 + 3]
    finite = center[np.isfinite(center) & (center >= 0.0)]
    if finite.size < 3:
        return 300.0
    # A short forward horizon must lower the target before the car accumulates
    # an unrecoverable heading error at the bend entry.
    scale = float(np.clip(speed_limit_scale, 0.25, 1.5))
    forward_min = float(np.min(finite))
    ceiling = 80.0 + 2.5 * scale * forward_min
    if sharp_turn_braking:
        # Forza presents a very short valid ray horizon before its hairpin.
        # Lower the floor only for the audited track profile; the generic
        # controller remains unchanged for unknown and other known tracks.
        ceiling = min(ceiling, 30.0 + 2.5 * forward_min)
    return float(
        np.clip(ceiling, 30.0 if sharp_turn_braking else 80.0, 300.0)
    )


def expert_tactical_action(
    sensors: Mapping[str, Any],
    *,
    speed_limit_scale: float = 1.0,
    sharp_turn_braking: bool = False,
) -> int:
    """Choose a safe nine-action intent from current SCR telemetry.

    This is deliberately deterministic and transparent. It is not counted as
    learned performance; it supplies a reproducible teacher/baseline for
    behavior-cloning and for diagnosing whether PPO has learned anything.
    """

    track_pos = float(sensors.get("trackPos", 0.0))
    angle = float(sensors.get("angle", 0.0))
    speed = float(sensors.get("speedX", 0.0))
    curvature = _curvature_signal(sensors)
    # Keep the same TORCS heading convention as the low-level controller:
    # positive angle requires positive steering for recovery.
    desired_steer = -0.8 * track_pos + 1.8 * angle
    if desired_steer < -0.12:
        lateral = 0
    elif desired_steer > 0.12:
        lateral = 2
    else:
        lateral = 1

    target_speed = 300.0 * (1.0 - 0.55 * curvature)
    target_speed *= 1.0 - min(abs(angle), 1.0) * 0.35
    target_speed = min(
        target_speed,
        track_speed_limit(
            sensors,
            speed_limit_scale=speed_limit_scale,
            sharp_turn_braking=sharp_turn_braking,
        ),
    )
    if speed > target_speed + 18.0 or abs(angle) > 0.55:
        pace = 0
    elif speed < target_speed - 25.0:
        pace = 2
    else:
        pace = 1
    return int(TacticalAction(lateral * 3 + pace))
