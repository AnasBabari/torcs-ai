"""Gymnasium-compatible native TORCS racing environment adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Optional, Protocol

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only without the RL extra
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from ..controllers import (
    DEFAULT_SLEW_RATES,
    TacticalIntent,
    apply_safety_shield,
    apply_slew_limiter,
    decode_tactical_action,
)
from .telemetry import OBSERVATION_SIZE, TelemetryObservationEncoder, TelemetryValidationError


class RacingTransport(Protocol):
    """Minimal transport seam used by the environment and fake tests."""

    def reset(self, *, seed: Optional[int] = None) -> Mapping[str, Any]: ...

    def step(self, controls: Mapping[str, float]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def default_low_level_controller(
    intent: TacticalIntent, sensors: Mapping[str, Any]
) -> dict[str, float]:
    """Convert a tactical intent into smooth bounded actuator targets."""

    lateral_error = float(sensors.get("trackPos", 0.0)) - intent.lateral_target
    heading_error = float(sensors.get("angle", 0.0))
    # TORCS reports a positive heading error when the car points left of the
    # track tangent, so the recovery steering term must be positive.  The
    # lateral term keeps the signed track-position correction unchanged.
    steer = _clip(-0.8 * lateral_error + 1.8 * heading_error, -1.0, 1.0)
    speed = float(sensors.get("speedX", 0.0))
    track = np.asarray(sensors.get("track", []), dtype=np.float32)
    valid_track = (
        track[np.isfinite(track) & (track >= 0.0)]
        if track.ndim == 1
        else np.asarray([], dtype=np.float32)
    )
    if valid_track.size >= 5:
        curvature_signal = float(np.mean(np.abs(np.diff(valid_track))))
        curvature = float(np.clip(curvature_signal / 80.0, 0.0, 1.0))
    else:
        curvature = 0.0
    target_speed = 300.0 * intent.speed_fraction * (1.0 - 0.55 * curvature)
    target_speed *= 1.0 - min(abs(heading_error), 1.0) * 0.35
    if speed > target_speed + 10.0:
        accel, brake = 0.0, _clip((speed - target_speed) / 60.0, 0.0, 1.0)
    else:
        accel, brake = _clip((target_speed - speed) / 80.0, 0.0, 1.0), 0.0
    if brake > 0.0:
        accel = 0.0
    gear = int(np.clip(round(float(sensors.get("gear", 1))), 1, 6))
    rpm = float(sensors.get("rpm", 0.0))
    if rpm > 8_500.0 and gear < 6:
        gear += 1
    elif rpm < 2_000.0 and gear > 1 and speed > 15.0:
        gear -= 1
    return {
        "steer": steer,
        "accel": accel,
        "brake": brake,
        "gear": float(gear),
    }


if gym is not None:

    class RacingEnv(gym.Env):  # type: ignore[misc]
        """A deterministic contract around an injected SCR transport."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            transport: RacingTransport,
            *,
            competitor_count: int = 1,
            max_steps: int = 100_000,
            track_length: float = 10_000.0,
            controller: Callable[[TacticalIntent, Mapping[str, Any]], Mapping[str, float]] = default_low_level_controller,
            safety_shield: bool = True,
            slew_rates: Optional[Mapping[str, float]] = None,
        ) -> None:
            if max_steps < 1:
                raise ValueError("max_steps must be positive")
            self.transport = transport
            self.max_steps = max_steps
            self.controller = controller
            self.safety_shield = safety_shield
            self.slew_rates = dict(slew_rates or DEFAULT_SLEW_RATES)
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(OBSERVATION_SIZE,),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(9)
            self.encoder = TelemetryObservationEncoder(
                competitor_count=competitor_count,
                track_length=track_length,
            )
            self._sensors: Optional[Mapping[str, Any]] = None
            self._step_count = 0

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
            super().reset(seed=seed)
            try:
                sensors = self.transport.reset(seed=seed)
            except TypeError:
                sensors = self.transport.reset()
            self.encoder.previous_sensors = None
            self.encoder.previous_controls = (0.0, 0.0, 0.0)
            self._sensors = dict(sensors)
            self._previous_controls = {
                "steer": 0.0,
                "accel": 0.0,
                "brake": 0.0,
                "gear": float(np.clip(self._sensors.get("gear", 1.0), 1.0, 6.0)),
            }
            self._step_count = 0
            try:
                observation = self.encoder.encode(self._sensors)
            except TelemetryValidationError as exc:
                raise RuntimeError(f"invalid telemetry at reset: {exc}") from exc
            return observation, {"schema": "competitive-telemetry-v1"}

        def step(self, action: int):
            if self._sensors is None:
                raise RuntimeError("reset must be called before step")
            intent = decode_tactical_action(action)
            controls = dict(self.controller(intent, self._sensors))
            policy_controls = dict(controls)
            controls["steer"] = _clip(float(controls.get("steer", 0.0)), -1.0, 1.0)
            controls["accel"] = _clip(float(controls.get("accel", 0.0)), 0.0, 1.0)
            controls["brake"] = _clip(float(controls.get("brake", 0.0)), 0.0, 1.0)
            if controls["brake"] > 0.0:
                controls["accel"] = 0.0
            controls, slew_limited = apply_slew_limiter(
                controls,
                self._previous_controls,
                rates=self.slew_rates,
            )
            shield_intervened = False
            shield_reasons: tuple[str, ...] = ()
            if self.safety_shield:
                shielded = apply_safety_shield(controls, self._sensors)
                controls = shielded.controls
                shield_intervened = shielded.intervened
                shield_reasons = shielded.reasons
            self._previous_controls = dict(controls)
            next_sensors = dict(self.transport.step(controls))
            previous = self._sensors
            self.encoder.previous_sensors = previous
            self.encoder.previous_controls = (
                controls["steer"], controls["accel"], controls["brake"]
            )
            observation = self.encoder.encode(next_sensors)
            self._sensors = next_sensors
            self._step_count += 1
            reward, components = self._reward(previous, next_sensors, controls)
            terminated, termination_reason = self._termination(next_sensors)
            truncated = self._step_count >= self.max_steps and not terminated
            if truncated:
                termination_reason = "max_steps"
            info = {
                "termination_reason": termination_reason,
                "reward_components": components,
                "tactical_action": intent.action_id,
                "controls": controls,
                "policy_controls": policy_controls,
                "slew_limited": slew_limited,
                "dist_raced": float(next_sensors.get("distRaced", 0.0)),
                "damage": float(next_sensors.get("damage", 0.0)),
                "race_position": int(next_sensors.get("racePos", 0)),
                "speed_x": float(next_sensors.get("speedX", 0.0)),
                "shield_intervened": shield_intervened,
                "shield_reasons": shield_reasons,
            }
            return observation, reward, terminated, truncated, info

        def close(self) -> None:
            self.transport.close()
            self._sensors = None

        @staticmethod
        def _reward(
            previous: Mapping[str, Any],
            current: Mapping[str, Any],
            controls: Mapping[str, float],
        ) -> tuple[float, dict[str, float]]:
            progress = _clip(
                float(current.get("distRaced", 0.0)) - float(previous.get("distRaced", 0.0)),
                -5.0,
                5.0,
            )
            position_gain = _clip(
                float(previous.get("racePos", 1.0))
                - float(current.get("racePos", 1.0)),
                -2.0,
                2.0,
            )
            track_penalty = 0.2 * abs(float(current.get("trackPos", 0.0)))
            angle_penalty = 0.1 * abs(float(current.get("angle", 0.0)))
            lateral_penalty = 0.05 * abs(float(current.get("speedY", 0.0))) / 50.0
            damage_delta = max(
                0.0,
                float(current.get("damage", 0.0)) - float(previous.get("damage", 0.0)),
            )
            damage_penalty = 2.0 * min(damage_delta / 1000.0, 1.0)
            finish_bonus = (
                10.0
                if bool(current.get("finished", False))
                or bool(current.get("raceFinished", False))
                else 0.0
            )
            components = {
                "progress": progress,
                "position_gain": 0.5 * position_gain,
                "track_penalty": track_penalty,
                "angle_penalty": angle_penalty,
                "lateral_penalty": lateral_penalty,
                "damage_penalty": damage_penalty,
                "finish_bonus": finish_bonus,
            }
            return (
                progress
                + components["position_gain"]
                + finish_bonus
                - sum(
                    components[key]
                    for key in (
                        "track_penalty",
                        "angle_penalty",
                        "lateral_penalty",
                        "damage_penalty",
                    )
                )
            ), components

        @staticmethod
        def _termination(sensors: Mapping[str, Any]) -> tuple[bool, str]:
            if bool(sensors.get("finished", False)) or bool(sensors.get("raceFinished", False)):
                return True, "race_finished"
            if abs(float(sensors.get("trackPos", 0.0))) >= 1.5:
                return True, "off_track"
            if float(sensors.get("speedX", 0.0)) < -5.0:
                return True, "backwards"
            if float(sensors.get("stucktimer", 0.0)) >= 300.0:
                return True, "stuck"
            if float(sensors.get("damage", 0.0)) >= 10_000.0:
                return True, "terminal_damage"
            return False, "running"

else:

    class RacingEnv:  # pragma: no cover - compatibility error path
        """Placeholder that explains how to install the RL environment extra."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("RacingEnv requires the 'rl' extra (gymnasium)")
