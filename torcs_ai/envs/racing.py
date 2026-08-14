"""Gymnasium-compatible racing environment built on raw TORCS telemetry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional RL dependency
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from torcs_ai.controllers.actions import TacticalIntent, decode_tactical_action
from torcs_ai.controllers.actuator import (
    DEFAULT_SLEW_RATES,
    apply_slew_limiter,
)
from torcs_ai.controllers.expert import expert_tactical_action
from torcs_ai.controllers.shield import apply_safety_shield
from torcs_ai.envs.telemetry import (
    OBSERVATION_SIZE,
    TelemetryObservationEncoder,
    TelemetryValidationError,
)


@runtime_checkable
class RacingTransport(Protocol):
    """Protocol for environment telemetry transport layers."""

    def reset(self, *, seed: int | None = None) -> Mapping[str, Any]: ...

    def step(self, controls: Mapping[str, float]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _controls_dict(
    steer: float, accel: float, brake: float, gear: int = 1
) -> dict[str, float]:
    return {
        "steer": _clip(float(steer), -1.0, 1.0),
        "accel": _clip(float(accel), 0.0, 1.0),
        "brake": _clip(float(brake), 0.0, 1.0),
        "gear": float(gear),
    }


def default_low_level_controller(
    intent: TacticalIntent,
    sensors: Mapping[str, Any],
) -> Mapping[str, float]:
    """Compute low-level continuous steering, pedals, and gear from tactical intent."""
    track_pos = float(sensors.get("trackPos", 0.0))
    angle = float(sensors.get("angle", 0.0))
    speed = float(sensors.get("speedX", 0.0))
    gear = int(sensors.get("gear", 1))

    steer = (intent.lateral_target - track_pos) * 0.8 + angle * 1.8
    steer = _clip(steer, -1.0, 1.0)

    target_speed = 300.0 * intent.speed_fraction
    if speed < target_speed:
        accel = _clip((target_speed - speed) / 50.0, 0.2, 1.0)
        brake = 0.0
    else:
        accel = 0.0
        brake = _clip((speed - target_speed) / 30.0, 0.1, 1.0)

    rpm = float(sensors.get("rpm", 0.0))
    if gear < 1:
        gear = 1
    if rpm > 7500 and gear < 6:
        gear += 1
    elif rpm < 3000 and gear > 1:
        gear -= 1

    return _controls_dict(steer=steer, accel=accel, brake=brake, gear=gear)


_BaseEnv: Any = gym.Env if gym is not None else object


class RacingEnv(_BaseEnv):
    """A deterministic contract around an injected SCR transport."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        transport: RacingTransport,
        *,
        competitor_count: int = 1,
        max_steps: int = 100_000,
        track_length: float = 10_000.0,
        controller: Callable[
            [TacticalIntent, Mapping[str, Any]], Mapping[str, float]
        ] = default_low_level_controller,
        safety_shield: bool = True,
        slew_rates: Mapping[str, float] | None = None,
        teacher_guidance: float = 0.0,
        speed_limit_scale: float = 1.0,
        sharp_turn_braking: bool = False,
    ) -> None:
        if gym is None:
            raise ImportError("RacingEnv requires the 'rl' extra (gymnasium)")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not np.isfinite(teacher_guidance) or not 0.0 <= teacher_guidance <= 1.0:
            raise ValueError("teacher_guidance must be finite and within [0, 1]")
        self.transport = transport
        self.max_steps = max_steps
        self.controller = controller
        self.safety_shield = safety_shield
        self.slew_rates = dict(slew_rates or DEFAULT_SLEW_RATES)
        self.teacher_guidance = float(teacher_guidance)
        if not np.isfinite(speed_limit_scale) or not 0.25 <= speed_limit_scale <= 1.5:
            raise ValueError("speed_limit_scale must be finite and within [0.25, 1.5]")
        self.speed_limit_scale = float(speed_limit_scale)
        self.sharp_turn_braking = bool(sharp_turn_braking)
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
        self._sensors: Mapping[str, Any] | None = None
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
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

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
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
            controls["steer"],
            controls["accel"],
            controls["brake"],
        )
        observation = self.encoder.encode(next_sensors)
        self._sensors = next_sensors
        self._step_count += 1
        reward, components = self._reward(previous, next_sensors, controls)
        teacher_action = expert_tactical_action(
            previous,
            speed_limit_scale=self.speed_limit_scale,
            sharp_turn_braking=self.sharp_turn_braking,
        )
        teacher_term = 0.0
        if self.teacher_guidance:
            teacher_term = self.teacher_guidance * (
                1.0 if intent.action_id == teacher_action else -1.0
            )
            components["teacher_guidance"] = teacher_term
            reward += teacher_term
        terminated, termination_reason = self._termination(next_sensors)
        if terminated and termination_reason != "race_finished":
            components["terminal_failure_penalty"] = 100.0
            reward -= components["terminal_failure_penalty"]
        truncated = self._step_count >= self.max_steps and not terminated
        if truncated:
            termination_reason = "max_steps"
        info = {
            "termination_reason": termination_reason,
            "reward_components": components,
            "tactical_action": intent.action_id,
            "teacher_action": teacher_action,
            "teacher_guidance": self.teacher_guidance,
            "speed_limit_scale": self.speed_limit_scale,
            "sharp_turn_braking": self.sharp_turn_braking,
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
            float(current.get("distRaced", 0.0))
            - float(previous.get("distRaced", 0.0)),
            -5.0,
            5.0,
        )
        position_gain = _clip(
            float(previous.get("racePos", 1.0)) - float(current.get("racePos", 1.0)),
            -2.0,
            2.0,
        )
        track_position = abs(float(current.get("trackPos", 0.0)))
        track_penalty = 0.35 * track_position * track_position
        angle_penalty = 0.2 * abs(float(current.get("angle", 0.0)))
        lateral_penalty = 0.02 * abs(float(current.get("speedY", 0.0)))
        damage_delta = max(
            0.0,
            float(current.get("damage", 0.0)) - float(previous.get("damage", 0.0)),
        )
        # Damage used to cost at most two reward points, making a 946-point
        # collision almost irrelevant beside a full race's progress. Keep
        # the incremental signal bounded, but large enough to affect the
        # policy and value targets.
        damage_penalty = min(0.1 * damage_delta, 100.0)
        finish_bonus = (
            100.0
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
        if bool(sensors.get("finished", False)) or bool(
            sensors.get("raceFinished", False)
        ):
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
