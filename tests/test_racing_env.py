"""Gymnasium contract tests with an injected deterministic transport."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from torcs_ai.envs import RacingEnv, default_low_level_controller
from torcs_ai.controllers import TacticalIntent


def _sensors(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "angle": 0.0,
        "trackPos": 0.0,
        "speedX": 20.0,
        "speedY": 0.0,
        "speedZ": 0.0,
        "rpm": 2_000.0,
        "gear": 1,
        "fuel": 90.0,
        "damage": 0.0,
        "wheelSpinVel": [10.0] * 4,
        "track": [100.0] * 19,
        "opponents": [200.0] * 36,
        "racePos": 1,
        "laps": 3,
        "lap": 0,
        "distRaced": 0.0,
        "curLapTime": 0.0,
        "lastLapTime": 0.0,
        "racePhase": 0,
        "stucktimer": 0.0,
    }
    values.update(overrides)
    return values


class FakeTransport:
    def __init__(self) -> None:
        self.current = _sensors()
        self.controls: list[Mapping[str, float]] = []

    def reset(self, *, seed: Optional[int] = None) -> Mapping[str, Any]:
        self.current = _sensors()
        self.controls.clear()
        return self.current

    def step(self, controls: Mapping[str, float]) -> Mapping[str, Any]:
        self.controls.append(dict(controls))
        self.current = _sensors(
            distRaced=float(self.current["distRaced"]) + 1.0,
            speedX=min(80.0, float(self.current["speedX"]) + 1.0),
        )
        return self.current

    def close(self) -> None:
        return None


def test_racing_env_observation_and_action_contract() -> None:
    transport = FakeTransport()
    env = RacingEnv(transport, max_steps=2)
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert info["schema"] == "competitive-telemetry-v1"
    next_observation, reward, terminated, truncated, step_info = env.step(5)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert step_info["tactical_action"] == 5
    assert transport.controls[-1]["accel"] >= 0.0
    assert not (transport.controls[-1]["accel"] > 0 and transport.controls[-1]["brake"] > 0)
    env.close()


def test_racing_env_uses_truncation_for_step_limit() -> None:
    env = RacingEnv(FakeTransport(), max_steps=1)
    env.reset()
    _, _, terminated, truncated, info = env.step(4)
    assert not terminated
    assert truncated
    assert info["termination_reason"] == "max_steps"
    env.close()


def test_racing_env_marks_off_track_as_terminal() -> None:
    transport = FakeTransport()

    def off_track_step(controls: Mapping[str, float]) -> Mapping[str, Any]:
        transport.controls.append(dict(controls))
        return _sensors(trackPos=1.6, distRaced=1.0)

    transport.step = off_track_step  # type: ignore[method-assign]
    env = RacingEnv(transport)
    env.reset()
    _, _, terminated, truncated, info = env.step(4)
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "off_track"
    env.close()


def test_racing_env_reports_safety_shield_intervention() -> None:
    transport = FakeTransport()
    transport.current = _sensors(trackPos=1.2, angle=0.8)

    def recovery_step(controls: Mapping[str, float]) -> Mapping[str, Any]:
        transport.controls.append(dict(controls))
        return _sensors(trackPos=1.2, angle=0.8, distRaced=1.0)

    transport.step = recovery_step  # type: ignore[method-assign]
    env = RacingEnv(transport)
    env.reset()
    transport.current = _sensors(trackPos=1.2, angle=0.8)
    env._sensors = transport.current  # type: ignore[attr-defined]
    _, _, _, _, info = env.step(8)
    assert info["shield_intervened"]
    assert "heading_recovery" in info["shield_reasons"]
    assert transport.controls[-1]["accel"] == 0.0
    env.close()


def test_low_level_controller_steers_into_positive_heading_error() -> None:
    controls = default_low_level_controller(
        TacticalIntent(action_id=4, lateral_target=0.0, speed_fraction=0.85),
        _sensors(angle=0.8, trackPos=0.0),
    )
    assert controls["steer"] > 0.0


def test_teacher_guidance_is_explicit_and_audited() -> None:
    env = RacingEnv(FakeTransport(), max_steps=1, teacher_guidance=0.25)
    env.reset()
    _, reward, _, _, info = env.step(4)
    assert info["teacher_action"] == 5
    assert info["teacher_guidance"] == 0.25
    assert info["reward_components"]["teacher_guidance"] == -0.25
    assert 0.0 < reward < 1.0
    env.close()
