"""Gymnasium environment contract tests using check_env and mock transport."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import pytest

try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
except ImportError:
    gym = None  # type: ignore[assignment]
    check_env = None  # type: ignore[assignment]

from torcs_ai.envs import (
    OBSERVATION_SIZE,
    MultiTrackRacingEnv,
    RacingEnv,
)


class MockTransport:
    """Deterministic, side-effect-free in-memory transport for environment contract testing."""

    def __init__(self) -> None:
        self.step_count = 0
        self.closed = False

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        self.step_count = 0
        self.closed = False
        return self._make_sensors(dist_raced=0.0, track_pos=0.0, speed_x=50.0)

    def step(self, controls: Mapping[str, float]) -> dict[str, Any]:
        self.step_count += 1
        accel = float(controls.get("accel", 0.0))
        brake = float(controls.get("brake", 0.0))
        steer = float(controls.get("steer", 0.0))
        speed = max(0.0, 50.0 + (accel - brake) * 10.0)
        dist = self.step_count * speed * 0.02
        track_pos = np.clip(steer * 0.1, -0.9, 0.9)
        return self._make_sensors(dist_raced=dist, track_pos=track_pos, speed_x=speed)

    def close(self) -> None:
        self.closed = True

    @staticmethod
    def _make_sensors(
        dist_raced: float, track_pos: float, speed_x: float
    ) -> dict[str, Any]:
        return {
            "angle": 0.0,
            "trackPos": track_pos,
            "speedX": speed_x,
            "speedY": 0.0,
            "speedZ": 0.0,
            "rpm": 4000.0,
            "gear": 3,
            "fuel": 90.0,
            "damage": 0.0,
            "wheelSpinVel": [50.0, 50.0, 50.0, 50.0],
            "track": [100.0] * 19,
            "opponents": [-1.0] * 36,
            "racePos": 1,
            "laps": 3,
            "lap": 0,
            "curLapTime": 10.0,
            "lastLapTime": 0.0,
            "racePhase": 1,
            "distRaced": dist_raced,
            "finished": False,
        }


@pytest.mark.skipif(gym is None, reason="Requires Gymnasium")
def test_racing_env_passes_gymnasium_check_env() -> None:
    transport = MockTransport()
    env = RacingEnv(transport, max_steps=100)
    # Gymnasium's check_env checks spaces, reset return, step return, dtype, and bounds
    check_env(env, skip_render_check=True)


@pytest.mark.skipif(gym is None, reason="Requires Gymnasium")
def test_multi_track_env_passes_gymnasium_check_env() -> None:
    env1 = RacingEnv(MockTransport(), max_steps=100)
    env2 = RacingEnv(MockTransport(), max_steps=100)
    multi_env = MultiTrackRacingEnv({"road/alpine-1": env1, "road/forza": env2})
    check_env(multi_env, skip_render_check=True)


@pytest.mark.skipif(gym is None, reason="Requires Gymnasium")
def test_racing_env_space_properties() -> None:
    env = RacingEnv(MockTransport(), max_steps=50)
    assert env.observation_space.shape == (OBSERVATION_SIZE,)
    assert env.observation_space.dtype == np.float32
    assert getattr(env.action_space, "n", None) == 9

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert isinstance(info, dict)
    assert info.get("schema") == "competitive-telemetry-v1"

    # Step with all 9 valid actions
    for action in range(9):
        next_obs, reward, term, trunc, step_info = env.step(action)
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (OBSERVATION_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(step_info, dict)
        assert "reward_components" in step_info
        assert "controls" in step_info


@pytest.mark.skipif(gym is None, reason="Requires Gymnasium")
def test_invalid_action_rejection() -> None:
    env = RacingEnv(MockTransport(), max_steps=50)
    env.reset(seed=1)
    with pytest.raises(ValueError, match="action must be an integer in"):
        env.step(9)  # Action out of bounds [0, 8]
    with pytest.raises(ValueError, match="action must be an integer in"):
        env.step(-1)


@pytest.mark.skipif(gym is None, reason="Requires Gymnasium")
def test_reset_after_truncation_and_close() -> None:
    transport = MockTransport()
    env = RacingEnv(transport, max_steps=5)
    env.reset(seed=1)
    truncated = False
    for _ in range(10):
        _, _, _, truncated, _ = env.step(4)
        if truncated:
            break
    assert truncated
    # Reset after truncation
    obs, info = env.reset(seed=2)
    assert obs.shape == (OBSERVATION_SIZE,)
    env.close()
    assert transport.closed
