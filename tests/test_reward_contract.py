"""Unit tests for the named reward components and termination logic in v3 reward."""

from __future__ import annotations

import numpy as np
import pytest

from torcs_ai.envs.racing import RacingEnv


def _make_sample_sensors(
    *,
    dist: float = 100.0,
    pos: int = 2,
    track_pos: float = 0.0,
    angle: float = 0.0,
    speed_y: float = 0.0,
    damage: float = 0.0,
    finished: bool = False,
) -> dict:
    return {
        "distRaced": dist,
        "racePos": pos,
        "trackPos": track_pos,
        "angle": angle,
        "speedY": speed_y,
        "damage": damage,
        "finished": finished,
        "raceFinished": finished,
        "speedX": 80.0,
        "stucktimer": 0.0,
    }


def test_reward_progress_component() -> None:
    prev = _make_sample_sensors(dist=100.0)
    curr = _make_sample_sensors(dist=103.0)
    controls = {"steer": 0.0, "accel": 1.0, "brake": 0.0}

    total_reward, components = RacingEnv._reward(prev, curr, controls)
    assert components["progress"] == pytest.approx(3.0)
    assert components["damage_penalty"] == 0.0
    assert components["finish_bonus"] == 0.0


def test_reward_position_gain_component() -> None:
    prev = _make_sample_sensors(dist=100.0, pos=3)
    curr = _make_sample_sensors(dist=100.0, pos=2)  # Overtook 1 opponent
    controls = {"steer": 0.0, "accel": 0.5, "brake": 0.0}

    _, components = RacingEnv._reward(prev, curr, controls)
    assert components["position_gain"] == pytest.approx(0.5)


def test_reward_damage_penalty_component() -> None:
    prev = _make_sample_sensors(dist=100.0, damage=50.0)
    curr = _make_sample_sensors(dist=100.0, damage=150.0)  # Delta damage = 100
    controls = {"steer": 0.0, "accel": 0.0, "brake": 0.5}

    _, components = RacingEnv._reward(prev, curr, controls)
    assert components["damage_penalty"] == pytest.approx(10.0)  # 0.1 * 100


def test_reward_finish_bonus_component() -> None:
    prev = _make_sample_sensors(dist=1000.0, finished=False)
    curr = _make_sample_sensors(dist=1002.0, finished=True)
    controls = {"steer": 0.0, "accel": 0.8, "brake": 0.0}

    total_reward, components = RacingEnv._reward(prev, curr, controls)
    assert components["finish_bonus"] == pytest.approx(100.0)
    assert total_reward > 100.0


def test_reward_track_and_angle_penalties() -> None:
    prev = _make_sample_sensors(dist=100.0)
    curr = _make_sample_sensors(dist=100.0, track_pos=0.8, angle=0.3, speed_y=5.0)
    controls = {"steer": 0.2, "accel": 0.5, "brake": 0.0}

    _, components = RacingEnv._reward(prev, curr, controls)
    assert components["track_penalty"] == pytest.approx(0.35 * (0.8**2))
    assert components["angle_penalty"] == pytest.approx(0.2 * 0.3)
    assert components["lateral_penalty"] == pytest.approx(0.02 * 5.0)


def test_termination_reasons() -> None:
    # Running
    term, reason = RacingEnv._termination(_make_sample_sensors())
    assert not term
    assert reason == "running"

    # Race finished
    term, reason = RacingEnv._termination(_make_sample_sensors(finished=True))
    assert term
    assert reason == "race_finished"

    # Off track
    term, reason = RacingEnv._termination(_make_sample_sensors(track_pos=1.6))
    assert term
    assert reason == "off_track"

    # Backwards driving
    sensors = _make_sample_sensors()
    sensors["speedX"] = -6.0
    term, reason = RacingEnv._termination(sensors)
    assert term
    assert reason == "backwards"

    # Stuck
    sensors = _make_sample_sensors()
    sensors["stucktimer"] = 350.0
    term, reason = RacingEnv._termination(sensors)
    assert term
    assert reason == "stuck"

    # Terminal damage
    term, reason = RacingEnv._termination(_make_sample_sensors(damage=12000.0))
    assert term
    assert reason == "terminal_damage"
