"""Regression tests for the public Gym-TORCS action contract."""

from unittest.mock import patch

import numpy as np

from gym_torcs import TorcsEnv


def _build_env(*, throttle: bool, gear_change: bool) -> TorcsEnv:
    with patch("gym_torcs.os.system"), patch("gym_torcs.time.sleep"):
        return TorcsEnv(vision=False, throttle=throttle, gear_change=gear_change)


def test_gear_action_without_throttle_uses_two_values_and_index_one():
    env = _build_env(throttle=False, gear_change=True)

    assert env.action_space.shape == (2,)
    np.testing.assert_array_equal(env.action_space.low, [-1.0, 1.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 6.0])
    assert env.agent_to_torcs(np.array([0.25, 4.0])) == {"steer": 0.25, "gear": 4.0}


def test_gear_action_with_throttle_uses_three_values_and_index_two():
    env = _build_env(throttle=True, gear_change=True)

    assert env.action_space.shape == (3,)
    assert env.agent_to_torcs(np.array([0.25, 0.8, 5.0])) == {
        "steer": 0.25,
        "accel": 0.8,
        "gear": 5.0,
    }
