"""Unit tests for reproducible seed schedules in environment resets and evaluation."""

from __future__ import annotations

import numpy as np

from tests.test_gymnasium_contract import MockTransport
from torcs_ai.envs.racing import RacingEnv


def test_racing_env_seed_repeatability() -> None:
    env = RacingEnv(MockTransport(), max_steps=50)

    obs1, info1 = env.reset(seed=12345)
    obs2, info2 = env.reset(seed=12345)
    np.testing.assert_array_equal(obs1, obs2)

    # Subsequent steps with same action produce identical observations
    obs1_step, r1, t1, tr1, _ = env.step(4)

    env.reset(seed=12345)
    obs2_step, r2, t2, tr2, _ = env.step(4)

    np.testing.assert_array_equal(obs1_step, obs2_step)
    assert r1 == r2
    assert t1 == t2
    assert tr1 == tr2
