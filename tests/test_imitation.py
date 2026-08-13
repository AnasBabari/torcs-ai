"""Behaviour-cloning data-contract tests."""

import numpy as np
import pytest

from test_racing_env import FakeTransport
from torcs_ai.envs import RacingEnv
from torcs_ai.imitation import (
    ExpertDemonstrations,
    behavior_clone_ppo_policy,
    collect_expert_demonstrations,
)


def test_expert_collection_returns_finite_aligned_samples() -> None:
    env = RacingEnv(FakeTransport(), max_steps=5)
    demonstrations = collect_expert_demonstrations(env, sample_stride=2)
    assert demonstrations.observations.shape == (3, 118)
    assert demonstrations.actions.shape == (3,)
    assert sum(demonstrations.action_counts) == 3
    env.close()


def test_demonstrations_reject_invalid_actions() -> None:
    with pytest.raises(ValueError, match="within"):
        ExpertDemonstrations(
            observations=np.zeros((1, 118), dtype=np.float32),
            actions=np.asarray([9], dtype=np.int64),
            tracks=("test",),
        )


def test_behavior_clone_updates_real_sb3_discrete_policy() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    env = RacingEnv(FakeTransport(), max_steps=4)
    demonstrations = collect_expert_demonstrations(env, sample_stride=1)
    model = PPO("MlpPolicy", env, n_steps=4, batch_size=4, verbose=0)
    summary = behavior_clone_ppo_policy(
        model,
        demonstrations,
        epochs=2,
        batch_size=4,
    )
    assert summary["samples"] == 4
    assert np.isfinite(summary["final_loss"])
    assert 0.0 <= summary["training_accuracy"] <= 1.0
    env.close()
