"""Comprehensive unit tests for RL pipelines, evaluation, and baseline comparisons."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.test_gymnasium_contract import MockTransport
from torcs_ai.envs.racing import RacingEnv
from torcs_ai.imitation import ExpertDemonstrations
from torcs_ai.rl import (
    build_run_manifest,
    compare_with_expert,
    evaluate_expert,
    evaluate_fixed_action,
    evaluate_policy,
    summarize_evaluation,
    train_ppo,
)


class DummyPolicy:
    def __init__(self, action: int = 4) -> None:
        self.action = action

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        del obs, deterministic
        return self.action, None


def test_evaluate_policy_collects_racing_metrics() -> None:
    env = RacingEnv(MockTransport(), max_steps=20)
    model = DummyPolicy(action=2)

    results = evaluate_policy(env, model, episodes=2)
    assert len(results) == 2
    for ep in results:
        assert "reward" in ep
        assert "steps" in ep
        assert "finish" in ep
        assert "damage_per_km" in ep
        assert "shield_interventions_per_km" in ep
        assert "action_counts" in ep
        assert "latency_p50_ms" in ep
        assert "latency_p95_ms" in ep
        assert ep["action_counts"][2] > 0


def test_evaluate_fixed_action_and_expert() -> None:
    env = RacingEnv(MockTransport(), max_steps=15)
    fixed_res = evaluate_fixed_action(env, action=4, episodes=2)
    assert len(fixed_res) == 2
    assert fixed_res[0]["action_counts"][4] > 0

    expert_res = evaluate_expert(env, episodes=2)
    assert len(expert_res) == 2


def test_summarize_evaluation_metrics() -> None:
    episodes = [
        {
            "reward": 50.0,
            "steps": 100,
            "finish": True,
            "win": True,
            "podium": True,
            "race_position": 1,
            "damage_per_km": 2.0,
            "dist_raced": 500.0,
            "shield_interventions_per_km": 0.0,
            "mean_speed_x": 80.0,
            "teacher_agreement_rate": 0.9,
            "overtakes": 2,
            "action_counts": [10, 10, 10, 10, 20, 10, 10, 10, 10],
            "latency_p50_ms": 0.5,
            "latency_p95_ms": 1.2,
        },
        {
            "reward": 30.0,
            "steps": 80,
            "finish": False,
            "win": False,
            "podium": False,
            "race_position": 4,
            "damage_per_km": 15.0,
            "dist_raced": 300.0,
            "shield_interventions_per_km": 1.0,
            "mean_speed_x": 60.0,
            "teacher_agreement_rate": 0.5,
            "overtakes": 0,
            "action_counts": [5, 5, 5, 5, 40, 5, 5, 5, 5],
            "latency_p50_ms": 0.6,
            "latency_p95_ms": 1.4,
        },
    ]

    summary = summarize_evaluation(episodes)
    assert summary["episodes"] == 2
    assert summary["finish_rate"] == 0.5
    assert summary["win_rate"] == 0.5
    assert summary["podium_rate"] == 0.5
    assert summary["median_return"] == 40.0
    assert summary["iqm_return"] == 40.0
    assert len(summary["return_ci_95"]) == 2
    assert summary["inference_latency_p50_ms"] > 0.0


def test_compare_with_expert_hierarchy() -> None:
    policy_summary = {
        "finish_rate": 1.0,
        "median_race_position": 1.0,
        "median_damage_per_km": 2.0,
        "median_finish_steps": 500.0,
        "action_collapsed": False,
    }
    expert_summary = {
        "finish_rate": 1.0,
        "median_race_position": 1.0,
        "median_damage_per_km": 2.5,
        "median_finish_steps": 510.0,
        "action_collapsed": False,
    }
    comp = compare_with_expert(policy_summary, expert_summary)
    assert comp["competitive"] is True
    assert all(comp["gates"].values())


def test_train_ppo_with_mock_env(tmp_path: Path) -> None:
    env = RacingEnv(MockTransport(), max_steps=20)
    output = tmp_path / "model"

    # Single-step PPO training
    model = train_ppo(
        env,
        output,
        total_timesteps=128,
        n_steps=64,
        batch_size=32,
        checkpoint_freq=1000,
        demonstrations=None,
    )
    assert model is not None
    assert (tmp_path / "model.zip").is_file()
    assert (tmp_path / "model.zip.sha256").is_file()


def test_train_ppo_with_bc_demonstrations(tmp_path: Path) -> None:
    env = RacingEnv(MockTransport(), max_steps=20)
    output = tmp_path / "model_bc"

    obs = np.zeros((10, 118), dtype=np.float32)
    actions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 4], dtype=np.int64)
    demos = ExpertDemonstrations(
        observations=obs, actions=actions, tracks=("road/alpine-1",)
    )

    model = train_ppo(
        env,
        output,
        total_timesteps=128,
        n_steps=64,
        batch_size=32,
        checkpoint_freq=1000,
        demonstrations=demos,
        bc_epochs=2,
        bc_batch_size=4,
    )
    assert model._torcs_bc_summary is not None
    assert model._torcs_bc_summary["training_accuracy"] >= 0.0
