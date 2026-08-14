"""Tests for auditable policy aggregation and competitiveness gates."""

from pathlib import Path

import pytest

from torcs_ai.rl import compare_with_expert, summarize_evaluation, train_ppo


def _episode(
    *, action: int, damage_per_km: float = 10.0, steps: int = 100, reward: float = 100.0
) -> dict:
    counts = [0] * 9
    counts[action] = steps
    return {
        "reward": reward,
        "finish": True,
        "steps": steps,
        "race_position": 1,
        "damage_per_km": damage_per_km,
        "mean_speed_x": 100.0,
        "teacher_agreement_rate": 0.8,
        "action_counts": counts,
    }


def test_summary_surfaces_single_action_collapse() -> None:
    summary = summarize_evaluation([_episode(action=5)])
    assert summary["action_collapsed"]
    assert summary["dominant_action_share"] == 1.0


def test_competitiveness_requires_diversity_damage_and_pace() -> None:
    expert = summarize_evaluation([_episode(action=4, steps=100)])
    policy = summarize_evaluation(
        [_episode(action=2, steps=100), _episode(action=8, steps=100)]
    )
    result = compare_with_expert(policy, expert)
    assert result["competitive"]
    collapsed = summarize_evaluation([_episode(action=5, steps=99)])
    assert not compare_with_expert(collapsed, expert)["competitive"]


def test_training_rejects_nonpositive_target_kl() -> None:
    with pytest.raises(ValueError, match="target_kl"):
        train_ppo(object(), Path("unused"), target_kl=0.0)
