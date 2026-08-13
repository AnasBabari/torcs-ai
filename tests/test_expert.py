"""Deterministic expert baseline tests."""

from torcs_ai.controllers import TacticalAction, expert_tactical_action


def _sensors(**overrides):
    values = {
        "trackPos": 0.0,
        "angle": 0.0,
        "speedX": 100.0,
        "track": [200.0] * 19,
    }
    values.update(overrides)
    return values


def test_expert_pushes_when_far_below_target_speed() -> None:
    assert expert_tactical_action(_sensors(speedX=5.0)) in (
        TacticalAction.LEFT_PUSH,
        TacticalAction.CENTER_PUSH,
        TacticalAction.RIGHT_PUSH,
    )


def test_expert_brakes_for_large_heading_error() -> None:
    action = expert_tactical_action(_sensors(angle=0.8, speedX=150.0))
    assert action in (TacticalAction.LEFT_BRAKE, TacticalAction.RIGHT_BRAKE)
