"""Deterministic expert baseline tests."""

from torcs_ai.controllers import (
    TacticalAction,
    expert_tactical_action,
    track_sharp_turn_braking,
    track_speed_limit,
    track_speed_limit_scale,
)


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


def test_expert_selects_right_recovery_for_positive_heading_error() -> None:
    action = expert_tactical_action(_sensors(angle=0.3, speedX=60.0))
    assert action in (TacticalAction.RIGHT_HOLD, TacticalAction.RIGHT_PUSH)


def test_forward_track_rays_lower_speed_limit_before_a_bend() -> None:
    assert track_speed_limit(_sensors(track=[12.0] * 19)) == 110.0
    assert track_speed_limit(_sensors(track=[200.0] * 19)) == 300.0


def test_expert_brakes_when_forward_track_distance_is_short() -> None:
    action = expert_tactical_action(_sensors(speedX=150.0, track=[12.0] * 19))
    assert action in (
        TacticalAction.LEFT_BRAKE,
        TacticalAction.CENTER_BRAKE,
        TacticalAction.RIGHT_BRAKE,
    )


def test_forza_profile_uses_short_horizon_braking_without_changing_default() -> None:
    sensors = _sensors(track=[12.0] * 19)
    assert track_speed_limit(sensors) == 110.0
    assert track_speed_limit(sensors, sharp_turn_braking=True) == 60.0
    assert track_speed_limit_scale("road\\forza") == 0.72
    assert track_sharp_turn_braking("road/forza")
    assert not track_sharp_turn_braking("road/alpine-1")
