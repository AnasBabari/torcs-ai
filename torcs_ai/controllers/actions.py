"""The canonical nine-action tactical racing contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TacticalAction(IntEnum):
    LEFT_BRAKE = 0
    LEFT_HOLD = 1
    LEFT_PUSH = 2
    CENTER_BRAKE = 3
    CENTER_HOLD = 4
    CENTER_PUSH = 5
    RIGHT_BRAKE = 6
    RIGHT_HOLD = 7
    RIGHT_PUSH = 8


@dataclass(frozen=True)
class TacticalIntent:
    """A high-level intent consumed by the smooth controller."""

    action_id: int
    lateral_target: float
    speed_fraction: float


def decode_tactical_action(
    action: int,
    *,
    left_target: float = -0.35,
    center_target: float = 0.0,
    right_target: float = 0.35,
) -> TacticalIntent:
    """Decode an action ID into a bounded lateral and pace target.

    The decoder is deliberately strict: values outside the nine-action
    contract are rejected rather than silently wrapped into another policy.
    """

    try:
        action_enum = TacticalAction(action)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"action must be an integer in [0, 8], got {action!r}") from exc

    lateral_index, pace_index = divmod(int(action_enum), 3)
    lateral_targets = (left_target, center_target, right_target)
    pace_fractions = (0.65, 0.85, 1.0)
    target = float(lateral_targets[lateral_index])
    if not -1.0 <= target <= 1.0:
        raise ValueError("lateral targets must be within [-1, 1]")
    return TacticalIntent(
        action_id=int(action_enum),
        lateral_target=target,
        speed_fraction=pace_fractions[pace_index],
    )
