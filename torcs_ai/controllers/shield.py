"""Deterministic safety shield for native TORCS actuator commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ShieldResult:
    """Controls after safety projection and whether the policy was changed."""

    controls: dict[str, float]
    intervened: bool
    reasons: tuple[str, ...]


def apply_safety_shield(
    controls: Mapping[str, float], sensors: Mapping[str, float]
) -> ShieldResult:
    """Project risky actions into a bounded, recoverable actuator command.

    The shield handles only catastrophic/recovery conditions; tactical
    decisions remain the policy's responsibility and every intervention is
    surfaced in ``ShieldResult``.
    """

    result = {
        "steer": float(np.clip(controls.get("steer", 0.0), -1.0, 1.0)),
        "accel": float(np.clip(controls.get("accel", 0.0), 0.0, 1.0)),
        "brake": float(np.clip(controls.get("brake", 0.0), 0.0, 1.0)),
        "gear": float(np.clip(controls.get("gear", 1.0), 1.0, 6.0)),
    }
    reasons: list[str] = []
    track_pos = float(sensors.get("trackPos", 0.0))
    angle = float(sensors.get("angle", 0.0))
    speed = float(sensors.get("speedX", 0.0))

    if abs(track_pos) >= 1.15:
        result["steer"] = float(np.clip(-np.sign(track_pos) * 0.8, -1.0, 1.0))
        result["accel"] = min(result["accel"], 0.2)
        reasons.append("edge_recovery")
    if abs(angle) >= 0.65:
        result["steer"] = float(np.clip(-angle * 1.25, -1.0, 1.0))
        result["accel"] = 0.0
        result["brake"] = max(result["brake"], min(1.0, abs(angle)))
        reasons.append("heading_recovery")
    if speed < -2.0:
        result["accel"] = 0.0
        result["brake"] = max(result["brake"], 0.4)
        reasons.append("reverse_speed")

    if result["accel"] > 0.0 and result["brake"] > 0.0:
        if result["brake"] >= result["accel"]:
            result["accel"] = 0.0
        else:
            result["brake"] = 0.0
        reasons.append("longitudinal_exclusion")

    return ShieldResult(result, bool(reasons), tuple(reasons))
