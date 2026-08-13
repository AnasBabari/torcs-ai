"""Bounded actuator projection for the tactical racing controller."""

from __future__ import annotations

from typing import Mapping

import numpy as np

DEFAULT_SLEW_RATES: dict[str, float] = {
    "steer": 0.15,
    "accel": 0.10,
    "brake": 0.20,
}


def apply_slew_limiter(
    target: Mapping[str, float],
    previous: Mapping[str, float],
    *,
    rates: Mapping[str, float] = DEFAULT_SLEW_RATES,
) -> tuple[dict[str, float], bool]:
    """Limit per-step actuator changes while preserving brake/accel exclusion.

    Longitudinal intent changes switch the inactive actuator off immediately;
    the requested actuator then ramps at its configured rate. This prevents a
    braking command from being masked by residual throttle while still
    avoiding steering and pedal spikes during ordinary policy changes.
    """

    raw = {
        "steer": float(np.clip(target.get("steer", 0.0), -1.0, 1.0)),
        "accel": float(np.clip(target.get("accel", 0.0), 0.0, 1.0)),
        "brake": float(np.clip(target.get("brake", 0.0), 0.0, 1.0)),
        "gear": float(np.clip(target.get("gear", previous.get("gear", 1.0)), 1.0, 6.0)),
    }
    if raw["brake"] > raw["accel"]:
        raw["accel"] = 0.0
    else:
        raw["brake"] = 0.0

    result: dict[str, float] = {}
    limited = False
    for key in ("steer", "accel", "brake"):
        rate = float(rates.get(key, DEFAULT_SLEW_RATES[key]))
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"slew rate for {key} must be finite and positive")
        prior = float(previous.get(key, 0.0))
        delta = float(np.clip(raw[key] - prior, -rate, rate))
        result[key] = float(np.clip(prior + delta, -1.0 if key == "steer" else 0.0, 1.0))
        limited = limited or not np.isclose(delta, raw[key] - prior)

    # Gear is discrete in TORCS and is not slew-limited.
    result["gear"] = float(round(raw["gear"]))
    if result["accel"] > 0.0 and result["brake"] > 0.0:
        if raw["brake"] > 0.0:
            result["accel"] = 0.0
        else:
            result["brake"] = 0.0
        limited = True
    return result, limited
