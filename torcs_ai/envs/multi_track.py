"""Seeded track-switching wrapper and track partition roles for native TORCS training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised without the RL extra
    gym = None  # type: ignore[assignment]

from .racing import RacingEnv

TRAINING_TRACKS: tuple[str, ...] = ("road/alpine-1", "road/forza", "oval/michigan")
VALIDATION_TRACKS: tuple[str, ...] = ("road/ruudskogen",)
HELD_OUT_TEST_TRACKS: tuple[str, ...] = ("road/spring", "road/street-1")
ALL_APPROVED_TRACKS: tuple[str, ...] = (
    TRAINING_TRACKS + VALIDATION_TRACKS + HELD_OUT_TEST_TRACKS
)


class TrackRoleError(ValueError):
    """Raised when training attempts to use held-out test tracks without explicit research override."""


def validate_track_training_selection(
    tracks: Sequence[str | None],
    *,
    allow_held_out_training: bool = False,
) -> bool:
    """Validate that training tracks do not include held-out test tracks.

    Returns True if the run is clean, or False if held-out tracks were trained on
    via explicit override (marking the run as contaminated).
    """
    for track in tracks:
        if track is None or track == "default":
            continue
        normalized = track.strip().lower().replace("\\", "/")
        if normalized in HELD_OUT_TEST_TRACKS:
            if not allow_held_out_training:
                raise TrackRoleError(
                    f"Track {track!r} is a held-out test track. Training on held-out tracks "
                    "is forbidden to preserve scientific integrity. Use --allow-held-out-training "
                    "only for explicit contamination studies."
                )
            return False  # Run is contaminated
    return True


_BaseEnv: Any = gym.Env if gym is not None else object


class MultiTrackRacingEnv(_BaseEnv):
    """Run one owned ``RacingEnv`` per episode and report its track.

    Environments are constructed outside this wrapper and only one is
    active at a time. Switching tracks closes the previous transport so
    a training process never shares an SCR port with another native race.
    """

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(self, environments: Mapping[str, RacingEnv]) -> None:
        if gym is None:
            raise ImportError("MultiTrackRacingEnv requires the 'rl' extra (gymnasium)")
        if not environments:
            raise ValueError("at least one track environment is required")
        self.environments = dict(environments)
        self.track_names = tuple(self.environments)
        first = next(iter(self.environments.values()))
        self.observation_space = first.observation_space
        self.action_space = first.action_space
        for name, environment in self.environments.items():
            if environment.observation_space != self.observation_space:
                raise ValueError(f"track {name!r} has a different observation space")
            if environment.action_space != self.action_space:
                raise ValueError(f"track {name!r} has a different action space")
        self._active_name: str | None = None
        self._active: RacingEnv | None = None

    @property
    def active_track(self) -> str | None:
        return self._active_name

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        requested = (options or {}).get("track")
        if requested is not None:
            if requested not in self.environments:
                raise ValueError(
                    f"track must be one of {self.track_names}, got {requested!r}"
                )
            name = str(requested)
        else:
            index = int(self.np_random.integers(0, len(self.track_names)))
            name = self.track_names[index]
        if self._active is not None and self._active_name != name:
            self._active.close()
        self._active_name = name
        self._active = self.environments[name]
        observation, info = self._active.reset(seed=seed)
        return observation, {**info, "track": name}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._active is None or self._active_name is None:
            raise RuntimeError("reset must be called before step")
        observation, reward, terminated, truncated, info = self._active.step(action)
        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                **info,
                "track": self._active_name,
            },
        )

    def close(self) -> None:
        for environment in self.environments.values():
            environment.close()
        self._active = None
        self._active_name = None
