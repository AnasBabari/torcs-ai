"""Seeded track-switching wrapper for native multi-track PPO training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised without the RL extra
    gym = None  # type: ignore[assignment]

from .racing import RacingEnv


if gym is not None:

    class MultiTrackRacingEnv(gym.Env):  # type: ignore[misc]
        """Run one owned ``RacingEnv`` per episode and report its track.

        Environments are constructed outside this wrapper and only one is
        active at a time.  Switching tracks closes the previous transport so
        a training process never shares an SCR port with another native race.
        """

        metadata = {"render_modes": []}

        def __init__(self, environments: Mapping[str, RacingEnv]) -> None:
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
            self._active_name: Optional[str] = None
            self._active: Optional[RacingEnv] = None

        @property
        def active_track(self) -> Optional[str]:
            return self._active_name

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
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

        def step(self, action: int):
            if self._active is None or self._active_name is None:
                raise RuntimeError("reset must be called before step")
            observation, reward, terminated, truncated, info = self._active.step(action)
            return observation, reward, terminated, truncated, {
                **info,
                "track": self._active_name,
            }

        def close(self) -> None:
            for environment in self.environments.values():
                environment.close()
            self._active = None
            self._active_name = None

else:

    class MultiTrackRacingEnv:  # pragma: no cover - compatibility error path
        """Placeholder that explains how to install the RL environment extra."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("MultiTrackRacingEnv requires the 'rl' extra (gymnasium)")
