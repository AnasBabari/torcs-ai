"""Optional native TORCS reinforcement-learning entry points.

This module deliberately keeps simulator startup explicit. Importing it does
not launch TORCS, open a UDP socket, or train a model.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .envs import RacingEnv, ScrTransportConfig, TorcsScrTransport
from .runtime import (
    SessionConfig,
    TorcsInstallation,
    TorcsSession,
    inspect_installation,
    write_single_track_race_config,
)

ACTION_SCHEMA_VERSION = "tactical-9-v1"
REWARD_SCHEMA_VERSION = "progress-position-safety-v1"


def build_run_manifest(
    torcs_home: Path,
    *,
    role: str,
    track: str | None,
    max_steps: int,
    seed: int | None = None,
    training: dict[str, Any] | None = None,
    results: Any = None,
) -> dict[str, Any]:
    """Build a reproducibility record for a train/evaluate/benchmark run."""

    if not role.strip() or max_steps < 1:
        raise ValueError("role cannot be empty and max_steps must be positive")
    manifest = inspect_installation(TorcsInstallation(torcs_home)).to_dict()
    dependency_versions: dict[str, str] = {}
    for package in ("numpy", "torch", "gymnasium", "stable-baselines3"):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = "unavailable"
    payload: dict[str, Any] = {
        "manifest_schema": "torcs-run-v1",
        "role": role,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulator": manifest,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependency_versions,
        },
        "environment": {
            "observation_schema": "competitive-telemetry-v1",
            "action_schema": ACTION_SCHEMA_VERSION,
            "reward_schema": REWARD_SCHEMA_VERSION,
            "track": track,
            "max_steps": max_steps,
        },
    }
    if seed is not None:
        payload["seed"] = seed
    if training is not None:
        payload["training"] = training
    if results is not None:
        payload["results"] = results
    return payload


def write_json_atomic(path: Path, payload: Any) -> Path:
    """Write a JSON artifact atomically beneath its requested directory."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def build_native_env(
    torcs_home: Path,
    runtime_home: Path,
    *,
    port: int = 3001,
    max_steps: int = 100_000,
    track: str | None = None,
    overwrite_runtime: bool = False,
) -> RacingEnv:
    """Build one explicit native environment from an immutable installation."""

    session = TorcsSession(
        TorcsInstallation(torcs_home),
        runtime_home,
        config=SessionConfig(
            race_config=(
                r"config\raceman\codex-race.xml"
                if track is not None
                else r"config\raceman\quickrace.xml"
            )
        ),
    )
    session.prepare(overwrite=overwrite_runtime)
    if track is not None:
        write_single_track_race_config(session.runtime_home, track)
    transport = TorcsScrTransport(
        session,
        config=ScrTransportConfig(port=port, max_steps=max_steps),
    )
    return RacingEnv(transport, max_steps=max_steps)


def train_ppo(
    env: RacingEnv,
    output_path: Path,
    *,
    total_timesteps: int = 100_000,
    seed: int = 7,
    device: str = "auto",
    checkpoint_freq: int = 10_000,
    n_steps: int = 256,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
) -> Any:
    """Train a PPO policy with bounded, explicit native-environment ownership."""

    if total_timesteps < 1 or checkpoint_freq < 1 or n_steps < 1 or batch_size < 1:
        raise ValueError("training bounds must be positive")
    if batch_size > n_steps:
        raise ValueError("batch_size cannot exceed n_steps")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as exc:  # pragma: no cover - optional RL dependency
        raise ImportError("train_ppo requires the 'rl' extra") from exc

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(output_path.parent / "checkpoints"),
        name_prefix=output_path.stem,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        policy_kwargs={"net_arch": [256, 256]},
    )
    model.learn(total_timesteps=total_timesteps, callback=checkpoint)
    model.save(str(output_path))
    return model


def evaluate_policy(
    env: RacingEnv,
    model: Any,
    *,
    episodes: int = 3,
) -> list[dict[str, Any]]:
    """Evaluate without exploration and return auditable episode summaries."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    results: list[dict[str, Any]] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=episode)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        info: dict[str, Any] = {}
        shield_interventions = 0
        speed_sum = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            steps += 1
            shield_interventions += int(bool(info.get("shield_intervened", False)))
            speed_sum += float(info.get("speed_x", 0.0))
        results.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": info.get("termination_reason", "unknown"),
                "dist_raced": float(info.get("dist_raced", 0.0)),
                "damage": float(info.get("damage", 0.0)),
                "race_position": int(info.get("race_position", 0)),
                "shield_interventions": shield_interventions,
                "mean_speed_x": speed_sum / max(steps, 1),
            }
        )
    return results


def evaluate_fixed_action(
    env: RacingEnv,
    *,
    action: int = 4,
    episodes: int = 3,
) -> list[dict[str, Any]]:
    """Evaluate the deterministic center/hold baseline for comparison."""

    if not 0 <= action <= 8:
        raise ValueError("baseline action must be in [0, 8]")
    if episodes < 1:
        raise ValueError("episodes must be positive")

    class _FixedPolicy:
        def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, None]:
            del observation, deterministic
            return action, None

    return evaluate_policy(env, _FixedPolicy(), episodes=episodes)
