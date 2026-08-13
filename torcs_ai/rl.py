"""Optional native TORCS reinforcement-learning entry points.

This module deliberately keeps simulator startup explicit. Importing it does
not launch TORCS, open a UDP socket, or train a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .envs import RacingEnv, ScrTransportConfig, TorcsScrTransport
from .runtime import (
    SessionConfig,
    TorcsInstallation,
    TorcsSession,
    write_single_track_race_config,
)


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
) -> Any:
    """Train a PPO policy with bounded, explicit native-environment ownership."""

    if total_timesteps < 1 or checkpoint_freq < 1:
        raise ValueError("training bounds must be positive")
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
