"""Optional native TORCS reinforcement-learning entry points.

This module deliberately keeps simulator startup explicit. Importing it does
not launch TORCS, open a UDP socket, or train a model.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .controllers import (
    expert_tactical_action,
    track_sharp_turn_braking,
    track_speed_limit_scale,
)
from .envs import (
    MultiTrackRacingEnv,
    RacingEnv,
    ScrTransportConfig,
    TorcsScrTransport,
    default_low_level_controller,
)
from .imitation import ExpertDemonstrations, behavior_clone_ppo_policy
from .runtime import (
    SessionConfig,
    TorcsInstallation,
    TorcsSession,
    inspect_installation,
    write_single_track_race_config,
)

ACTION_SCHEMA_VERSION = "tactical-9-v1"
REWARD_SCHEMA_VERSION = "progress-position-safety-teacher-v3"


def build_run_manifest(
    torcs_home: Path,
    *,
    role: str,
    track: str | None,
    max_steps: int,
    seed: int | None = None,
    training: dict[str, Any] | None = None,
    results: Any = None,
    teacher_guidance: float = 0.0,
) -> dict[str, Any]:
    """Build a reproducibility record for a train/evaluate/benchmark run."""

    if not role.strip() or max_steps < 1:
        raise ValueError("role cannot be empty and max_steps must be positive")
    if not 0.0 <= teacher_guidance <= 1.0:
        raise ValueError("teacher_guidance must be within [0, 1]")
    profile_tracks: list[str | None]
    if track == "matrix":
        profile_tracks = []
        source = training.get("tracks") if isinstance(training, dict) else None
        if isinstance(source, (list, tuple)):
            profile_tracks.extend(
                item if item is None or isinstance(item, str) else str(item)
                for item in source
            )
        if not profile_tracks and isinstance(results, dict):
            for result in results.values():
                if isinstance(result, dict):
                    value = result.get("track")
                    if value is None or isinstance(value, str):
                        profile_tracks.append(value)
        profile_tracks = list(dict.fromkeys(profile_tracks))
        if not profile_tracks:
            profile_tracks = [None]
    else:
        profile_tracks = [track]
    driving_profiles = {
        str(item or "default"): {
            "speed_limit_scale": track_speed_limit_scale(item),
            "sharp_turn_braking": track_sharp_turn_braking(item),
        }
        for item in profile_tracks
    }
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
            "teacher_guidance": teacher_guidance,
            "driving_profiles": driving_profiles,
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
    teacher_guidance: float = 0.0,
    text_only: bool = True,
    simulator_timeout_microseconds: int = 100_000,
) -> RacingEnv:
    """Build one explicit native environment from an immutable installation."""

    if simulator_timeout_microseconds < 1:
        raise ValueError("simulator_timeout_microseconds must be positive")

    session = TorcsSession(
        TorcsInstallation(torcs_home),
        runtime_home,
        config=SessionConfig(
            race_config=(
                r"config\raceman\codex-race.xml"
                if track is not None
                else r"config\raceman\quickrace.xml"
            ),
            text_only=text_only,
            timeout_microseconds=simulator_timeout_microseconds,
        ),
    )
    session.prepare(overwrite=overwrite_runtime)
    if track is not None:
        write_single_track_race_config(session.runtime_home, track)
    transport = TorcsScrTransport(
        session,
        config=ScrTransportConfig(port=port, max_steps=max_steps),
    )
    speed_limit_scale = track_speed_limit_scale(track)
    sharp_turn_braking = track_sharp_turn_braking(track)
    from functools import partial

    controller = partial(
        default_low_level_controller,
        speed_limit_scale=speed_limit_scale,
        sharp_turn_braking=sharp_turn_braking,
    )
    return RacingEnv(
        transport,
        max_steps=max_steps,
        controller=controller,
        teacher_guidance=teacher_guidance,
        speed_limit_scale=speed_limit_scale,
        sharp_turn_braking=sharp_turn_braking,
    )


def build_multi_track_env(
    torcs_home: Path,
    runtime_home: Path,
    tracks: list[str] | tuple[str, ...],
    *,
    port: int = 3001,
    max_steps: int = 100_000,
    overwrite_runtime: bool = False,
    teacher_guidance: float = 0.0,
    simulator_timeout_microseconds: int = 100_000,
) -> MultiTrackRacingEnv:
    """Build seeded per-track environments for one-at-a-time PPO training."""

    unique_tracks = tuple(dict.fromkeys(tracks))
    if not unique_tracks or any(not track.strip() for track in unique_tracks):
        raise ValueError("tracks must contain at least one non-empty value")
    environments = {
        track: build_native_env(
            torcs_home,
            runtime_home / f"track-{index}",
            port=port,
            max_steps=max_steps,
            track=track,
            overwrite_runtime=overwrite_runtime,
            teacher_guidance=teacher_guidance,
            simulator_timeout_microseconds=simulator_timeout_microseconds,
        )
        for index, track in enumerate(unique_tracks)
    }
    return MultiTrackRacingEnv(environments)


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
    target_kl: float | None = 0.02,
    demonstrations: ExpertDemonstrations | None = None,
    bc_epochs: int = 8,
    bc_batch_size: int = 256,
    bc_learning_rate: float = 1e-3,
) -> Any:
    """Train a PPO policy with bounded, explicit native-environment ownership."""

    if total_timesteps < 1 or checkpoint_freq < 1 or n_steps < 1 or batch_size < 1:
        raise ValueError("training bounds must be positive")
    if batch_size > n_steps:
        raise ValueError("batch_size cannot exceed n_steps")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if target_kl is not None and target_kl <= 0.0:
        raise ValueError("target_kl must be positive when provided")
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
        target_kl=target_kl,
        policy_kwargs={"net_arch": [256, 256]},
    )
    model._torcs_bc_summary = None
    if demonstrations is not None:
        model._torcs_bc_summary = behavior_clone_ppo_policy(
            model,
            demonstrations,
            epochs=bc_epochs,
            batch_size=bc_batch_size,
            learning_rate=bc_learning_rate,
            seed=seed,
        )
        bc_path = output_path.with_name(f"{output_path.name}_bc")
        model.save(str(bc_path))
        model._torcs_bc_summary["checkpoint_path"] = str(
            bc_path.with_suffix(".zip")
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
        action_counts = [0] * 9
        teacher_matches = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            steps += 1
            shield_interventions += int(bool(info.get("shield_intervened", False)))
            speed_sum += float(info.get("speed_x", 0.0))
            action = int(info.get("tactical_action", action))
            if 0 <= action <= 8:
                action_counts[action] += 1
            teacher_matches += int(action == int(info.get("teacher_action", -1)))
        probabilities = [count / max(steps, 1) for count in action_counts]
        action_entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
        distance_km = abs(float(info.get("dist_raced", 0.0))) / 1000.0
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
                "finish": info.get("termination_reason") == "race_finished",
                "damage_per_km": float(info.get("damage", 0.0))
                / max(distance_km, 1e-6),
                "action_counts": action_counts,
                "action_entropy": action_entropy,
                "teacher_agreement_rate": teacher_matches / max(steps, 1),
            }
        )
    return results


def summarize_evaluation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode evidence without hiding failures behind mean reward."""

    if not results:
        raise ValueError("evaluation results cannot be empty")
    finishes = [item for item in results if bool(item.get("finish", False))]
    action_counts = [0] * 9
    for item in results:
        counts = item.get("action_counts", [0] * 9)
        if isinstance(counts, list) and len(counts) == 9:
            action_counts = [
                left + int(right) for left, right in zip(action_counts, counts)
            ]
    action_total = max(sum(action_counts), 1)
    dominant_action_share = max(action_counts) / action_total
    return {
        "episodes": len(results),
        "finish_rate": len(finishes) / len(results),
        "median_finish_steps": (
            statistics.median(float(item["steps"]) for item in finishes)
            if finishes
            else None
        ),
        "median_race_position": statistics.median(
            float(item.get("race_position", 0)) for item in results
        ),
        "median_damage_per_km": statistics.median(
            float(item.get("damage_per_km", 0.0)) for item in results
        ),
        "mean_speed_x": statistics.fmean(
            float(item.get("mean_speed_x", 0.0)) for item in results
        ),
        "mean_teacher_agreement": statistics.fmean(
            float(item.get("teacher_agreement_rate", 0.0)) for item in results
        ),
        "action_counts": action_counts,
        "dominant_action_share": dominant_action_share,
        "action_collapsed": dominant_action_share >= 0.9,
    }


def compare_with_expert(policy: dict[str, Any], expert: dict[str, Any]) -> dict[str, Any]:
    """Apply an explicit initial competitiveness gate to aggregate evidence."""

    finish_ok = float(policy["finish_rate"]) >= float(expert["finish_rate"])
    position_ok = float(policy["median_race_position"]) <= float(
        expert["median_race_position"]
    )
    policy_damage = float(policy["median_damage_per_km"])
    expert_damage = float(expert["median_damage_per_km"])
    damage_ok = policy_damage <= max(expert_damage * 1.1, 5.0)
    policy_steps = policy.get("median_finish_steps")
    expert_steps = expert.get("median_finish_steps")
    pace_ok = (
        policy_steps is not None
        and expert_steps is not None
        and float(policy_steps) <= float(expert_steps) * 1.05
    )
    diversity_ok = not bool(policy.get("action_collapsed", True))
    checks = {
        "finish_rate": finish_ok,
        "position": position_ok,
        "damage_per_km": damage_ok,
        "finish_pace": pace_ok,
        "action_diversity": diversity_ok,
    }
    return {"competitive": all(checks.values()), "checks": checks}


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


def evaluate_expert(env: RacingEnv, *, episodes: int = 3) -> list[dict[str, Any]]:
    """Evaluate the deterministic tactical teacher as a non-learned baseline."""

    class _ExpertPolicy:
        def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, None]:
            del deterministic
            sensors = getattr(env, "_sensors", None)
            if sensors is None:
                raise RuntimeError("expert policy requires an active environment")
            return expert_tactical_action(
                sensors,
                speed_limit_scale=getattr(env, "speed_limit_scale", 1.0),
                sharp_turn_braking=getattr(env, "sharp_turn_braking", False),
            ), None

    return evaluate_policy(env, _ExpertPolicy(), episodes=episodes)
