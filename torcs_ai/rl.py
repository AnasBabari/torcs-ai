"""Reinforcement learning pipelines, evaluation, and benchmark contracts for TORCS."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .controllers import (
    expert_tactical_action,
    track_sharp_turn_braking,
    track_speed_limit_scale,
)
from .envs import (
    HELD_OUT_TEST_TRACKS,
    OBSERVATION_SCHEMA_VERSION,
    TRAINING_TRACKS,
    VALIDATION_TRACKS,
    MultiTrackRacingEnv,
    RacingEnv,
    ScrTransportConfig,
    TorcsScrTransport,
    default_low_level_controller,
    validate_track_training_selection,
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


def compute_bootstrap_ci(
    data: Sequence[float],
    *,
    num_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 7,
) -> tuple[float, float]:
    """Compute percentile bootstrap confidence interval over independent episode outcomes."""
    if not data:
        return (0.0, 0.0)
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 1 or np.all(arr == arr[0]):
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    resamples = rng.choice(arr, size=(num_resamples, len(arr)), replace=True)
    means = np.mean(resamples, axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low = float(np.percentile(means, 100.0 * alpha))
    high = float(np.percentile(means, 100.0 * (1.0 - alpha)))
    return (low, high)


def compute_iqm(data: Sequence[float]) -> float:
    """Compute 25% trimmed interquartile mean (IQM) across independent episodes."""
    if not data:
        return 0.0
    arr = np.sort(np.asarray(data, dtype=np.float64))
    n = len(arr)
    if n < 4:
        return float(np.mean(arr))
    q1_idx = int(np.floor(0.25 * n))
    q3_idx = int(np.ceil(0.75 * n))
    trimmed = arr[q1_idx:q3_idx]
    return float(np.mean(trimmed)) if len(trimmed) else float(np.mean(arr))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _get_git_info() -> dict[str, Any]:
    """Query git revision and dirty status if available."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return {"commit_sha": commit, "dirty": bool(status)}
    except Exception:
        return {"commit_sha": "unknown", "dirty": False}


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
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Build a complete, scientifically rigorous reproducibility record for a run."""
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
                    val = result.get("track")
                    if val is None or isinstance(val, str):
                        profile_tracks.append(val)
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

    # Hardware & PyTorch device info
    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except Exception:
        pass

    # Check for held-out training contamination
    contaminated_held_out = False
    if training and "tracks" in training:
        for t in training["tracks"]:
            if t and str(t).strip().lower().replace("\\", "/") in HELD_OUT_TEST_TRACKS:
                contaminated_held_out = True

    payload: dict[str, Any] = {
        "manifest_schema": "torcs-run-v1",
        "role": role,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git": _get_git_info(),
        "simulator": manifest,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cuda_available": cuda_available,
            "dependencies": dependency_versions,
        },
        "environment": {
            "observation_schema": OBSERVATION_SCHEMA_VERSION,
            "action_schema": ACTION_SCHEMA_VERSION,
            "reward_schema": REWARD_SCHEMA_VERSION,
            "track": track,
            "max_steps": max_steps,
            "teacher_guidance": teacher_guidance,
            "driving_profiles": driving_profiles,
            "track_partition_roles": {
                "training": list(TRAINING_TRACKS),
                "validation": list(VALIDATION_TRACKS),
                "held_out_test": list(HELD_OUT_TEST_TRACKS),
            },
            "contaminated_held_out": contaminated_held_out,
        },
    }
    if seed is not None:
        payload["seed"] = seed
    if training is not None:
        payload["training"] = training
    if results is not None:
        payload["results"] = results
    if checkpoint_path is not None and checkpoint_path.is_file():
        payload["checkpoint"] = {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "format": "stable-baselines3-zip",
            "security_notice": "Load only explicitly trusted local checkpoint artifacts",
        }
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


def save_checkpoint_atomic(
    model: Any,
    output_path: Path,
) -> tuple[Path, str]:
    """Save a Stable-Baselines3 model atomically and return its path and SHA-256 digest."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = output_path.with_suffix(".zip")
    temp_zip = output_path.parent / f".{zip_path.name}.tmp"

    model.save(str(temp_zip))
    # Replace atomically
    temp_zip.replace(zip_path)

    sha256_digest = _sha256_file(zip_path)
    sha_file = zip_path.with_suffix(".zip.sha256")
    sha_file.write_text(f"{sha256_digest}  {zip_path.name}\n", encoding="utf-8")
    return zip_path, sha256_digest


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
    return RacingEnv(
        transport,
        max_steps=max_steps,
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
    allow_held_out_training: bool = False,
) -> MultiTrackRacingEnv:
    """Build seeded per-track environments for one-at-a-time PPO training."""
    unique_tracks = tuple(dict.fromkeys(tracks))
    if not unique_tracks or any(not track.strip() for track in unique_tracks):
        raise ValueError("tracks must contain at least one non-empty value")

    validate_track_training_selection(
        unique_tracks, allow_held_out_training=allow_held_out_training
    )

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
    env: Any,
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
    setattr(model, "_torcs_bc_summary", None)
    if demonstrations is not None:
        bc_summary = behavior_clone_ppo_policy(
            model,
            demonstrations,
            epochs=bc_epochs,
            batch_size=bc_batch_size,
            learning_rate=bc_learning_rate,
            seed=seed,
        )
        bc_path = output_path.with_name(f"{output_path.name}_bc")
        save_checkpoint_atomic(model, bc_path)
        bc_summary["checkpoint_path"] = str(bc_path.with_suffix(".zip"))
        setattr(model, "_torcs_bc_summary", bc_summary)

    model.learn(total_timesteps=total_timesteps, callback=checkpoint)
    saved_zip, sha256_hash = save_checkpoint_atomic(model, output_path)
    setattr(model, "_torcs_checkpoint_path", str(saved_zip))
    setattr(model, "_torcs_checkpoint_sha256", sha256_hash)
    return model


def evaluate_policy(
    env: Any,
    model: Any,
    *,
    episodes: int = 3,
) -> list[dict[str, Any]]:
    """Evaluate without exploration and return auditable episode summaries with latency timing."""
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
        latencies_ms: list[float] = []
        start_position = None
        max_dist = 0.0

        while not (terminated or truncated):
            t0 = time.perf_counter_ns()
            action, _ = model.predict(observation, deterministic=True)
            t1 = time.perf_counter_ns()
            latencies_ms.append((t1 - t0) / 1_000_000.0)

            observation, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            steps += 1
            shield_interventions += int(bool(info.get("shield_intervened", False)))
            speed_sum += float(info.get("speed_x", 0.0))
            max_dist = max(max_dist, float(info.get("dist_raced", 0.0)))
            if start_position is None:
                start_position = int(info.get("race_position", 1))

            act = int(info.get("tactical_action", action))
            if 0 <= act <= 8:
                action_counts[act] += 1
            teacher_matches += int(act == int(info.get("teacher_action", -1)))

        probabilities = [count / max(steps, 1) for count in action_counts]
        action_entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
        distance_km = abs(float(info.get("dist_raced", max_dist))) / 1000.0
        final_position = int(info.get("race_position", 1))
        overtakes = max(0, (start_position or 1) - final_position)
        lost_positions = max(0, final_position - (start_position or 1))

        p50_latency = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
        p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

        results.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": info.get("termination_reason", "unknown"),
                "dist_raced": float(info.get("dist_raced", max_dist)),
                "damage": float(info.get("damage", 0.0)),
                "race_position": final_position,
                "finish": info.get("termination_reason") == "race_finished",
                "win": final_position == 1
                and info.get("termination_reason") == "race_finished",
                "podium": final_position <= 3
                and info.get("termination_reason") == "race_finished",
                "overtakes": overtakes,
                "lost_positions": lost_positions,
                "shield_interventions": shield_interventions,
                "shield_interventions_per_km": shield_interventions
                / max(distance_km, 1e-6),
                "mean_speed_x": speed_sum / max(steps, 1),
                "damage_per_km": float(info.get("damage", 0.0))
                / max(distance_km, 1e-6),
                "action_counts": action_counts,
                "action_entropy": action_entropy,
                "teacher_agreement_rate": teacher_matches / max(steps, 1),
                "latency_p50_ms": p50_latency,
                "latency_p95_ms": p95_latency,
            }
        )
    return results


def summarize_evaluation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode evidence without hiding failures behind mean reward."""
    if not results:
        raise ValueError("evaluation results cannot be empty")
    finishes = [item for item in results if bool(item.get("finish", False))]
    wins = [item for item in results if bool(item.get("win", False))]
    podiums = [item for item in results if bool(item.get("podium", False))]

    action_counts = [0] * 9
    for item in results:
        counts = item.get("action_counts", [0] * 9)
        if isinstance(counts, list) and len(counts) == 9:
            action_counts = [
                left + int(right) for left, right in zip(action_counts, counts)
            ]
    action_total = max(sum(action_counts), 1)
    dominant_action_share = max(action_counts) / action_total

    rewards = [float(item["reward"]) for item in results]
    distances = [float(item.get("dist_raced", 0.0)) for item in results]
    reward_ci_low, reward_ci_high = compute_bootstrap_ci(rewards)

    all_latencies_p50 = [float(item.get("latency_p50_ms", 0.0)) for item in results]
    all_latencies_p95 = [float(item.get("latency_p95_ms", 0.0)) for item in results]

    return {
        "episodes": len(results),
        "finish_rate": len(finishes) / len(results),
        "win_rate": len(wins) / len(results),
        "podium_rate": len(podiums) / len(results),
        "median_return": statistics.median(rewards),
        "iqm_return": compute_iqm(rewards),
        "return_ci_95": [reward_ci_low, reward_ci_high],
        "median_distance_m": statistics.median(distances),
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
        "median_shield_interventions_per_km": statistics.median(
            float(item.get("shield_interventions_per_km", 0.0)) for item in results
        ),
        "mean_speed_x": statistics.fmean(
            float(item.get("mean_speed_x", 0.0)) for item in results
        ),
        "mean_teacher_agreement": statistics.fmean(
            float(item.get("teacher_agreement_rate", 0.0)) for item in results
        ),
        "mean_overtakes": statistics.fmean(
            float(item.get("overtakes", 0)) for item in results
        ),
        "action_counts": action_counts,
        "dominant_action_share": dominant_action_share,
        "action_collapsed": dominant_action_share >= 0.85,
        "inference_latency_p50_ms": statistics.median(all_latencies_p50)
        if all_latencies_p50
        else 0.0,
        "inference_latency_p95_ms": statistics.median(all_latencies_p95)
        if all_latencies_p95
        else 0.0,
    }


def compare_with_expert(
    policy: dict[str, Any], expert: dict[str, Any]
) -> dict[str, Any]:
    """Apply an explicit multi-gate competitiveness hierarchy to aggregate evidence."""
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

    gates = {
        "gate_1_completion": float(policy["finish_rate"]) > 0.0,
        "gate_2_traffic_completion": finish_ok,
        "gate_3_damage_control": damage_ok,
        "gate_4_pace_near_baseline": pace_ok
        if finishes_exist(policy, expert)
        else False,
        "gate_5_position_overtaking": position_ok,
        "gate_6_action_diversity": diversity_ok,
    }
    return {
        "competitive": all(gates.values()),
        "gates": gates,
        "summary": "Passed all gates"
        if all(gates.values())
        else "Failed one or more competitive gates",
    }


def finishes_exist(policy: dict[str, Any], expert: dict[str, Any]) -> bool:
    return (
        policy.get("median_finish_steps") is not None
        and expert.get("median_finish_steps") is not None
    )


def evaluate_fixed_action(
    env: Any,
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
        def predict(
            self, observation: Any, deterministic: bool = True
        ) -> tuple[int, None]:
            del observation, deterministic
            return action, None

    return evaluate_policy(env, _FixedPolicy(), episodes=episodes)


def evaluate_expert(env: Any, *, episodes: int = 3) -> list[dict[str, Any]]:
    """Evaluate the deterministic tactical teacher as a non-learned baseline."""

    class _ExpertPolicy:
        def predict(
            self, observation: Any, deterministic: bool = True
        ) -> tuple[int, None]:
            del deterministic
            sensors = getattr(env, "_sensors", None)
            if sensors is None:
                # Multi-track environment fallback
                active = getattr(env, "_active", None)
                sensors = (
                    getattr(active, "_sensors", None) if active is not None else None
                )
            if sensors is None:
                raise RuntimeError(
                    "expert policy requires an active environment with sensors"
                )
            return expert_tactical_action(
                sensors,
                speed_limit_scale=getattr(env, "speed_limit_scale", 1.0),
                sharp_turn_braking=getattr(env, "sharp_turn_braking", False),
            ), None

    return evaluate_policy(env, _ExpertPolicy(), episodes=episodes)
