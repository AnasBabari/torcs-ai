"""Expert-demonstration collection and PPO behavioural-cloning warm starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .controllers import expert_tactical_action


@dataclass(frozen=True)
class ExpertDemonstrations:
    """Finite observation/action pairs collected from the audited teacher."""

    observations: np.ndarray
    actions: np.ndarray
    tracks: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.observations.ndim != 2 or self.actions.ndim != 1:
            raise ValueError("demonstrations must be a matrix and an action vector")
        if len(self.observations) != len(self.actions) or not len(self.actions):
            raise ValueError("demonstrations must contain aligned non-empty samples")
        if not np.isfinite(self.observations).all():
            raise ValueError("demonstration observations must be finite")
        if np.any((self.actions < 0) | (self.actions > 8)):
            raise ValueError("demonstration actions must be within [0, 8]")

    @property
    def action_counts(self) -> list[int]:
        return np.bincount(self.actions, minlength=9).astype(int).tolist()


def _active_racing_env(env: Any) -> Any:
    active = getattr(env, "_active", None)
    return active if active is not None else env


def collect_expert_demonstrations(
    env: Any,
    *,
    episodes_per_track: int = 1,
    sample_stride: int = 2,
    seed: int = 7,
) -> ExpertDemonstrations:
    """Drive complete teacher episodes and retain bend-aware training pairs."""

    if episodes_per_track < 1 or sample_stride < 1:
        raise ValueError("expert episode and stride bounds must be positive")
    track_names = tuple(getattr(env, "track_names", ("default",)))
    observations: list[np.ndarray] = []
    actions: list[int] = []
    for track_index, track in enumerate(track_names):
        for episode in range(episodes_per_track):
            options = {"track": track} if track != "default" else None
            observation, _ = env.reset(
                seed=seed + track_index * episodes_per_track + episode,
                options=options,
            )
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                active = _active_racing_env(env)
                sensors = getattr(active, "_sensors", None)
                if sensors is None:
                    raise RuntimeError("expert collection requires active telemetry")
                action = expert_tactical_action(
                    sensors,
                    speed_limit_scale=getattr(active, "speed_limit_scale", 1.0),
                    sharp_turn_braking=getattr(active, "sharp_turn_braking", False),
                )
                if step % sample_stride == 0:
                    observations.append(np.asarray(observation, dtype=np.float32).copy())
                    actions.append(action)
                observation, _, terminated, truncated, _ = env.step(action)
                step += 1
    return ExpertDemonstrations(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        tracks=track_names,
    )


def behavior_clone_ppo_policy(
    model: Any,
    demonstrations: ExpertDemonstrations,
    *,
    epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 7,
) -> dict[str, Any]:
    """Warm-start a discrete SB3 policy from expert actions using weighted CE."""

    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("behaviour-cloning bounds must be positive")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - part of the RL extra
        raise ImportError("behaviour cloning requires torch") from exc

    policy = model.policy
    device = policy.device
    observations = torch.as_tensor(demonstrations.observations, device=device)
    actions = torch.as_tensor(demonstrations.actions, device=device)
    counts = np.asarray(demonstrations.action_counts, dtype=np.float64)
    present = counts > 0
    weights = np.zeros(9, dtype=np.float32)
    weights[present] = np.sqrt(counts[present].sum() / counts[present])
    if present.any():
        weights[present] /= weights[present].mean()
    class_weights = torch.as_tensor(weights, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    final_loss = 0.0
    policy.train()
    for _ in range(epochs):
        order = torch.randperm(len(actions), generator=generator)
        for start in range(0, len(actions), batch_size):
            indices = order[start : start + batch_size].to(device)
            distribution = policy.get_distribution(observations[indices])
            logits = distribution.distribution.logits
            loss = functional.cross_entropy(
                logits,
                actions[indices],
                weight=class_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    policy.eval()
    with torch.no_grad():
        predicted = policy.get_distribution(observations).distribution.probs.argmax(dim=1)
        accuracy = float((predicted == actions).float().mean().cpu())
        per_action_accuracy = {
            str(action): float((predicted[actions == action] == action).float().mean().cpu())
            for action in range(9)
            if bool((actions == action).any())
        }
    return {
        "samples": int(len(actions)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": final_loss,
        "training_accuracy": accuracy,
        "balanced_training_accuracy": float(
            np.mean(list(per_action_accuracy.values()))
        ),
        "per_action_accuracy": per_action_accuracy,
        "action_counts": demonstrations.action_counts,
        "tracks": list(demonstrations.tracks),
    }
