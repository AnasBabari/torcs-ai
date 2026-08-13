"""Compare a saved policy with deterministic center and tactical baselines."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torcs_ai.rl import (  # noqa: E402
    build_native_env,
    build_run_manifest,
    evaluate_expert,
    evaluate_fixed_action,
    evaluate_policy,
    write_json_atomic,
)


def _runtime_slug(track: str | None) -> str:
    """Return a filesystem-safe label for one allowlisted benchmark track."""

    label = track or "default"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")
    return slug or "default"


def benchmark_track(args: argparse.Namespace, track: str | None) -> dict[str, object]:
    """Run the three-policy comparison for one isolated track runtime."""

    runtime_root = args.runtime_home / _runtime_slug(track)
    baseline_env = build_native_env(
        args.torcs_home,
        runtime_root / "baseline",
        max_steps=args.max_steps,
        track=track,
        overwrite_runtime=args.overwrite_runtime,
    )
    try:
        baseline = evaluate_fixed_action(baseline_env, episodes=args.episodes)
    finally:
        baseline_env.close()

    expert_env = build_native_env(
        args.torcs_home,
        runtime_root / "expert",
        max_steps=args.max_steps,
        track=track,
        overwrite_runtime=args.overwrite_runtime,
    )
    try:
        expert = evaluate_expert(expert_env, episodes=args.episodes)
    finally:
        expert_env.close()

    model_env = build_native_env(
        args.torcs_home,
        runtime_root / "model",
        max_steps=args.max_steps,
        track=track,
        overwrite_runtime=args.overwrite_runtime,
    )
    try:
        from stable_baselines3 import PPO

        model = PPO.load(str(args.model), env=model_env)
        learned = evaluate_policy(model_env, model, episodes=args.episodes)
    finally:
        model_env.close()

    return {
        "track": track,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "baseline": baseline,
        "expert": expert,
        "policy": learned,
        "comparison_rule": (
            "learned policy must beat the fixed baseline on held-out runs; "
            "no claim from reward alone"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torcs-home", type=Path, default=Path(r"C:\torcs\torcs"))
    parser.add_argument("--runtime-home", type=Path, default=Path(".runtime") / "benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--track",
        action="append",
        dest="tracks",
        help="approved track; repeat for a matrix (default: quickrace config)",
    )
    parser.add_argument("--overwrite-runtime", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional RL dependency
        raise SystemExit("benchmark_native requires the 'rl' extra") from exc

    tracks = args.tracks or [None]
    results_by_track = {
        _runtime_slug(track): benchmark_track(args, track) for track in tracks
    }
    if len(results_by_track) == 1:
        # Preserve the original single-track artifact contract for consumers.
        payload = next(iter(results_by_track.values())).copy()
    else:
        payload = {
            "track": None,
            "tracks": results_by_track,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "comparison_rule": (
                "evaluate each track independently; a learned policy must beat "
                "the fixed baseline on held-out runs"
            ),
        }
    payload["manifest"] = build_run_manifest(
        args.torcs_home,
        role="benchmark",
        track=tracks[0] if len(tracks) == 1 else "matrix",
        max_steps=args.max_steps,
        results=results_by_track,
    )
    if args.output is not None:
        print(f"benchmark artifact: {write_json_atomic(args.output, payload)}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
