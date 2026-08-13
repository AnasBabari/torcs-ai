"""Compare a saved policy with the deterministic center/hold baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torcs_ai.rl import build_native_env, evaluate_fixed_action, evaluate_policy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torcs-home", type=Path, default=Path(r"C:\torcs\torcs"))
    parser.add_argument("--runtime-home", type=Path, default=Path(".runtime") / "benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--track", default=None)
    parser.add_argument("--overwrite-runtime", action="store_true")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional RL dependency
        raise SystemExit("benchmark_native requires the 'rl' extra") from exc

    baseline_env = build_native_env(
        args.torcs_home,
        args.runtime_home / "baseline",
        max_steps=100_000,
        track=args.track,
        overwrite_runtime=args.overwrite_runtime,
    )
    try:
        baseline = evaluate_fixed_action(baseline_env, episodes=args.episodes)
    finally:
        baseline_env.close()

    model_env = build_native_env(
        args.torcs_home,
        args.runtime_home / "model",
        max_steps=100_000,
        track=args.track,
        overwrite_runtime=args.overwrite_runtime,
    )
    try:
        model = PPO.load(str(args.model), env=model_env)
        learned = evaluate_policy(model_env, model, episodes=args.episodes)
    finally:
        model_env.close()

    print(
        json.dumps(
            {
                "track": args.track,
                "episodes": args.episodes,
                "baseline": baseline,
                "policy": learned,
                "comparison_rule": "learned policy must beat baseline on held-out runs; no claim from reward alone",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
