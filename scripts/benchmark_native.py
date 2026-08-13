"""Compare a saved policy with the deterministic center/hold baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torcs_ai.rl import (  # noqa: E402
    build_native_env,
    build_run_manifest,
    evaluate_fixed_action,
    evaluate_policy,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torcs-home", type=Path, default=Path(r"C:\torcs\torcs"))
    parser.add_argument("--runtime-home", type=Path, default=Path(".runtime") / "benchmark")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--track", default=None)
    parser.add_argument("--overwrite-runtime", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional RL dependency
        raise SystemExit("benchmark_native requires the 'rl' extra") from exc

    baseline_env = build_native_env(
        args.torcs_home,
        args.runtime_home / "baseline",
        max_steps=args.max_steps,
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
        max_steps=args.max_steps,
        track=args.track,
        overwrite_runtime=args.overwrite_runtime,
    )
    try:
        model = PPO.load(str(args.model), env=model_env)
        learned = evaluate_policy(model_env, model, episodes=args.episodes)
    finally:
        model_env.close()

    payload = {
        "track": args.track,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "baseline": baseline,
        "policy": learned,
        "comparison_rule": "learned policy must beat baseline on held-out runs; no claim from reward alone",
    }
    payload["manifest"] = build_run_manifest(
        args.torcs_home,
        role="benchmark",
        track=args.track,
        max_steps=args.max_steps,
        results={"baseline": baseline, "policy": learned},
    )
    if args.output is not None:
        print(f"benchmark artifact: {write_json_atomic(args.output, payload)}")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
