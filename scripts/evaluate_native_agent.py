"""Evaluate a saved PPO policy on native TORCS and emit JSON summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torcs_ai.rl import (
    build_native_env,
    build_run_manifest,
    evaluate_policy,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torcs-home", type=Path, default=Path(r"C:\torcs\torcs"))
    parser.add_argument(
        "--runtime-home", type=Path, default=Path(".runtime") / "ppo-evaluation"
    )
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
        raise SystemExit("evaluate_native_agent requires the 'rl' extra") from exc

    env = build_native_env(
        args.torcs_home,
        args.runtime_home,
        max_steps=args.max_steps,
        track=args.track,
        overwrite_runtime=args.overwrite_runtime,
    )
    try:
        model = PPO.load(str(args.model), env=env)
        results = evaluate_policy(env, model, episodes=args.episodes)
        payload = {
            "results": results,
            "manifest": build_run_manifest(
                args.torcs_home,
                role="evaluate",
                track=args.track,
                max_steps=args.max_steps,
                results=results,
            ),
        }
        if args.output is not None:
            print(f"evaluation artifact: {write_json_atomic(args.output, payload)}")
        else:
            print(json.dumps(payload, indent=2))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
