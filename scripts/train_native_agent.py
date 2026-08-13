"""Train a PPO policy against an isolated native Windows TORCS runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torcs_ai.rl import build_native_env, build_run_manifest, train_ppo, write_json_atomic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torcs-home", type=Path, default=Path(r"C:\torcs\torcs"))
    parser.add_argument(
        "--runtime-home", type=Path, default=Path(".runtime") / "ppo-training"
    )
    parser.add_argument("--output", type=Path, default=Path("runs") / "ppo_native")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--track", default=None)
    parser.add_argument("--overwrite-runtime", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--teacher-guidance",
        type=float,
        default=0.0,
        help="optional expert-action reward coefficient in [0, 1]",
    )
    args = parser.parse_args()

    env = build_native_env(
        args.torcs_home,
        args.runtime_home,
        max_steps=args.max_steps,
        track=args.track,
        overwrite_runtime=args.overwrite_runtime,
        teacher_guidance=args.teacher_guidance,
    )
    try:
        train_ppo(
            env,
            args.output,
            total_timesteps=args.timesteps,
            seed=args.seed,
            device=args.device,
            checkpoint_freq=args.checkpoint_freq,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        manifest = build_run_manifest(
            args.torcs_home,
            role="train",
            track=args.track,
            max_steps=args.max_steps,
            teacher_guidance=args.teacher_guidance,
            seed=args.seed,
            training={
                "algorithm": "PPO",
                "total_timesteps": args.timesteps,
                "checkpoint_freq": args.checkpoint_freq,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "teacher_guidance": args.teacher_guidance,
                "device": args.device,
                "model_path": str(args.output.with_suffix(".zip")),
            },
        )
        manifest_path = write_json_atomic(
            args.output.with_suffix(".manifest.json"), manifest
        )
        print(f"run manifest: {manifest_path}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
