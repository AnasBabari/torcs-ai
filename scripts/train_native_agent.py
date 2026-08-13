"""Train a PPO policy against an isolated native Windows TORCS runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torcs_ai.rl import (
    build_multi_track_env,
    build_native_env,
    build_run_manifest,
    train_ppo,
    write_json_atomic,
)
from torcs_ai.imitation import collect_expert_demonstrations


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
    parser.add_argument(
        "--track",
        action="append",
        dest="tracks",
        help="approved track; repeat to train on a seeded track matrix",
    )
    parser.add_argument("--overwrite-runtime", action="store_true")
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="stop a PPO optimizer epoch when policy KL drift exceeds this target",
    )
    parser.add_argument(
        "--teacher-guidance",
        type=float,
        default=0.0,
        help="optional expert-action reward coefficient in [0, 1]",
    )
    parser.add_argument("--expert-episodes", type=int, default=0)
    parser.add_argument("--expert-stride", type=int, default=2)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--simulator-timeout-seconds",
        type=float,
        default=30.0,
        help=(
            "maximum TORCS wait for the next action; training needs a longer "
            "deadline than inference while PPO performs gradient updates"
        ),
    )
    parser.add_argument(
        "--allow-smoke-training",
        action="store_true",
        help="permit deliberately short runs that cannot support competitiveness claims",
    )
    args = parser.parse_args()

    if args.simulator_timeout_seconds <= 0.0:
        parser.error("--simulator-timeout-seconds must be positive")
    if args.target_kl <= 0.0:
        parser.error("--target-kl must be positive")
    simulator_timeout_microseconds = int(
        args.simulator_timeout_seconds * 1_000_000
    )

    tracks = args.tracks or []
    track_count = max(1, len(dict.fromkeys(tracks)))
    if not args.allow_smoke_training:
        if args.max_steps < 5_000:
            parser.error("competitive training requires --max-steps >= 5000")
        if args.timesteps < args.max_steps * track_count:
            parser.error(
                "competitive training must cover at least one maximum-length "
                "episode per selected track; increase --timesteps or use "
                "--allow-smoke-training for a non-competitive smoke run"
            )
    if len(tracks) > 1:
        env = build_multi_track_env(
            args.torcs_home,
            args.runtime_home,
            tracks,
            max_steps=args.max_steps,
            overwrite_runtime=args.overwrite_runtime,
            teacher_guidance=args.teacher_guidance,
            simulator_timeout_microseconds=simulator_timeout_microseconds,
        )
        manifest_track = "matrix"
    else:
        track = tracks[0] if tracks else None
        env = build_native_env(
            args.torcs_home,
            args.runtime_home,
            max_steps=args.max_steps,
            track=track,
            overwrite_runtime=args.overwrite_runtime,
            teacher_guidance=args.teacher_guidance,
            simulator_timeout_microseconds=simulator_timeout_microseconds,
        )
        manifest_track = track
    try:
        demonstrations = None
        if args.expert_episodes:
            demonstrations = collect_expert_demonstrations(
                env,
                episodes_per_track=args.expert_episodes,
                sample_stride=args.expert_stride,
                seed=args.seed,
            )
        model = train_ppo(
            env,
            args.output,
            total_timesteps=args.timesteps,
            seed=args.seed,
            device=args.device,
            checkpoint_freq=args.checkpoint_freq,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
            demonstrations=demonstrations,
            bc_epochs=args.bc_epochs,
            bc_batch_size=args.bc_batch_size,
            bc_learning_rate=args.bc_learning_rate,
        )
        bc_summary = getattr(model, "_torcs_bc_summary", None)
        manifest = build_run_manifest(
            args.torcs_home,
            role="train",
            track=manifest_track,
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
                "target_kl": args.target_kl,
                "teacher_guidance": args.teacher_guidance,
                "tracks": tracks or [None],
                "device": args.device,
                "simulator_timeout_seconds": args.simulator_timeout_seconds,
                "model_path": str(args.output.with_suffix(".zip")),
                "expert_episodes": args.expert_episodes,
                "expert_stride": args.expert_stride,
                "behavior_cloning": bc_summary,
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
