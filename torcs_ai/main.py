"""TORCS AI - Unified CLI entry point for research tooling and runtime verification."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("torcs_ai")


def cmd_doctor(args: argparse.Namespace) -> int:
    from scripts.torcs_doctor import main as doctor_main

    argv = []
    if args.torcs_home:
        argv.extend(["--torcs-home", str(args.torcs_home)])
    if args.as_json:
        argv.append("--json")
    return doctor_main(argv)


def cmd_probe(args: argparse.Namespace) -> int:
    from scripts.native_smoke import main as smoke_main

    argv = ["--steps", str(args.steps)]
    if args.torcs_home:
        argv.extend(["--torcs-home", str(args.torcs_home)])
    if args.runtime_home:
        argv.extend(["--runtime-home", str(args.runtime_home)])
    if args.track:
        argv.extend(["--track", str(args.track)])
    if args.as_json:
        argv.append("--json")
    return smoke_main(argv)


def cmd_train(args: argparse.Namespace) -> int:
    from scripts.train_native_agent import main as train_main

    argv = [
        "--timesteps",
        str(args.timesteps),
        "--max-steps",
        str(args.max_steps),
        "--output",
        str(args.output),
        "--seed",
        str(args.seed),
    ]
    if args.torcs_home:
        argv.extend(["--torcs-home", str(args.torcs_home)])
    if args.tracks:
        for t in args.tracks:
            argv.extend(["--track", t])
    if args.teacher_guidance > 0.0:
        argv.extend(["--teacher-guidance", str(args.teacher_guidance)])
    if args.expert_episodes > 0:
        argv.extend(["--expert-episodes", str(args.expert_episodes)])
    if args.allow_smoke_training:
        argv.append("--allow-smoke-training")
    if args.allow_held_out_training:
        argv.append("--allow-held-out-training")
    sys.argv = [sys.argv[0]] + argv
    return train_main()


def cmd_evaluate(args: argparse.Namespace) -> int:
    from scripts.evaluate_native_agent import main as eval_main

    argv = [
        "--model",
        str(args.model),
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
    ]
    if args.torcs_home:
        argv.extend(["--torcs-home", str(args.torcs_home)])
    if args.track:
        argv.extend(["--track", str(args.track)])
    sys.argv = [sys.argv[0]] + argv
    return eval_main()


def cmd_benchmark(args: argparse.Namespace) -> int:
    from scripts.benchmark_native import main as bench_main

    argv = [
        "--model",
        str(args.model),
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
    ]
    if args.torcs_home:
        argv.extend(["--torcs-home", str(args.torcs_home)])
    if args.tracks:
        for t in args.tracks:
            argv.extend(["--track", t])
    if args.output:
        argv.extend(["--output", str(args.output)])
    sys.argv = [sys.argv[0]] + argv
    return bench_main()


def cmd_report(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"Error: manifest file not found at {manifest_path}", file=sys.stderr)
        return 1
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torcs-ai",
        description="Hierarchical Reinforcement Learning Racing Agent for native TORCS",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # doctor
    p_doc = subparsers.add_parser("doctor", help="Verify native TORCS installation")
    p_doc.add_argument("--torcs-home", type=Path, default=None)
    p_doc.add_argument("--json", action="store_true", dest="as_json")
    p_doc.set_defaults(func=cmd_doctor)

    # probe
    p_probe = subparsers.add_parser(
        "probe", help="Run bounded native staging and SCR handshake probe"
    )
    p_probe.add_argument("--torcs-home", type=Path, default=None)
    p_probe.add_argument("--runtime-home", type=Path, default=None)
    p_probe.add_argument("--track", type=str, default="road/alpine-1")
    p_probe.add_argument("--steps", type=int, default=500)
    p_probe.add_argument("--json", action="store_true", dest="as_json")
    p_probe.set_defaults(func=cmd_probe)

    # train
    p_train = subparsers.add_parser(
        "train", help="Train PPO policy against native TORCS"
    )
    p_train.add_argument("--torcs-home", type=Path, default=None)
    p_train.add_argument("--output", type=Path, default=Path("runs") / "ppo_native")
    p_train.add_argument("--timesteps", type=int, default=100_000)
    p_train.add_argument("--max-steps", type=int, default=15_000)
    p_train.add_argument("--seed", type=int, default=7)
    p_train.add_argument("--track", action="append", dest="tracks")
    p_train.add_argument("--teacher-guidance", type=float, default=0.0)
    p_train.add_argument("--expert-episodes", type=int, default=0)
    p_train.add_argument("--allow-smoke-training", action="store_true")
    p_train.add_argument("--allow-held-out-training", action="store_true")
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser(
        "evaluate", help="Evaluate trained policy on native track"
    )
    p_eval.add_argument("--torcs-home", type=Path, default=None)
    p_eval.add_argument("--model", type=Path, required=True)
    p_eval.add_argument("--track", type=str, default=None)
    p_eval.add_argument("--episodes", type=int, default=3)
    p_eval.add_argument("--max-steps", type=int, default=15_000)
    p_eval.set_defaults(func=cmd_evaluate)

    # benchmark
    p_bench = subparsers.add_parser(
        "benchmark",
        help="Run multi-track benchmark comparing PPO, baseline, and expert",
    )
    p_bench.add_argument("--torcs-home", type=Path, default=None)
    p_bench.add_argument("--model", type=Path, required=True)
    p_bench.add_argument("--track", action="append", dest="tracks")
    p_bench.add_argument("--episodes", type=int, default=3)
    p_bench.add_argument("--max-steps", type=int, default=15_000)
    p_bench.add_argument("--output", type=Path, default=None)
    p_bench.set_defaults(func=cmd_benchmark)

    # report
    p_rep = subparsers.add_parser(
        "report", help="Inspect run manifest and evaluation report"
    )
    p_rep.add_argument("--manifest", type=Path, required=True)
    p_rep.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
