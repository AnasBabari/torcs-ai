"""Compare a saved policy with deterministic center and tactical baselines across tracks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torcs_ai.rl import (  # noqa: E402
    build_native_env,
    build_run_manifest,
    compare_with_expert,
    evaluate_expert,
    evaluate_fixed_action,
    evaluate_policy,
    summarize_evaluation,
    write_json_atomic,
)


def _runtime_slug(track: str | None) -> str:
    """Return a filesystem-safe label for one allowlisted benchmark track."""
    label = track or "default"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")
    return slug or "default"


def benchmark_track(args: argparse.Namespace, track: str | None) -> dict[str, Any]:
    """Run the three-policy comparison for one isolated track runtime."""
    runtime_root = args.runtime_home / _runtime_slug(track)

    # 1. Deterministic Center Baseline (Action 4: center + hold)
    baseline_env = build_native_env(
        args.torcs_home,
        runtime_root / "baseline",
        max_steps=args.max_steps,
        track=track,
        overwrite_runtime=args.overwrite_runtime,
        teacher_guidance=0.0,
    )
    try:
        baseline = evaluate_fixed_action(baseline_env, episodes=args.episodes)
    finally:
        baseline_env.close()

    # 2. Deterministic Expert Tactical Teacher Baseline
    expert_env = build_native_env(
        args.torcs_home,
        runtime_root / "expert",
        max_steps=args.max_steps,
        track=track,
        overwrite_runtime=args.overwrite_runtime,
        teacher_guidance=0.0,
    )
    try:
        expert = evaluate_expert(expert_env, episodes=args.episodes)
    finally:
        expert_env.close()

    # 3. Learned PPO Policy
    model_env = build_native_env(
        args.torcs_home,
        runtime_root / "model",
        max_steps=args.max_steps,
        track=track,
        overwrite_runtime=args.overwrite_runtime,
        teacher_guidance=0.0,
    )
    try:
        from stable_baselines3 import PPO

        model = PPO.load(str(args.model), env=model_env)
        learned = evaluate_policy(model_env, model, episodes=args.episodes)
    finally:
        model_env.close()

    summaries = {
        "baseline": summarize_evaluation(baseline),
        "expert": summarize_evaluation(expert),
        "policy": summarize_evaluation(learned),
    }
    return {
        "track": track,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "baseline": baseline,
        "expert": expert,
        "policy": learned,
        "summaries": summaries,
        "competitiveness": compare_with_expert(
            summaries["policy"], summaries["expert"]
        ),
        "comparison_rule": (
            "Learned policy must beat the fixed baseline on held-out runs; "
            "never attribute deterministic controller performance to PPO."
        ),
    }


def render_benchmark_markdown(payload: dict[str, Any]) -> str:
    """Generate a clean GitHub-Flavored Markdown report table."""
    lines = [
        "# TORCS Racing Agent Benchmark Report",
        "",
        "| Track | Model / Controller | Finish Rate | Win Rate | Med Position | Damage/km | Shield/km | Dom Action % | Collapsed | p50 Latency (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    tracks_data: list[dict[str, Any]]
    if "tracks" in payload and isinstance(payload["tracks"], dict):
        tracks_data = list(payload["tracks"].values())
    else:
        tracks_data = [payload]

    for item in tracks_data:
        track_name = item.get("track") or "default"
        summaries = item.get("summaries", {})
        for role, name in [
            ("policy", "Learned PPO"),
            ("baseline", "Fixed Center"),
            ("expert", "Expert Teacher"),
        ]:
            s = summaries.get(role, {})
            finish = f"{s.get('finish_rate', 0.0) * 100:.0f}%"
            win = f"{s.get('win_rate', 0.0) * 100:.0f}%"
            pos = f"{s.get('median_race_position', 0):.1f}"
            dmg = f"{s.get('median_damage_per_km', 0.0):.1f}"
            shield = f"{s.get('median_shield_interventions_per_km', 0.0):.1f}"
            dom_action = f"{s.get('dominant_action_share', 0.0) * 100:.1f}%"
            collapsed = "YES" if s.get("action_collapsed", False) else "NO"
            lat = f"{s.get('inference_latency_p50_ms', 0.0):.2f}"
            lines.append(
                f"| `{track_name}` | **{name}** | {finish} | {win} | {pos} | {dmg} | {shield} | {dom_action} | {collapsed} | {lat} |"
            )

    lines.append("")
    lines.append("## Competitiveness Gate Hierarchy")
    for item in tracks_data:
        track_name = item.get("track") or "default"
        comp = item.get("competitiveness", {})
        gates = comp.get("gates", {})
        status = "PASSED" if comp.get("competitive", False) else "FAILED"
        lines.append(f"### Track: `{track_name}` ({status})")
        for gate_name, passed in gates.items():
            check_mark = "PASS" if passed else "FAIL"
            lines.append(f"- **{gate_name}**: {check_mark}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torcs-home", type=Path, default=Path(r"C:\torcs\torcs"))
    parser.add_argument(
        "--runtime-home", type=Path, default=Path(".runtime") / "benchmark"
    )
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
        checkpoint_path=args.model if args.model.is_file() else None,
    )
    markdown_report = render_benchmark_markdown(payload)

    if args.output is not None:
        json_path = write_json_atomic(args.output.with_suffix(".json"), payload)
        md_path = args.output.with_suffix(".md")
        md_path.write_text(markdown_report, encoding="utf-8")
        print(f"benchmark JSON artifact: {json_path}")
        print(f"benchmark Markdown artifact: {md_path}")
    else:
        print(json.dumps(payload, indent=2))
        print("\n" + markdown_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
