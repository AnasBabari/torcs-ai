"""Run a bounded native TORCS staging, SCR, and cleanup smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torcs_ai.client import Client  # noqa: E402
from torcs_ai.runtime import SessionConfig, TorcsSession  # noqa: E402
from torcs_ai.runtime.config import resolve_installation  # noqa: E402
from torcs_ai.runtime.race_config import write_single_track_race_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torcs-home", type=Path, default=None)
    parser.add_argument("--runtime-home", type=Path, default=None)
    parser.add_argument("--track", type=str, default="road/alpine-1")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")

    source = resolve_installation(args.torcs_home)
    runtime_home = (
        args.runtime_home
        or Path(__file__).resolve().parents[1] / ".runtime" / "native-smoke"
    )
    session = TorcsSession(
        source,
        runtime_home,
        config=SessionConfig(race_config=r"config\raceman\codex-race.xml"),
    )
    client: Client | None = None
    completed_steps = 0
    max_distance = 0.0
    try:
        manifest = session.prepare(overwrite=True)
        write_single_track_race_config(session.runtime_home, args.track)
        session.start()
        client = Client(
            host="127.0.0.1",
            port=3001,
            connect_attempts=25,
            connect_timeout=1.0,
        )
        for _ in range(args.steps):
            client.get_servers_input()
            if client.so is None:
                break
            state = client.S.d
            max_distance = max(max_distance, float(state.get("distRaced", 0.0)))
            steer = max(
                -1.0,
                min(
                    1.0,
                    float(state.get("angle", 0.0)) * 2.0
                    + float(state.get("trackPos", 0.0)) * 0.5,
                ),
            )
            client.R.d.update({"steer": steer, "accel": 0.25, "brake": 0.0, "gear": 1})
            client.respond_to_server()
            completed_steps += 1
    except (ConnectionError, TimeoutError, ValueError, OSError) as exc:
        payload = {"status": "failed", "error": str(exc), "steps": completed_steps}
        if args.as_json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(f"native smoke failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.shutdown()
        session.stop()

    payload = {
        "status": "ok",
        "track": args.track,
        "steps": completed_steps,
        "requested_steps": args.steps,
        "max_distance": max_distance,
        "executable_sha256": manifest.executable_sha256,
        "runtime_home": str(runtime_home),
        "process_clean": not session.running,
    }
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"native smoke passed: {completed_steps}/{args.steps} steps on {args.track}"
        )
        print(f"max distance: {max_distance:.3f}m")
        print(f"runtime: {runtime_home}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
