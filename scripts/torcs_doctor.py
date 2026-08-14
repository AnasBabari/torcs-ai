"""Validate the native Windows TORCS installation used by local runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permit direct ``python scripts/torcs_doctor.py`` usage before the package is
# installed in editable mode.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torcs_ai.runtime import inspect_installation
from torcs_ai.runtime.config import resolve_installation
from torcs_ai.runtime.manifest import TorcsInstallationError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torcs-home", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        installation = resolve_installation(args.torcs_home)
        manifest = inspect_installation(installation)
    except (OSError, ValueError, TorcsInstallationError) as exc:
        err_payload: dict[str, object] = {"status": "failed", "error": str(exc)}
        if args.as_json:
            print(json.dumps(err_payload, sort_keys=True))
        else:
            print(f"TORCS doctor failed: {exc}", file=sys.stderr)
        return 1

    payload: dict[str, object] = {"status": "ok", "manifest": manifest.to_dict()}
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"TORCS installation: {manifest.home}")
        print(f"Executable SHA-256: {manifest.executable_sha256}")
        print(f"SCR slots: {manifest.scr_driver_slots}")
        print(f"Tracks validated: {len(manifest.tracks)}")
        print(f"Opponents validated: {', '.join(manifest.opponents)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
