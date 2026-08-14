"""Safe preparation of an isolated, mutable TORCS runtime copy."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from .config import TorcsConfigurationError


def stage_installation(
    source: Path,
    destination: Path,
    *,
    overwrite: bool = False,
    ignore_names: set[str] | None = None,
) -> Path:
    """Copy a TORCS installation into a private runtime directory.

    The source is never removed or modified.  Existing destinations are
    rejected unless ``overwrite`` is explicitly requested, and the destination
    cannot be the source, an ancestor, or a descendant of it.
    """

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise TorcsConfigurationError(
            f"TORCS source directory does not exist: {source}"
        )
    if (
        destination == source
        or source in destination.parents
        or destination in source.parents
    ):
        raise TorcsConfigurationError(
            "staging destination cannot contain or be contained by the source installation"
        )
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"staging destination already exists: {destination}")
        if not destination.is_dir():
            raise TorcsConfigurationError(
                f"staging destination is not a directory: {destination}"
            )
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ignored = set(ignore_names or ())

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in ignored}

    shutil.copytree(source, destination, copy_function=shutil.copy2, ignore=ignore)
    return destination
