"""Configuration and path resolution for the native Windows TORCS runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

DEFAULT_TORCS_HOME = Path(r"C:\torcs\torcs")
TORCS_HOME_ENV = "TORCS_HOME"


class TorcsConfigurationError(ValueError):
    """Raised when a TORCS installation or runtime path is invalid."""


@dataclass(frozen=True)
class TorcsInstallation:
    """Resolved paths for an installed TORCS distribution.

    The object describes an installation only.  It never creates directories
    or changes files under ``home``.
    """

    home: Path

    @property
    def executable(self) -> Path:
        return self.home / "wtorcs.exe"

    @property
    def scr_server_dll(self) -> Path:
        return self.home / "drivers" / "scr_server" / "scr_server.dll"

    @property
    def scr_server_xml(self) -> Path:
        return self.home / "drivers" / "scr_server" / "scr_server.xml"

    @property
    def config_dir(self) -> Path:
        return self.home / "config"

    @property
    def tracks_dir(self) -> Path:
        return self.home / "tracks"

    @property
    def drivers_dir(self) -> Path:
        return self.home / "drivers"

    def track_path(self, category: str, name: str) -> Path:
        """Return a track path after rejecting path traversal."""

        for value, label in ((category, "category"), (name, "name")):
            if not value or Path(value).name != value or value in {".", ".."}:
                raise TorcsConfigurationError(f"Invalid track {label}: {value!r}")
        path = (self.tracks_dir / category / name).resolve()
        tracks_root = self.tracks_dir.resolve()
        if tracks_root not in path.parents:
            raise TorcsConfigurationError(
                "Track path escapes the TORCS tracks directory"
            )
        return path


def resolve_torcs_home(value: str | Path | None = None) -> Path:
    """Resolve the native TORCS home from an argument, env var, or default."""

    raw_value = value if value is not None else os.environ.get(TORCS_HOME_ENV)
    home = Path(raw_value) if raw_value else DEFAULT_TORCS_HOME
    return home.expanduser().resolve()


def resolve_installation(value: str | Path | None = None) -> TorcsInstallation:
    """Resolve a ``TorcsInstallation`` without performing filesystem writes."""

    return TorcsInstallation(resolve_torcs_home(value))
