"""Explicit native TORCS staging and process session."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import TorcsInstallation
from .manifest import InstallationManifest, inspect_installation
from .process import ProcessConfig, TorcsProcess, build_torcs_command
from .staging import stage_installation


@dataclass(frozen=True)
class SessionConfig:
    race_config: str = r"config\raceman\quickrace.xml"
    protocol_version: str = "2010"
    timeout_microseconds: int = 100_000
    text_only: bool = True
    startup_timeout_seconds: float = 15.0
    shutdown_timeout_seconds: float = 5.0


class TorcsSession:
    """Own one isolated copy of TORCS and the process launched from it."""

    def __init__(
        self,
        source: TorcsInstallation,
        runtime_home: Path,
        *,
        config: SessionConfig = SessionConfig(),
    ) -> None:
        self.source = source
        self.runtime_home = runtime_home.expanduser().resolve()
        self.config = config
        self.manifest: InstallationManifest | None = None
        self.process: TorcsProcess | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.running

    def prepare(self, *, overwrite: bool = False) -> InstallationManifest:
        """Stage and validate an isolated runtime, without starting TORCS."""

        stage_installation(self.source.home, self.runtime_home, overwrite=overwrite)
        staged = TorcsInstallation(self.runtime_home)
        self.manifest = inspect_installation(staged)
        return self.manifest

    def start(self) -> int:
        """Start the staged executable using an argument vector and owned PID."""

        if self.manifest is None:
            self.manifest = inspect_installation(TorcsInstallation(self.runtime_home))
        if self.running:
            raise RuntimeError("TORCS session is already running")
        installation = TorcsInstallation(self.runtime_home)
        command = build_torcs_command(
            installation.executable,
            race_config=self.config.race_config,
            text_only=self.config.text_only,
            timeout_microseconds=self.config.timeout_microseconds,
            protocol_version=self.config.protocol_version,
        )
        self.process = TorcsProcess(
            command,
            cwd=self.runtime_home,
            config=ProcessConfig(
                startup_timeout_seconds=self.config.startup_timeout_seconds,
                shutdown_timeout_seconds=self.config.shutdown_timeout_seconds,
            ),
            log_path=self.runtime_home / "runtime.log",
        )
        return self.process.start()

    def stop(self) -> None:
        if self.process is not None:
            self.process.stop()
