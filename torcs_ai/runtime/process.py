"""Owned native TORCS process lifecycle helpers."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Optional

from .config import TorcsConfigurationError


def build_torcs_command(
    executable: Path,
    *,
    race_config: str | None = None,
    text_only: bool = True,
    timeout_microseconds: int = 100_000,
    protocol_version: str = "2010",
    no_damage: bool = False,
    no_fuel: bool = False,
    no_lap_time: bool = False,
) -> list[str]:
    """Build an argument vector for the installed Windows TORCS binary."""

    if timeout_microseconds <= 0:
        raise TorcsConfigurationError("TORCS timeout must be positive")
    if not protocol_version.strip():
        raise TorcsConfigurationError("TORCS protocol version cannot be empty")

    command = [str(executable)]
    if text_only:
        command.append("-T")
    command.extend(["-t", str(timeout_microseconds), "-ver", protocol_version])
    if no_damage:
        command.append("-nodamage")
    if no_fuel:
        command.append("-nofuel")
    if no_lap_time:
        command.append("-nolaptime")
    if race_config is not None:
        if (
            not race_config.strip()
            or Path(race_config).is_absolute()
            or PureWindowsPath(race_config).is_absolute()
        ):
            raise TorcsConfigurationError(
                "race_config must be a non-empty relative path"
            )
        command.extend(["-r", race_config])
    return command


@dataclass(frozen=True)
class ProcessConfig:
    startup_timeout_seconds: float = 15.0
    shutdown_timeout_seconds: float = 5.0


class TorcsProcess:
    """Own exactly one child process and never terminate unrelated processes."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str] | None = None,
        config: ProcessConfig = ProcessConfig(),
        log_path: Path | None = None,
    ) -> None:
        if not command:
            raise TorcsConfigurationError("TORCS command cannot be empty")
        if not cwd.is_dir():
            raise TorcsConfigurationError(
                f"TORCS working directory does not exist: {cwd}"
            )
        if config.startup_timeout_seconds <= 0 or config.shutdown_timeout_seconds <= 0:
            raise TorcsConfigurationError("Process timeouts must be positive")
        self.command = tuple(str(value) for value in command)
        self.cwd = cwd
        self.environment = dict(environment) if environment is not None else None
        self.config = config
        self.log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> int:
        if self._process is not None:
            raise RuntimeError("TORCS process has already been started")
        environment = os.environ.copy()
        if self.environment is not None:
            environment.update(self.environment)
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        stdout: Any = subprocess.DEVNULL
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("ab")
            stdout = self._log_handle
        try:
            self._process = subprocess.Popen(
                list(self.command),
                cwd=str(self.cwd),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                shell=False,
                creationflags=creation_flags,
            )
        except Exception:
            self._close_log_handle()
            raise
        return self._process.pid

    def wait_for_exit(self, timeout_seconds: float | None = None) -> int:
        if self._process is None:
            raise RuntimeError("TORCS process has not been started")
        timeout = timeout_seconds or self.config.startup_timeout_seconds
        try:
            return self._process.wait(timeout=timeout)
        finally:
            self._close_log_handle()

    def _close_log_handle(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def stop(self) -> None:
        """Stop only the child process created by this instance."""

        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self.config.shutdown_timeout_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=self.config.shutdown_timeout_seconds)
        finally:
            self._close_log_handle()

    def __enter__(self) -> TorcsProcess:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
