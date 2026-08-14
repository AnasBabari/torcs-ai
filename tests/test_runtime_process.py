"""Unit tests for TORCS process lifecycle, command building, and session management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from torcs_ai.runtime.config import TorcsInstallation
from torcs_ai.runtime.process import (
    ProcessConfig,
    TorcsProcess,
    build_torcs_command,
)
from torcs_ai.runtime.session import SessionConfig, TorcsSession


def test_build_torcs_command_options() -> None:
    exe = Path(r"C:\torcs\torcs\wtorcs.exe")
    cmd = build_torcs_command(
        exe,
        race_config=r"config\raceman\quickrace.xml",
        text_only=True,
        timeout_microseconds=50000,
        protocol_version="2010",
        no_damage=True,
        no_fuel=True,
        no_lap_time=True,
    )
    assert str(exe) in cmd
    assert "-T" in cmd
    assert "-t" in cmd
    assert "50000" in cmd
    assert "-ver" in cmd
    assert "2010" in cmd
    assert "-nodamage" in cmd
    assert "-nofuel" in cmd
    assert "-nolaptime" in cmd
    assert "-r" in cmd
    assert r"config\raceman\quickrace.xml" in cmd


def test_torcs_process_lifecycle(tmp_path: Path) -> None:
    config = ProcessConfig(startup_timeout_seconds=2.0, shutdown_timeout_seconds=2.0)
    proc = TorcsProcess(
        ["python", "-c", "import time; time.sleep(0.01)"], cwd=tmp_path, config=config
    )
    assert proc.pid is None
    assert not proc.running

    with patch("subprocess.Popen") as mock_popen:
        mock_handle = MagicMock()
        mock_handle.pid = 12345
        mock_handle.poll.return_value = None
        mock_popen.return_value = mock_handle

        pid = proc.start()
        assert pid == 12345
        assert proc.running
        proc.stop()
        mock_handle.terminate.assert_called()


def test_torcs_session_prepare_and_start(tmp_path: Path) -> None:
    # Setup mock source installation
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "wtorcs.exe").write_bytes(b"exe")
    (source_dir / "drivers" / "scr_server").mkdir(parents=True)
    (source_dir / "drivers" / "scr_server" / "scr_server.dll").write_bytes(b"dll")
    (source_dir / "drivers" / "scr_server" / "scr_server.xml").write_text(
        '<params name="SCR"><section name="index"><section name="1"/></section></params>',
        encoding="utf-8",
    )
    for cat, name in [
        ("road", "alpine-1"),
        ("road", "forza"),
        ("oval", "michigan"),
        ("road", "ruudskogen"),
        ("road", "spring"),
        ("road", "street-1"),
    ]:
        tdir = source_dir / "tracks" / cat / name
        tdir.mkdir(parents=True)
        (tdir / f"{name}.xml").write_text("<params/>", encoding="utf-8")
    for opp in ["berniw", "bt", "inferno", "olethros", "tita"]:
        (source_dir / "drivers" / opp).mkdir(parents=True)

    runtime_dir = tmp_path / "staged"
    session = TorcsSession(
        TorcsInstallation(source_dir),
        runtime_dir,
        config=SessionConfig(race_config=r"config\raceman\quickrace.xml"),
    )
    manifest = session.prepare(overwrite=True)
    assert manifest is not None
    assert (runtime_dir / "wtorcs.exe").is_file()

    with patch.object(TorcsProcess, "start", return_value=9999):
        pid = session.start()
        assert pid == 9999
        session.stop()
