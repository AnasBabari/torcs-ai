"""Unit tests for the torcs-ai unified CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from torcs_ai.main import build_parser, main


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    assert parser.prog == "torcs-ai"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_doctor_subcommand() -> None:
    with patch("scripts.torcs_doctor.main", return_value=0) as mock_doc:
        ret = main(["doctor", "--json"])
        assert ret == 0
        mock_doc.assert_called_once_with(["--json"])


def test_cli_probe_subcommand() -> None:
    with patch("scripts.native_smoke.main", return_value=0) as mock_smoke:
        ret = main(["probe", "--steps", "50", "--track", "road/alpine-1"])
        assert ret == 0
        mock_smoke.assert_called_once_with(
            ["--steps", "50", "--track", "road/alpine-1"]
        )


def test_cli_report_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps({"role": "train", "status": "ok"}), encoding="utf-8"
    )

    ret = main(["report", "--manifest", str(manifest_file)])
    assert ret == 0
    captured = capsys.readouterr()
    assert '"role": "train"' in captured.out


def test_cli_report_missing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "non_existent.json"
    ret = main(["report", "--manifest", str(missing)])
    assert ret == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_cli_no_args_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "torcs-ai" in captured.out
