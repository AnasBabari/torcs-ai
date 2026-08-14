"""Unit tests for atomic checkpoint saving, SHA-256 integrity, and manifest generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from torcs_ai.rl import (
    build_run_manifest,
    save_checkpoint_atomic,
    write_json_atomic,
)


def test_write_json_atomic(tmp_path: Path) -> None:
    target = tmp_path / "subdir" / "test.json"
    payload = {"status": "ok", "value": 123}
    written = write_json_atomic(target, payload)
    assert written == target
    assert target.is_file()
    # Check that temp file does not remain
    assert not (tmp_path / "subdir" / ".test.json.tmp").exists()


def test_save_checkpoint_atomic(tmp_path: Path) -> None:
    output = tmp_path / "checkpoints" / "model"
    mock_model = MagicMock()

    def fake_save(path_str: str) -> None:
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"dummy zip content for testing")

    mock_model.save.side_effect = fake_save
    zip_path, digest = save_checkpoint_atomic(mock_model, output)

    assert zip_path == tmp_path / "checkpoints" / "model.zip"
    assert zip_path.is_file()
    assert len(digest) == 64  # Hex sha256

    sha_file = tmp_path / "checkpoints" / "model.zip.sha256"
    assert sha_file.is_file()
    assert digest in sha_file.read_text(encoding="utf-8")


def test_manifest_metadata_generation(tmp_path: Path) -> None:
    # Setup dummy torcs installation
    torcs_dir = tmp_path / "torcs"
    torcs_dir.mkdir()
    (torcs_dir / "wtorcs.exe").write_bytes(b"exe")
    (torcs_dir / "drivers" / "scr_server").mkdir(parents=True)
    (torcs_dir / "drivers" / "scr_server" / "scr_server.dll").write_bytes(b"dll")
    (torcs_dir / "drivers" / "scr_server" / "scr_server.xml").write_text(
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
        tdir = torcs_dir / "tracks" / cat / name
        tdir.mkdir(parents=True)
        (tdir / f"{name}.xml").write_text("<params/>", encoding="utf-8")
    for opp in ["berniw", "bt", "inferno", "olethros", "tita"]:
        (torcs_dir / "drivers" / opp).mkdir(parents=True)

    dummy_chk = tmp_path / "model.zip"
    dummy_chk.write_bytes(b"checkpoint-binary")

    manifest = build_run_manifest(
        torcs_dir,
        role="train",
        track="road/alpine-1",
        max_steps=5000,
        seed=42,
        checkpoint_path=dummy_chk,
    )

    assert manifest["role"] == "train"
    assert manifest["seed"] == 42
    assert manifest["environment"]["observation_schema"] == "competitive-telemetry-v1"
    assert manifest["environment"]["action_schema"] == "tactical-9-v1"
    assert (
        manifest["environment"]["reward_schema"]
        == "progress-position-safety-teacher-v3"
    )
    assert "checkpoint" in manifest
    assert (
        manifest["checkpoint"]["sha256"]
        == hashlib.sha256(b"checkpoint-binary").hexdigest()
    )
    assert manifest["environment"]["contaminated_held_out"] is False
