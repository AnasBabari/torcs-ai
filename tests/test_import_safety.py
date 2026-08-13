"""Ensure package import does not train, connect, or create model files."""

import subprocess
import sys
from pathlib import Path


def test_package_import_is_side_effect_free(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(project_root)
    command = [
        sys.executable,
        "-c",
        "import torcs_ai; print(torcs_ai.__version__)",
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "2.1.0a1"
    assert list(tmp_path.iterdir()) == []


def test_main_module_import_is_side_effect_free(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(__import__("os").environ)
    environment["PYTHONPATH"] = str(project_root)
    command = [sys.executable, "-c", "import torcs_ai.main"]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []
