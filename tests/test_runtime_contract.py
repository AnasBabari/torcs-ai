"""Read-only tests for the native Windows TORCS runtime contract."""

from pathlib import Path

import pytest

from torcs_ai.client import Client, DriverAction, ServerState
from torcs_ai.controllers import decode_tactical_action
from torcs_ai.rl import build_native_env
from torcs_ai.runtime.config import TorcsConfigurationError, TorcsInstallation
from torcs_ai.runtime.manifest import (
    TorcsInstallationError,
    inspect_installation,
)
from torcs_ai.runtime.process import build_torcs_command
from torcs_ai.runtime.staging import stage_installation


def test_all_tactical_actions_are_exactly_nine() -> None:
    intents = [decode_tactical_action(action) for action in range(9)]
    assert [intent.action_id for intent in intents] == list(range(9))
    assert all(-1.0 <= intent.lateral_target <= 1.0 for intent in intents)
    assert all(0.65 <= intent.speed_fraction <= 1.0 for intent in intents)


def test_tactical_action_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match=r"\[0, 8\]"):
        decode_tactical_action(9)


def test_torcs_command_is_argument_vector() -> None:
    command = build_torcs_command(
        Path(r"C:\torcs\torcs\wtorcs.exe"),
        race_config=r"config\raceman\quickrace.xml",
        no_damage=True,
    )
    assert command == [
        r"C:\torcs\torcs\wtorcs.exe",
        "-T",
        "-t",
        "100000",
        "-ver",
        "2010",
        "-nodamage",
        "-r",
        r"config\raceman\quickrace.xml",
    ]


def test_torcs_command_rejects_absolute_race_config() -> None:
    with pytest.raises(TorcsConfigurationError):
        build_torcs_command(Path("wtorcs.exe"), race_config=r"C:\race.xml")


def test_native_environment_rejects_nonpositive_simulator_timeout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="simulator_timeout_microseconds"):
        build_native_env(
            tmp_path / "missing-installation",
            tmp_path / "runtime",
            simulator_timeout_microseconds=0,
        )


def test_staging_rejects_source_descendant(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(TorcsConfigurationError):
        stage_installation(source, source / "staged")


def test_staging_rejects_source_ancestor(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(TorcsConfigurationError):
        stage_installation(source, tmp_path)


def test_staging_does_not_modify_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "wtorcs.exe"
    source_file.write_bytes(b"immutable")
    destination = tmp_path / "runtime"
    stage_installation(source, destination)
    assert (destination / "wtorcs.exe").read_bytes() == b"immutable"
    assert source_file.read_bytes() == b"immutable"


def test_inspection_requires_real_installation(tmp_path: Path) -> None:
    with pytest.raises(TorcsInstallationError):
        inspect_installation(TorcsInstallation(tmp_path / "missing"))


def test_server_state_rejects_malformed_packets_without_stale_fields() -> None:
    state = ServerState()
    state.parse_server_str("(speedX 10)(trackPos 0)")
    with pytest.raises(ValueError):
        state.parse_server_str("not-an-scr-packet")
    assert state.d == {"speedX": 10.0, "trackPos": 0.0}


def test_driver_action_never_sends_throttle_and_brake_together() -> None:
    action = DriverAction()
    action.d["accel"] = 1.0
    action.d["brake"] = 1.0
    action.clip_to_limits()
    assert not (action.d["accel"] > 0 and action.d["brake"] > 0)


def test_client_supports_non_connecting_protocol_fixture() -> None:
    client = Client(connect=False)
    assert client.so is None
    assert client.port == 3001


def test_client_rejects_nonpositive_telemetry_timeout() -> None:
    with pytest.raises(ValueError):
        Client(connect=False, telemetry_timeout=0)
