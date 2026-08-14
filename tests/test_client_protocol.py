"""Unit tests for the low-level SCR client, packet parsing, and driver action formatting."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from torcs_ai.client import Client, DriverAction, ServerState, bargraph, destringify


def test_destringify_scalar_and_lists() -> None:
    assert destringify(["123"]) == 123.0
    assert destringify(["12.5"]) == 12.5
    assert destringify(["abc"]) == "abc"
    assert destringify(["1.0", "2.0", "3.0"]) == [1.0, 2.0, 3.0]
    assert destringify([]) == []


def test_bargraph_formatting() -> None:
    bg = bargraph(50.0, 0.0, 100.0, 10, "#")
    assert bg.startswith("[") and bg.endswith("]")
    assert "#" in bg
    assert "_" in bg


def test_server_state_fancyout() -> None:
    state = ServerState()
    raw = (
        "(angle 0.12)"
        "(trackPos -0.4)"
        "(speedX 120.5)"
        "(speedY 2.0)"
        "(speedZ 0.1)"
        "(gear 3)"
        "(damage 150)"
        "(fuel 85)"
        "(rpm 5500)"
        "(track 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190)"
        "(opponents 5 10 20 50 100 200)"
    )
    state.parse_server_str(raw)
    out = state.fancyout()
    assert "angle" in out
    assert "trackPos" in out
    assert "speedX" in out
    assert repr(state) == out


def test_driver_action_fancyout_and_clip() -> None:
    action = DriverAction()
    action.d["steer"] = 0.5
    action.d["accel"] = 0.8
    action.d["brake"] = 0.0
    action.d["gear"] = 2
    action.clip_to_limits()
    out = action.fancyout()
    assert "steer" in out
    assert "accel" in out
    assert "(steer 0.500)" in repr(action)


def test_client_init_and_shutdown() -> None:
    client = Client(connect=False)
    assert client.port == 3001
    assert client.so is None
    client.shutdown()  # safe when None


def test_client_mock_identify_and_step() -> None:
    client = Client(connect=False)
    mock_socket = MagicMock()
    mock_socket.recvfrom.return_value = (b"***identified***", ("127.0.0.1", 3001))
    client.so = mock_socket

    with patch("socket.socket", return_value=mock_socket):
        client.setup_connection()
        mock_socket.sendto.assert_called()

    # Test receiving telemetry
    mock_socket.recvfrom.return_value = (
        b"(speedX 100)(trackPos 0)",
        ("127.0.0.1", 3001),
    )
    client.get_servers_input()
    assert client.S.d.get("speedX") == 100.0

    # Test sending response
    client.respond_to_server()
    mock_socket.sendto.assert_called()
    client.shutdown()
