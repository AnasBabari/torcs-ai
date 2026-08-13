"""Transport lifecycle tests without launching the native simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from torcs_ai.envs.torcs_transport import ScrTransportConfig, TorcsScrTransport


class _FakeSocket:
    pass


class _FakeState:
    def __init__(self) -> None:
        self.d = {"speedX": 10.0}


class _FakeAction:
    def __init__(self) -> None:
        self.d: dict[str, Any] = {}

    def clip_to_limits(self) -> None:
        return None


class _FakeClient:
    def __init__(self, **kwargs: Any) -> None:
        self.so = _FakeSocket()
        self.S = _FakeState()
        self.R = _FakeAction()
        self.sent = False
        self.closed = False

    def get_servers_input(self) -> None:
        self.S.d = {"speedX": 11.0}

    def respond_to_server(self) -> None:
        self.sent = True

    def shutdown(self) -> None:
        self.closed = True
        self.so = None


class _FakeSession:
    def __init__(self) -> None:
        self.running = False
        self.started = 0
        self.stopped = 0

    def start(self) -> int:
        self.running = True
        self.started += 1
        return 123

    def stop(self) -> None:
        self.running = False
        self.stopped += 1


def test_config_rejects_invalid_port() -> None:
    with pytest.raises(ValueError):
        ScrTransportConfig(port=0)


def test_transport_owns_client_and_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession()
    transport = TorcsScrTransport(session)  # type: ignore[arg-type]
    monkeypatch.setattr(
        "torcs_ai.envs.torcs_transport.Client", _FakeClient
    )

    initial = transport.reset()
    assert initial == {"speedX": 11.0}
    assert session.started == 1
    next_state = transport.step({"steer": 0.2, "accel": 0.5, "brake": 0.0, "gear": 2})
    assert next_state == {"speedX": 11.0}
    assert transport.client is not None and transport.client.R.d["gear"] == 2
    transport.close()
    assert session.stopped == 1


def test_transport_requires_reset() -> None:
    session = _FakeSession()
    transport = TorcsScrTransport(session)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        transport.step({"steer": 0.0})
