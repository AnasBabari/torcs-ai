"""Owned native SCR transport for the Gymnasium racing environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..client import Client
from ..runtime.session import TorcsSession


@dataclass(frozen=True)
class ScrTransportConfig:
    """Bounded connection settings for one native SCR slot."""

    host: str = "localhost"
    port: int = 3001
    connect_attempts: int = 20
    connect_timeout_seconds: float = 0.5
    telemetry_timeout_seconds: float = 5.0
    max_steps: int = 100_000

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("host cannot be empty")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be in [1, 65535]")
        if (
            self.connect_attempts < 1
            or self.connect_timeout_seconds <= 0
            or self.telemetry_timeout_seconds <= 0
        ):
            raise ValueError("connection attempts and timeouts must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")


class TorcsScrTransport:
    """Translate environment controls to one owned native TORCS SCR client.

    The transport owns only the process supplied by ``TorcsSession`` and the
    UDP client on the configured port. It never searches for or terminates
    unrelated TORCS processes. A new client is created for each reset so a
    completed SCR race cannot leak socket state into the next episode.
    """

    def __init__(
        self,
        session: TorcsSession,
        *,
        config: ScrTransportConfig = ScrTransportConfig(),
    ) -> None:
        self.session = session
        self.config = config
        self.client: Optional[Client] = None
        self._started_session = False
        self._steps = 0

    def _ensure_session(self) -> None:
        if not self.session.running:
            self.session.start()
            self._started_session = True

    def _new_client(self) -> Client:
        return Client(
            host=self.config.host,
            port=self.config.port,
            max_steps=self.config.max_steps,
            connect_attempts=self.config.connect_attempts,
            connect_timeout=self.config.connect_timeout_seconds,
            telemetry_timeout=self.config.telemetry_timeout_seconds,
        )

    @staticmethod
    def _snapshot(client: Client) -> dict[str, Any]:
        return dict(client.S.d)

    def reset(self, *, seed: Optional[int] = None) -> Mapping[str, Any]:
        """Start/reconnect one SCR client and return the first telemetry packet."""

        del seed  # TORCS itself is seeded by the selected race configuration.
        self._close_client()
        self._ensure_session()
        try:
            self.client = self._new_client()
            self.client.get_servers_input()
        except Exception as first_error:
            self._close_client()
            # A completed native race can leave the owned TORCS process alive
            # while its SCR listener has stopped accepting clients.  Restart
            # only a session started by this transport, then make one bounded
            # reconnect attempt; never search for or terminate external PIDs.
            if not self._started_session:
                raise
            self.session.stop()
            self._started_session = False
            try:
                self._ensure_session()
                self.client = self._new_client()
                self.client.get_servers_input()
            except Exception as retry_error:
                self._close_client()
                raise first_error from retry_error
        self._steps = 0
        return self._snapshot(self.client)

    def step(self, controls: Mapping[str, float]) -> Mapping[str, Any]:
        """Send one bounded control packet and receive the next telemetry."""

        if self.client is None or self.client.so is None:
            raise RuntimeError("reset must be called before step")
        if self._steps >= self.config.max_steps:
            raise RuntimeError("maximum transport steps reached; reset required")

        self.client.R.d.update(
            {
                "steer": float(controls.get("steer", 0.0)),
                "accel": float(controls.get("accel", 0.0)),
                "brake": float(controls.get("brake", 0.0)),
                "gear": int(round(float(controls.get("gear", 1.0)))),
                "clutch": float(controls.get("clutch", 0.0)),
            }
        )
        self.client.R.clip_to_limits()
        self.client.respond_to_server()
        self.client.get_servers_input()
        self._steps += 1
        snapshot = self._snapshot(self.client)
        if self.client.so is None:
            # Client shuts its socket on an SCR shutdown packet. Preserve the
            # final state and let RacingEnv's terminal check close the episode.
            snapshot["raceFinished"] = True
        return snapshot

    def _close_client(self) -> None:
        if self.client is not None:
            self.client.shutdown()
            self.client = None

    def close(self) -> None:
        self._close_client()
        if self._started_session:
            self.session.stop()
            self._started_session = False

    def __enter__(self) -> "TorcsScrTransport":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
