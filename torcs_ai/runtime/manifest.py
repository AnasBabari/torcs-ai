"""Read-only inspection and fingerprinting of a TORCS installation."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .config import TorcsInstallation, TorcsConfigurationError

DEFAULT_TRACKS: tuple[tuple[str, str], ...] = (
    ("road", "alpine-1"),
    ("road", "forza"),
    ("oval", "michigan"),
    ("road", "ruudskogen"),
    ("road", "spring"),
    ("road", "street-1"),
)
DEFAULT_OPPONENTS: tuple[str, ...] = ("berniw", "bt", "inferno", "olethros", "tita")


class TorcsInstallationError(RuntimeError):
    """Raised when the native installation cannot satisfy the runtime contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise TorcsInstallationError(f"Missing {label}: {path}")


def _require_track(installation: TorcsInstallation, category: str, name: str) -> Path:
    path = installation.track_path(category, name)
    if not path.is_dir():
        raise TorcsInstallationError(f"Missing track directory: {path}")
    xml_files = tuple(path.glob("*.xml"))
    if not xml_files:
        raise TorcsInstallationError(f"Track has no XML definition: {path}")
    return xml_files[0]


def _count_scr_slots(path: Path) -> int:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise TorcsInstallationError(f"Cannot parse SCR configuration {path}: {exc}") from exc
    slots = root.findall(".//section[@name='index']/section")
    return len(slots)


@dataclass(frozen=True)
class InstallationManifest:
    """Immutable evidence describing a validated TORCS installation."""

    home: str
    executable: str
    executable_sha256: str
    scr_server_dll_sha256: str
    scr_server_xml_sha256: str
    protocol_version: str
    scr_driver_slots: int
    tracks: Mapping[str, str]
    opponents: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "home": self.home,
            "executable": self.executable,
            "executable_sha256": self.executable_sha256,
            "scr_server_dll_sha256": self.scr_server_dll_sha256,
            "scr_server_xml_sha256": self.scr_server_xml_sha256,
            "protocol_version": self.protocol_version,
            "scr_driver_slots": self.scr_driver_slots,
            "tracks": dict(self.tracks),
            "opponents": list(self.opponents),
        }


def inspect_installation(
    installation: TorcsInstallation,
    tracks: Sequence[tuple[str, str]] = DEFAULT_TRACKS,
    opponents: Sequence[str] = DEFAULT_OPPONENTS,
    protocol_version: str = "2010",
) -> InstallationManifest:
    """Validate required native assets and return checksum evidence.

    This function is strictly read-only.  It intentionally fails closed when
    a required track, SCR slot, executable, or opponent is absent.
    """

    home = installation.home
    if not home.is_dir():
        raise TorcsInstallationError(f"TORCS home does not exist: {home}")

    _require_file(installation.executable, "TORCS executable")
    _require_file(installation.scr_server_dll, "SCR server DLL")
    _require_file(installation.scr_server_xml, "SCR server XML")

    slot_count = _count_scr_slots(installation.scr_server_xml)
    if slot_count < 1:
        raise TorcsInstallationError("SCR server configuration defines no driver slots")

    track_evidence: dict[str, str] = {}
    for category, name in tracks:
        xml_path = _require_track(installation, category, name)
        track_evidence[f"{category}/{name}"] = sha256_file(xml_path)

    missing_opponents = [
        name
        for name in opponents
        if not (installation.drivers_dir / name).is_dir()
    ]
    if missing_opponents:
        raise TorcsInstallationError(
            "Missing opponent modules: " + ", ".join(sorted(missing_opponents))
        )

    return InstallationManifest(
        home=str(home),
        executable=str(installation.executable),
        executable_sha256=sha256_file(installation.executable),
        scr_server_dll_sha256=sha256_file(installation.scr_server_dll),
        scr_server_xml_sha256=sha256_file(installation.scr_server_xml),
        protocol_version=protocol_version,
        scr_driver_slots=slot_count,
        tracks=track_evidence,
        opponents=tuple(opponents),
    )
