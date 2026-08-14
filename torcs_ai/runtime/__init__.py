"""Native TORCS runtime support.

The runtime package is deliberately dependency-light and has no import-time
side effects.  It only resolves and validates an existing TORCS installation;
starting a simulator is an explicit operation performed by a caller.
"""

from .config import TorcsInstallation, resolve_installation, resolve_torcs_home
from .manifest import InstallationManifest, inspect_installation
from .process import TorcsProcess, build_torcs_command
from .race_config import write_single_track_race_config
from .session import SessionConfig, TorcsSession
from .staging import stage_installation

__all__ = [
    "InstallationManifest",
    "TorcsInstallation",
    "TorcsProcess",
    "build_torcs_command",
    "inspect_installation",
    "resolve_torcs_home",
    "resolve_installation",
    "stage_installation",
    "TorcsSession",
    "SessionConfig",
    "write_single_track_race_config",
]
