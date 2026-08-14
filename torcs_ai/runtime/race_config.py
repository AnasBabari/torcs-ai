"""Generate allowlisted single-track race configs inside a staged runtime."""

from __future__ import annotations

import re
from pathlib import Path

from .config import TorcsConfigurationError
from .manifest import DEFAULT_TRACKS

_TRACK_BLOCK = re.compile(
    r'(?P<prefix><section name="Tracks">.*?<attstr name="name" val=")'
    r'(?P<name>[^"]+)'
    r'(?P<middle>"\s*/>.*?<attstr name="category" val=")'
    r'(?P<category>[^"]+)"\s*/>',
    flags=re.DOTALL,
)
_FOCUSED_HUMAN = re.compile(
    r'(?P<prefix><attstr\s+name="focused module"\s+val=")human("\s*/>)'
)
_PLAYER_DRIVER = re.compile(
    r'(?P<prefix><section\s+name="1">\s*'
    r'<attnum\s+name="idx"\s+val="0"\s*/>\s*'
    r'<attstr\s+name="module"\s+val=")human(?P<suffix>"\s*/>)',
    flags=re.DOTALL,
)
_PLAYER_START = re.compile(
    r'(?P<prefix><section\s+name="1">\s*'
    r'<attstr\s+name="module"\s+val=")human(?P<suffix>"\s*/>\s*'
    r'<attnum\s+name="idx"\s+val="0"\s*/>)',
    flags=re.DOTALL,
)


def write_single_track_race_config(
    runtime_home: Path,
    track: str,
    *,
    base_config: str = r"config\raceman\quickrace.xml",
    output_config: str = r"config\raceman\codex-race.xml",
) -> str:
    """Write one approved track selection and return its relative path."""

    approved_tracks = {f"{category}/{name}" for category, name in DEFAULT_TRACKS}
    if track not in approved_tracks:
        raise TorcsConfigurationError(
            f"track {track!r} is not in the approved benchmark track set"
        )
    runtime_home = runtime_home.expanduser().resolve()
    source = (runtime_home / Path(base_config.replace("\\", "/"))).resolve()
    destination = (runtime_home / Path(output_config.replace("\\", "/"))).resolve()
    if runtime_home not in source.parents or runtime_home not in destination.parents:
        raise TorcsConfigurationError(
            "race config must remain inside the staged runtime"
        )
    if not source.is_file():
        raise TorcsConfigurationError(f"base race config does not exist: {source}")
    category, name = track.split("/", 1)
    contents = source.read_text(encoding="utf-8")
    replacement_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacement_count
        replacement_count += 1
        return f'{match.group("prefix")}{name}{match.group("middle")}{category}" />'

    contents = _TRACK_BLOCK.sub(replace, contents, count=1)
    if replacement_count != 1:
        raise TorcsConfigurationError("base race config has no unambiguous track block")
    # The source installation may be configured for interactive human play.
    # A generated single-track config is exclusively for the owned SCR client,
    # so convert only the focused index-0 driver and leave opponent entries
    # untouched.  This keeps the installed quick-race menu human-playable.
    contents = _FOCUSED_HUMAN.sub(r"\g<prefix>scr_server\2", contents, count=1)
    contents = _PLAYER_DRIVER.sub(r"\g<prefix>scr_server\g<suffix>", contents, count=1)
    contents = _PLAYER_START.sub(r"\g<prefix>scr_server\g<suffix>", contents, count=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(contents, encoding="utf-8", newline="\n")
    # Return a path that composes correctly on both Windows and POSIX callers.
    # The native command builder may still receive the platform's separator
    # when the staged runtime is launched.
    return str(Path(output_config.replace("\\", "/")))
