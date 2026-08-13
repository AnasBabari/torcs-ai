"""Safe staged race-config generation tests."""

from pathlib import Path

import pytest

from torcs_ai.runtime import write_single_track_race_config
from torcs_ai.runtime.config import TorcsConfigurationError


def _runtime(tmp_path: Path) -> Path:
    config = tmp_path / "config" / "raceman"
    config.mkdir(parents=True)
    (config / "quickrace.xml").write_text(
        '<section name="Tracks">\n'
        '  <section name="1">\n'
        '    <attstr name="name" val="corkscrew"/>\n'
        '    <attstr name="category" val="road"/>\n'
        "  </section>\n"
        "</section>\n",
        encoding="utf-8",
    )
    return tmp_path


def test_track_config_is_written_inside_runtime(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    relative = write_single_track_race_config(runtime, "road/alpine-1")
    output = runtime / Path(relative)
    assert output.is_file()
    contents = output.read_text(encoding="utf-8")
    assert 'name" val="alpine-1"' in contents
    assert 'category" val="road"' in contents


def test_track_config_rejects_unapproved_track(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    with pytest.raises(TorcsConfigurationError):
        write_single_track_race_config(runtime, "../outside")


def test_track_config_converts_only_interactive_player_to_scr_server(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    source = runtime / "config" / "raceman" / "quickrace.xml"
    source.write_text(
        source.read_text(encoding="utf-8")
        + '<attstr name="focused module" val="human"/>\n'
        + '<section name="1"><attnum name="idx" val="0"/>'
        + '<attstr name="module" val="human"/></section>\n'
        + '<section name="Drivers Start List"><section name="1">'
        + '<attstr name="module" val="human"/><attnum name="idx" val="0"/>'
        + '</section></section>\n'
        + '<section name="2"><attstr name="module" val="human"/></section>\n',
        encoding="utf-8",
    )
    relative = write_single_track_race_config(runtime, "road/forza")
    contents = (runtime / Path(relative)).read_text(encoding="utf-8")
    assert 'focused module" val="scr_server"' in contents
    assert contents.count('module" val="scr_server"') == 3
    assert '<section name="2"><attstr name="module" val="human"/></section>' in contents
