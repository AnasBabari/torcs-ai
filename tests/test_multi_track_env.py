"""Contract tests for seeded multi-track environment selection."""

from __future__ import annotations

from torcs_ai.envs import MultiTrackRacingEnv, RacingEnv

from test_racing_env import FakeTransport


def test_multi_track_selection_is_explicit_and_reported() -> None:
    wrapper = MultiTrackRacingEnv(
        {
            "alpha": RacingEnv(FakeTransport(), max_steps=2),
            "beta": RacingEnv(FakeTransport(), max_steps=2),
        }
    )
    observation, info = wrapper.reset(seed=7, options={"track": "beta"})
    assert wrapper.observation_space.contains(observation)
    assert wrapper.active_track == "beta"
    assert info["track"] == "beta"
    _, _, _, _, step_info = wrapper.step(4)
    assert step_info["track"] == "beta"
    wrapper.close()


def test_multi_track_selection_is_seeded() -> None:
    def selected(seed: int) -> str:
        wrapper = MultiTrackRacingEnv(
            {
                "alpha": RacingEnv(FakeTransport(), max_steps=1),
                "beta": RacingEnv(FakeTransport(), max_steps=1),
            }
        )
        try:
            _, info = wrapper.reset(seed=seed)
            return str(info["track"])
        finally:
            wrapper.close()

    assert selected(11) == selected(11)


def test_multi_track_rejects_unknown_explicit_track() -> None:
    wrapper = MultiTrackRacingEnv({"alpha": RacingEnv(FakeTransport(), max_steps=1)})
    try:
        try:
            wrapper.reset(options={"track": "missing"})
        except ValueError as exc:
            assert "track must be one of" in str(exc)
        else:  # pragma: no cover - assertion branch
            raise AssertionError("unknown track was accepted")
    finally:
        wrapper.close()
