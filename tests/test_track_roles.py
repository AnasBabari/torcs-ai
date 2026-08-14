"""Unit tests for track partition roles and training contamination enforcement."""

from __future__ import annotations

import pytest

from torcs_ai.envs.multi_track import (
    ALL_APPROVED_TRACKS,
    HELD_OUT_TEST_TRACKS,
    TRAINING_TRACKS,
    VALIDATION_TRACKS,
    TrackRoleError,
    validate_track_training_selection,
)


def test_track_partition_roles_are_disjoint() -> None:
    train_set = set(TRAINING_TRACKS)
    val_set = set(VALIDATION_TRACKS)
    test_set = set(HELD_OUT_TEST_TRACKS)

    assert len(train_set) == 3
    assert len(val_set) == 1
    assert len(test_set) == 2

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set | val_set | test_set == set(ALL_APPROVED_TRACKS)


def test_validate_training_selection_permits_training_and_validation() -> None:
    # Training tracks permitted
    assert validate_track_training_selection(TRAINING_TRACKS) is True
    # Validation tracks permitted in training selection
    assert (
        validate_track_training_selection(["road/alpine-1", "road/ruudskogen"]) is True
    )


def test_validate_training_selection_rejects_held_out_by_default() -> None:
    # road/spring is held out
    with pytest.raises(TrackRoleError, match="held-out test track"):
        validate_track_training_selection(["road/alpine-1", "road/spring"])

    # road/street-1 is held out
    with pytest.raises(TrackRoleError, match="held-out test track"):
        validate_track_training_selection(["road/street-1"])


def test_validate_training_selection_with_override_marks_contamination() -> None:
    # When override is active, returns False to indicate contamination without raising error
    is_clean = validate_track_training_selection(
        ["road/alpine-1", "road/spring"], allow_held_out_training=True
    )
    assert is_clean is False
