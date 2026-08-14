"""Unit tests for bootstrap confidence intervals and robust aggregation statistics."""

from __future__ import annotations

import numpy as np
import pytest

from torcs_ai.rl import compute_bootstrap_ci, compute_iqm


def test_bootstrap_ci_constant_values() -> None:
    data = [10.0, 10.0, 10.0, 10.0]
    low, high = compute_bootstrap_ci(data)
    assert low == pytest.approx(10.0)
    assert high == pytest.approx(10.0)


def test_bootstrap_ci_bounds() -> None:
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    low, high = compute_bootstrap_ci(data, num_resamples=1000, seed=42)
    mean_val = np.mean(data)
    assert low < mean_val < high
    assert low >= 10.0
    assert high <= 100.0


def test_compute_iqm() -> None:
    # 25% trimmed mean of [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # Length = 10, trimmed indices 2 to 8: [20, 30, 40, 50, 60, 70] -> mean = 45.0
    data = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    iqm = compute_iqm(data)
    assert iqm == pytest.approx(45.0)


def test_compute_iqm_small_sample() -> None:
    data = [10.0, 20.0]
    assert compute_iqm(data) == pytest.approx(15.0)
