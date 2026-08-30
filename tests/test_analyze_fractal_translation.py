from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_fractal_translation import denormalize, linear_tracking


def test_denormalize_maps_endpoints_and_midpoint() -> None:
    q01 = np.array([-2.0, -1.0, 2.0])
    q99 = np.array([2.0, 3.0, 6.0])
    normalized = np.array([[-1.0, 0.0, 1.0]])

    decoded = denormalize(normalized, q01, q99)

    np.testing.assert_allclose(decoded, [[-2.0, 1.0, 6.0]])


def test_linear_tracking_recovers_controller_scale() -> None:
    requested = np.array(
        [[1.0, 2.0, -1.0], [-2.0, 1.0, 2.0], [3.0, -1.0, 1.0]]
    )
    achieved = requested * np.array([0.2, 0.4, 0.8])

    result = linear_tracking(requested, achieved)

    assert result["x"]["achieved_per_requested_slope_through_origin"] == pytest.approx(0.2)
    assert result["y"]["achieved_per_requested_slope_through_origin"] == pytest.approx(0.4)
    assert result["z"]["achieved_per_requested_slope_through_origin"] == pytest.approx(0.8)
