"""The AC-dipole kick fit has no DC term: an AC dipole's kick has zero mean."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon.acd.cleaning import _solve_harmonic_series_fit
from tmom_recon.acd.models import (
    ACDipoleFitResult,
    ACDipoleHarmonicFit,
    ACDipoleStateEstimate,
    ACDipoleStateSeries,
)
from tmom_recon.acd.reconstruction import _assemble_result_dataframe


def _zero_estimate(turns: np.ndarray) -> ACDipoleStateEstimate:
    zeros = np.zeros_like(turns, dtype=float)
    state = ACDipoleStateSeries(x=zeros, px=zeros, y=zeros, py=zeros, t=zeros, pt=zeros)
    return ACDipoleStateEstimate(
        state=state,
        var_x=np.ones_like(zeros),
        var_px=np.ones_like(zeros),
        var_y=np.ones_like(zeros),
        var_py=np.ones_like(zeros),
    )


def test_static_difference_is_not_fitted_as_a_kick() -> None:
    # A static upstream/downstream mismatch (e.g. off-momentum model error) must
    # not become a static kick at the marker: the fit keeps only the harmonic.
    # 640 turns is exactly 104 periods, so a static offset is orthogonal to the harmonic.
    turns = np.arange(640, dtype=float)
    tune = 0.1625
    amplitude = 2.5e-6
    phase = 0.37
    values = amplitude * np.sin(2 * np.pi * tune * turns + phase) + 8.0e-6

    fit, _ = _solve_harmonic_series_fit(
        turns, tune=tune, values=values, variances=np.full_like(values, 1e-12)
    )

    assert fit.amplitude == pytest.approx(amplitude, rel=1e-3)
    assert abs(fit.fitted.mean()) < 1e-2 * amplitude
    np.testing.assert_allclose(
        fit.fitted, amplitude * np.sin(2 * np.pi * tune * turns + phase), atol=1e-8, rtol=0.0
    )


def test_summary_exports_the_fitted_harmonic() -> None:
    turns = np.arange(8, dtype=float)
    harmonic = 2.0e-6 * np.sin(2 * np.pi * 0.125 * turns)
    horizontal = ACDipoleHarmonicFit(tune=0.125, amplitude=2.0e-6, phase=0.0, fitted=harmonic)
    vertical = ACDipoleHarmonicFit(tune=0.2, amplitude=0.0, phase=0.0, fitted=np.zeros_like(turns))
    zero = _zero_estimate(turns)
    fit = ACDipoleFitResult(
        summary=pd.DataFrame({"turn": turns}),
        turns=turns,
        raw_upstream=zero,
        raw_downstream=zero,
        cleaned_upstream=zero,
        cleaned_downstream=zero,
        dpx_raw=harmonic,
        dpy_raw=np.zeros_like(turns),
        dpx_fit=horizontal,
        dpy_fit=vertical,
        dpx_r2=1.0,
        dpy_r2=1.0,
    )

    result = _assemble_result_dataframe(fit.summary.copy(), fit=fit, pt_est=0.0)

    np.testing.assert_allclose(result["dpx_fit_rad"], harmonic, atol=1e-18)


def _tracked_table(turns: np.ndarray, px: np.ndarray, var: float = 1e-12) -> pd.DataFrame:
    zeros = np.zeros_like(turns)
    return pd.DataFrame(
        {
            "turn": turns.astype(int),
            "x": zeros,
            "px": px,
            "y": zeros,
            "py": zeros,
            "var_x": np.full_like(zeros, var),
            "var_px": np.full_like(zeros, var),
            "var_y": np.full_like(zeros, var),
            "var_py": np.full_like(zeros, var),
            "pt": zeros,
        }
    )


def test_closed_orbit_step_is_kept_in_the_states_not_in_the_kick() -> None:
    # Each side oscillates about its own closed orbit; the constant step between
    # them is an orbit-angle difference.  The kick stays zero-mean, the pre- and
    # post-kick states keep their own side's orbit, even on a window that is not
    # an integer number of periods (where a constant would leak into sin/cos).
    from tmom_recon.acd.cleaning import _clean_ac_dipole_states

    turns = np.arange(700, dtype=float)
    tune = 0.1625
    kick = 2.5e-6 * np.sin(2 * np.pi * tune * turns + 0.37)
    orbit_up, orbit_down = 1.0e-5, -3.7e-5
    upstream = _tracked_table(turns, orbit_up + 0.0 * turns)
    downstream = _tracked_table(turns, orbit_down + kick)

    result = _clean_ac_dipole_states(
        turns,
        [upstream],
        [downstream],
        dpx_tune=tune,
        dpy_tune=tune,
        smooth_lambda=0.0,
    )

    np.testing.assert_allclose(result.dpx_fit.fitted, kick, atol=1e-8, rtol=0.0)
    assert result.dpx_orbit_step == pytest.approx(orbit_down - orbit_up, abs=1e-8)
    np.testing.assert_allclose(result.upstream.state.px, orbit_up, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(result.downstream.state.px, orbit_down + kick, atol=1e-8, rtol=0.0)
