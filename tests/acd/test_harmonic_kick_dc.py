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
