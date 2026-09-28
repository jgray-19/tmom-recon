"""``measure_dispersion``: dispersion from repeated closed-orbit acquisitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon.physics.closed_orbit import measure_dispersion

pytestmark = pytest.mark.unit

BPMS = ["BPM1", "BPM2"]
DX = {"BPM1": 2.0, "BPM2": 0.0}
DY = {"BPM1": 0.3, "BPM2": -0.1}
X0 = {"BPM1": 1.0e-3, "BPM2": -5.0e-4}
Y0 = {"BPM1": 2.0e-4, "BPM2": 3.0e-4}
DDX = {"BPM1": 40.0, "BPM2": -15.0}
DDY = {"BPM1": 5.0, "BPM2": 0.0}
# A full model twiss: measure_dispersion picks ddx/ddy out itself.
TWISS = pd.DataFrame({"betx": 10.0, "bety": 12.0, "dx": DX, "dy": DY, "ddx": DDX, "ddy": DDY})


def _orbit(pt: float, noise: np.random.Generator | None, sigma: float = 0.0) -> pd.DataFrame:
    x = np.array([X0[b] + DX[b] * pt + DDX[b] * pt**2 for b in BPMS])
    y = np.array([Y0[b] + DY[b] * pt + DDY[b] * pt**2 for b in BPMS])
    if noise is not None and sigma:
        x = x + noise.normal(0.0, sigma, size=len(BPMS))
        y = y + noise.normal(0.0, sigma, size=len(BPMS))
    return pd.DataFrame({"name": BPMS, "x": x, "y": y})


def _noiseless_orbits_by_pt(pts) -> dict[float, list[pd.DataFrame]]:
    return {pt: [_orbit(pt, None), _orbit(pt, None)] for pt in pts}


def test_recovers_exact_slope_from_noiseless_repeats():
    pts = [-2.0e-3, -1.0e-3, 0.0, 1.0e-3, 2.0e-3]
    orbits_by_pt = _noiseless_orbits_by_pt(pts)
    pt_sigma = dict.fromkeys(pts, 0.0)

    result = measure_dispersion(orbits_by_pt, pt_sigma, tws=TWISS)

    for bpm in BPMS:
        assert result.loc[bpm, "dx"] == pytest.approx(DX[bpm], abs=1e-9)
        assert result.loc[bpm, "dy"] == pytest.approx(DY[bpm], abs=1e-9)


def test_uncertainty_shrinks_with_more_repeats():
    rng = np.random.default_rng(0)
    pts = [-2.0e-3, -1.0e-3, 0.0, 1.0e-3, 2.0e-3]
    sigma = 5.0e-5
    pt_sigma = dict.fromkeys(pts, 0.0)

    few = {pt: [_orbit(pt, rng, sigma) for _ in range(2)] for pt in pts}
    many = {pt: [_orbit(pt, rng, sigma) for _ in range(20)] for pt in pts}

    err_few = measure_dispersion(few, pt_sigma, tws=TWISS).loc["BPM1", "dx_err"]
    err_many = measure_dispersion(many, pt_sigma, tws=TWISS).loc["BPM1", "dx_err"]

    assert err_many < err_few


def test_pt_sigma_inflates_error_in_proportion_to_slope():
    rng = np.random.default_rng(1)
    pts = [-2.0e-3, -1.0e-3, 0.0, 1.0e-3, 2.0e-3]
    orbits_by_pt = {pt: [_orbit(pt, rng, 1.0e-6) for _ in range(5)] for pt in pts}

    tight = measure_dispersion(orbits_by_pt, dict.fromkeys(pts, 0.0), tws=TWISS)
    loose = measure_dispersion(orbits_by_pt, dict.fromkeys(pts, 1.0e-4), tws=TWISS)

    # BPM1 has a large dispersion slope: pt uncertainty should inflate its
    # reported error. BPM2's slope is ~0, so pt uncertainty barely matters.
    assert loose.loc["BPM1", "dx_err"] > tight.loc["BPM1", "dx_err"]
    assert loose.loc["BPM2", "dx_err"] == pytest.approx(tight.loc["BPM2", "dx_err"], rel=0.05)


def test_no_repeat_scatter_reports_no_uncertainty():
    """Without scatter the slope is still fitted, but its error has no scale."""
    pts = [-1.0e-3, 0.0, 1.0e-3]
    orbits_by_pt = {pt: [_orbit(pt, None)] for pt in pts}

    result = measure_dispersion(orbits_by_pt, dict.fromkeys(pts, 1.0e-5), tws=TWISS)

    assert result.loc["BPM1", "dx"] == pytest.approx(DX["BPM1"], abs=1e-9)
    assert result["dx_err"].isna().all()


def test_sem_counts_only_acquisitions_that_saw_the_bpm():
    """A repeat that lost a BPM must not shrink that BPM's standard error."""
    rng = np.random.default_rng(2)
    pts = [-1.0e-3, 0.0, 1.0e-3]
    sigma = 5.0e-5
    full = {pt: [_orbit(pt, rng, sigma) for _ in range(3)] for pt in pts}
    lossy = {pt: [*acqs, _orbit(pt, None).assign(x=np.nan, y=np.nan)] for pt, acqs in full.items()}
    pt_sigma = dict.fromkeys(pts, 0.0)

    expected = measure_dispersion(full, pt_sigma, tws=TWISS)
    result = measure_dispersion(lossy, pt_sigma, tws=TWISS)

    pd.testing.assert_frame_equal(result, expected)


def test_fewer_than_two_pt_settings_gives_nan():
    orbits_by_pt = _noiseless_orbits_by_pt([0.0])
    result = measure_dispersion(orbits_by_pt, {0.0: 0.0}, tws=TWISS)

    assert result["dx"].isna().all()
    assert result["dx_err"].isna().all()
    assert result["dy"].isna().all()
    assert result["dy_err"].isna().all()


def test_missing_pt_sigma_key_raises():
    orbits_by_pt = _noiseless_orbits_by_pt([0.0, 1.0e-3])

    with pytest.raises(ValueError, match="pt_sigma"):
        measure_dispersion(orbits_by_pt, {0.0: 0.0}, tws=TWISS)


def test_asymmetric_scan_gives_the_derivative_at_zero_not_the_secant():
    """With D2 removed the slope is D(0) even when the settings are not symmetric."""
    pts = [0.0, 1.0e-3, 2.0e-3]
    orbits_by_pt = _noiseless_orbits_by_pt(pts)
    pt_sigma = dict.fromkeys(pts, 0.0)

    result = measure_dispersion(orbits_by_pt, pt_sigma, tws=TWISS)
    secant = measure_dispersion(orbits_by_pt, pt_sigma, tws=TWISS.assign(ddx=0.0, ddy=0.0))

    assert result.loc["BPM1", "dx"] == pytest.approx(DX["BPM1"], abs=1e-9)
    # Ignoring D2 biases the slope by D2 * (mean pt weighting), here 2e-3 * 40 = 0.08.
    assert abs(secant.loc["BPM1", "dx"] - DX["BPM1"]) > 0.05


def test_twiss_missing_second_order_columns_raises():
    orbits_by_pt = _noiseless_orbits_by_pt([0.0, 1.0e-3])

    with pytest.raises(KeyError, match="ddx"):
        measure_dispersion(orbits_by_pt, {0.0: 0.0, 1.0e-3: 0.0}, tws=TWISS.drop(columns="ddx"))


def test_twiss_missing_a_bpm_raises():
    orbits_by_pt = _noiseless_orbits_by_pt([0.0, 1.0e-3])

    with pytest.raises(KeyError, match="BPM2"):
        measure_dispersion(orbits_by_pt, {0.0: 0.0, 1.0e-3: 0.0}, tws=TWISS.loc[["BPM1"]])
