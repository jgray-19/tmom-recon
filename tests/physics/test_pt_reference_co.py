"""``estimate_pt_from_model`` requires a measured nominal-RF closed orbit.

Why it is mandatory rather than optional: the bend response matrix spans the
entire horizontal BPM space (rank 16 of 16 on PSB ring 3), so an unknown
dipole-error closed orbit is *exactly* degenerate with the dispersive orbit at a
single momentum. Referencing to a measured orbit cancels the error orbit
identically; referencing to a model orbit that does not carry the machine's
errors leaks the whole error orbit into pt.

The second-order test pins the other half: ``pt`` and ``pt**2`` are one unknown,
so a single orbit determines the quadratic solution -- no momentum scan needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon.physics.pt_calculation import _solve_pt_quadratic, estimate_pt_from_model

BPMS = [f"bpm{i}" for i in range(12)]
PT = 4.17e-3


def _twiss(with_ddx: bool = True) -> pd.DataFrame:
    phase = np.linspace(0.0, 2.0 * np.pi, len(BPMS), endpoint=False)
    tws = pd.DataFrame(
        {"dx": 2.0 + 0.9 * np.cos(phase), "dy": np.zeros(len(BPMS))},
        index=pd.Index(BPMS, name="name"),
    )
    if with_ddx:
        tws["ddx"] = 1.3 + 0.5 * np.sin(phase)
    return tws


def _turn_data(orbit: np.ndarray, turns: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": np.tile(BPMS, turns),
            "turn": np.repeat(np.arange(turns), len(BPMS)),
            "x": np.tile(orbit, turns),
            "y": np.zeros(len(BPMS) * turns),
        }
    )


def _orbit(tws: pd.DataFrame, error_co: np.ndarray, pt: float = PT) -> np.ndarray:
    """Closed orbit at *pt*: error orbit + first- and second-order dispersion."""
    return error_co + pt * tws["dx"].to_numpy() + pt**2 * tws["ddx"].to_numpy()


def test_missing_reference_co_is_rejected() -> None:
    tws = _twiss()
    data = _turn_data(_orbit(tws, np.zeros(len(BPMS))))
    with pytest.raises(ValueError, match="requires `reference_co`"):
        estimate_pt_from_model(data, tws, reference_co=None, info=False)


def test_reference_co_must_cover_the_selected_bpms() -> None:
    tws = _twiss()
    data = _turn_data(_orbit(tws, np.zeros(len(BPMS))))
    partial = pd.DataFrame({"x": 0.0}, index=pd.Index(BPMS[:5]))
    with pytest.raises(ValueError, match="missing BPMs"):
        estimate_pt_from_model(data, tws, reference_co=partial, info=False)


def test_measured_reference_cancels_an_arbitrary_error_closed_orbit() -> None:
    """A millimetre-scale error orbit must not leak into pt once referenced."""
    tws = _twiss()
    rng = np.random.default_rng(3)
    # A realistic error orbit is not random noise: the bend response spans the
    # whole horizontal BPM space, so it generically carries a component *along*
    # dx -- which is precisely the component indistinguishable from momentum.
    # Here that component alone mimics a pt of 2e-3, half the true value.
    mimicked_pt = 2.0e-3
    error_co = mimicked_pt * tws["dx"].to_numpy() + rng.normal(0.0, 5e-4, len(BPMS))

    reference = pd.DataFrame({"x": error_co}, index=pd.Index(BPMS))
    data = _turn_data(_orbit(tws, error_co))

    referenced = estimate_pt_from_model(data, tws, reference_co=reference, info=False)
    assert referenced == pytest.approx(PT, rel=1e-9)

    # Referencing to a *model* orbit that does not know the errors (here, zero)
    # leaks the whole error orbit into pt. This is the failure the mandatory
    # argument exists to prevent, so pin that it is large.
    zero_reference = pd.DataFrame({"x": np.zeros(len(BPMS))}, index=pd.Index(BPMS))
    unreferenced = estimate_pt_from_model(data, tws, reference_co=zero_reference, info=False)
    assert unreferenced == pytest.approx(PT + mimicked_pt, rel=0.05)


def test_second_order_beats_first_order_on_a_single_orbit() -> None:
    """One orbit suffices: pt and pt**2 are a single unknown, not two."""
    tws = _twiss()
    reference = pd.DataFrame({"x": np.zeros(len(BPMS))}, index=pd.Index(BPMS))
    data = _turn_data(_orbit(tws, np.zeros(len(BPMS))))

    second = estimate_pt_from_model(data, tws, reference_co=reference, info=False)
    first = estimate_pt_from_model(
        data, tws.drop(columns=["ddx"]), reference_co=reference, info=False
    )

    assert second == pytest.approx(PT, rel=1e-9)
    # The first-order estimate is biased high by the neglected pt**2*ddx term.
    assert first > PT
    assert abs(second - PT) < abs(first - PT) / 100.0


def test_quadratic_solver_picks_the_root_next_to_the_linear_solution() -> None:
    """The far root is spurious; it must never be returned."""
    s_dx2, s_ddx_dx = 50.0, 30.0
    pt = 4.0e-3
    numerator = pt * s_dx2 + pt**2 * s_ddx_dx
    assert _solve_pt_quadratic(numerator, s_dx2, s_ddx_dx) == pytest.approx(pt, rel=1e-12)
    # Degenerate second-order term falls back to the linear solution.
    assert _solve_pt_quadratic(numerator, s_dx2, 0.0) == pytest.approx(numerator / s_dx2)
