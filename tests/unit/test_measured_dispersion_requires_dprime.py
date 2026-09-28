"""Measured dispersion must arrive complete; resolve_optics never derives D'."""

from __future__ import annotations

import numpy as np
import pytest
import tfs

from tmom_recon.optics import LoadedMeasurement, OpticsInput, resolve_optics

NAMES = [f"bpm{index}" for index in range(8)]


def _model() -> tfs.TfsDataFrame:
    phase = np.linspace(0.0, 1.0, len(NAMES), endpoint=False)
    frame = tfs.TfsDataFrame(
        {
            "s": np.arange(len(NAMES), dtype=float),
            "betx": 10.0 + phase,
            "bety": 8.0 - phase,
            "alfx": phase,
            "alfy": -phase,
            "mu1": 4.17 * phase,
            "mu2": 4.23 * phase,
            "dx": 1.0 + phase,
            "dy": 0.1 * phase,
            "dpx": 0.05 * phase,
            "dpy": 0.02 * phase,
        },
        index=NAMES,
    )
    frame.headers = {"q1": 4.17, "q2": 4.23}
    return frame


def _measurement(*, with_dprime: bool) -> LoadedMeasurement:
    model = _model()
    tws = model.copy(deep=True)
    tws["dx"] += 0.3
    tws["dy"] += 0.05
    for column in ("dx", "dy"):
        tws[f"{column}_err"] = 1e-3
    if with_dprime:
        for column in ("dpx", "dpy"):
            tws[f"{column}_err"] = 1e-4
    else:
        tws = tws.drop(columns=["dpx", "dpy"])
    return LoadedMeasurement(tws=tws, dispersion_found=True)


def _resolve(measured: LoadedMeasurement):
    return resolve_optics(
        optics_tws=_model(),
        zero_tws=_model(),
        optics=OpticsInput(measurement_dir="unused", sources={"dispersion": "measurement"}),
        measured=measured,
    )


def test_measured_dispersion_without_dprime_is_rejected() -> None:
    with pytest.raises(KeyError, match="dpx"):
        _resolve(_measurement(with_dprime=False))


def test_supplied_dprime_is_used_verbatim() -> None:
    resolved = _resolve(_measurement(with_dprime=True))
    expected = _measurement(with_dprime=True).tws
    for column in ("dx", "dy", "dpx", "dpy"):
        np.testing.assert_allclose(resolved.tws[column], expected[column])
