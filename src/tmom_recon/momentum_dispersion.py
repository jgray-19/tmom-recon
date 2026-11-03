"""Derive the momentum dispersion ``D'`` from measured position dispersion.

A BPM measures position, so a momentum scan yields ``Dx``/``Dy`` and nothing
else.  ``D'`` is not lost, though: dispersion propagates exactly like a
trajectory, so the pair :math:`(D, D')` at one BPM is mapped to the next by the
same transfer matrix that maps :math:`(x, p_x)`.  One measured ``D`` at each of
two BPMs therefore determines ``D'`` at the first, which is what
:func:`derive_momentum_dispersion` does.

This is the method omc3 uses in ``optics_measurements.dispersion._calculate_dp``,
reimplemented here in this package's lowercase MAD-NG column convention and
extended with error propagation, which the omc3 version does not provide.

It is not a transcription. omc3 builds :math:`m_{11}` and :math:`m_{12}` from
pandas Series indexed by BPM name, e.g.
``np.sqrt(df.loc[shifted, "BETX"] / df.loc[:, "BETX"])``. ``shifted`` is a
permutation of the same labels, so pandas aligns the two operands and that ratio
evaluates to exactly 1 -- the :math:`\beta_{i+1}/\beta_i` factor is silently
dropped, and with it every dependence on the *next* BPM's optics. The error
cancels out of the final ratio when the measurement equals the model, which is
why the two agree on a round trip, but not for a real measurement. This
implementation works on numpy arrays, where position and not label decides what
pairs with what.

The transfer matrix between the two BPMs is the full coupled :math:`4\times 4`
one, rebuilt from the model's Mais-Ripken columns (see
:func:`_normalising_matrices`), so ``Dx'`` and ``Dy'`` are solved together from
the two measured ``Dx``/``Dy`` pairs. Per-plane :math:`2\times 2` maps drop the
x-y transfer terms and lose about a fifth of the coupled ``Dy'`` response.

The dipole term is what makes it exact rather than approximate.  Between two
BPMs the dispersion obeys an *inhomogeneous* equation -- bends generate
dispersion along the way -- so the homogeneous map alone does not connect
:math:`D_i` to :math:`D_{i+1}`.  The generated part is taken from the model as
the residual its own dispersion needs beyond homogeneous propagation,

.. math::

   g = \vec D^{\\mathrm{mdl}}_{i+1} - M \vec D^{\\mathrm{mdl}}_i,

with :math:`\vec D = (D_x, D'_x, D_y, D'_y)`, and subtracted from the
measurement.  Only this bend contribution is model sourced; the dispersion
values entering the answer are measured.  Because :math:`g` absorbs everything
between the two BPMs, the pair need not be physically adjacent -- a measurement
covering a subset of the ring's BPMs is handled correctly, each row simply
pairing with the next one present.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Smallest normalised angle sensitivity accepted between a BPM and its
#: successor: the smaller singular value of the position-from-angle block of the
#: transfer matrix, scaled by the betas (``|sin dphi|`` without coupling). ``D'``
#: is recovered by inverting that block, so a pair close to a half-integer phase
#: advance carries almost no information about the angle. Such a pair raises:
#: the fix is a different BPM set, not a quieter threshold.
MIN_SIN_PHASE_ADVANCE = 0.05

#: Model columns of the coupled linear optics (MAD-NG ``twiss{coupling=true}``).
_OPTICS_COLUMNS = (
    "beta11",
    "alfa11",
    "beta21",
    "alfa21",
    "beta12",
    "alfa12",
    "beta22",
    "alfa22",
    "mu1",
    "mu2",
    "kpa",
    "ev1",
    "ev2",
)
_DISPERSION_COLUMNS = ("dx", "dpx", "dy", "dpy")


def _mode_vector(beta_x, alfa_x, u_x, phase_x, beta_y, alfa_y, u_y, phase_y) -> np.ndarray:
    """Complex eigenvector ``(x, px, y, py)`` of one mode, shape ``(n, 4)``."""

    def plane(beta, alfa, u, phase):
        root = np.sqrt(np.clip(beta, 0.0, None))
        rotation = np.exp(1j * phase)
        momentum = np.divide(
            -alfa + 1j * u, root, out=np.zeros_like(root, dtype=complex), where=root > 0
        )
        return root * rotation, momentum * rotation

    x, px = plane(beta_x, alfa_x, u_x, phase_x)
    y, py = plane(beta_y, alfa_y, u_y, phase_y)
    return np.stack([x, px, y, py], axis=-1)


def _normalising_matrices(model: pd.DataFrame, phase_shift: tuple[float, float]) -> np.ndarray:
    """MAD-NG's normalising map A at each row, shape ``(n, 4, 4)``.

    Mode k's eigenvector is ``A[:, 2k] + i A[:, 2k+1]``: in its own plane
    ``sqrt(beta_kk) exp(2 pi i mu_k)`` with momentum ``(-alfa + i kpa)/sqrt(beta)``,
    and in the other plane the same with ``1 - kpa`` and a phase lag of
    ``arg(ev_k)``. ``phase_shift`` (in turns) is added to ``mu1``/``mu2``.
    """
    col = {c: model[c].to_numpy() for c in _OPTICS_COLUMNS}
    kpa = col["kpa"].astype(float)
    phase1 = 2.0 * np.pi * (col["mu1"].astype(float) + phase_shift[0])
    phase2 = 2.0 * np.pi * (col["mu2"].astype(float) + phase_shift[1])
    lag1 = np.angle(col["ev1"].astype(complex))
    lag2 = np.angle(col["ev2"].astype(complex))
    real = {k: col[k].astype(float) for k in _OPTICS_COLUMNS[:8]}
    mode1 = _mode_vector(
        real["beta11"],
        real["alfa11"],
        kpa,
        phase1,
        real["beta21"],
        real["alfa21"],
        1.0 - kpa,
        phase1 - lag1,
    )
    mode2 = _mode_vector(
        real["beta12"],
        real["alfa12"],
        1.0 - kpa,
        phase2 - lag2,
        real["beta22"],
        real["alfa22"],
        kpa,
        phase2,
    )
    return np.stack([mode1.real, mode1.imag, mode2.real, mode2.imag], axis=-1)


def _pair_transfer_matrices(model: pd.DataFrame, tunes: tuple[float, float]) -> np.ndarray:
    """Coupled transfer matrix from each row to the next, the last wrapping to the first."""
    here = _normalising_matrices(model, (0.0, 0.0))
    nxt = np.roll(here, -1, axis=0)
    # The first BPM one turn later: A propagates without rephasing, so mu grows by Q.
    nxt[-1] = _normalising_matrices(model.iloc[:1], tunes)[0]
    return nxt @ np.linalg.inv(here)


def _check_angle_sensitivity(
    angle_block: np.ndarray, model: pd.DataFrame, names: np.ndarray
) -> None:
    """Raise if any pair's position change is nearly blind to the angle."""
    betx = model["beta11"].to_numpy(dtype=float)
    bety = model["beta22"].to_numpy(dtype=float)
    nxt = np.roll(np.arange(len(betx)), -1)
    scale_out = np.stack([np.sqrt(betx[nxt]), np.sqrt(bety[nxt])], axis=-1)
    scale_in = np.stack([np.sqrt(betx), np.sqrt(bety)], axis=-1)
    normalised = angle_block / scale_out[:, :, None] / scale_in[:, None, :]
    sensitivity = np.linalg.svd(normalised, compute_uv=False)[:, -1]
    degenerate = sensitivity < MIN_SIN_PHASE_ADVANCE
    if degenerate.any():
        pairs = ", ".join(
            f"{names[i]}->{names[nxt[i]]} (sensitivity={sensitivity[i]:.4f})"
            for i in np.flatnonzero(degenerate)
        )
        raise ValueError(
            "Cannot derive momentum dispersion: the phase advance to the next BPM is "
            f"too close to a multiple of pi for {int(degenerate.sum())} pair(s): {pairs}. "
            "D' inverts as 1/sin(dphi), so these carry no usable angle information."
        )


def derive_momentum_dispersion(
    measured: pd.DataFrame,
    model: pd.DataFrame,
    *,
    tunes: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Derive ``dpx``/``dpy`` and their errors from measured ``dx``/``dy``.

    Args:
        measured: Ring-ordered frame carrying measured ``dx``/``dy``, and
            optionally ``dx_err``/``dy_err`` (assumed zero when absent).
        model: Coupled MAD-NG twiss on the same index, carrying the
            Mais-Ripken columns (``beta11``... ``alfa22``, ``mu1``/``mu2``,
            ``kpa``, ``ev1``/``ev2``) and ``dx``/``dpx``/``dy``/``dpy``.
            Phases are in turns.
        tunes: ``(q1, q2)``, used to close the ring for the final BPM pair.

    Returns:
        ``{"dpx": ..., "dpx_err": ..., "dpy": ..., "dpy_err": ...}``, each an
        array aligned with *measured*'s row order.

    Raises:
        KeyError: If a required column is missing from either frame.
        ValueError: If the frames are misaligned, hold fewer than two BPMs, or
            contain a BPM pair too close to a half-integer phase advance.
    """
    if not measured.index.equals(model.index):
        raise ValueError(
            "Measured and model frames must share one ring-ordered index to derive "
            "momentum dispersion"
        )
    if len(measured) < 2:
        raise ValueError(
            f"Deriving momentum dispersion needs at least two BPMs, got {len(measured)}"
        )
    # Each row pairs with the row after it and the last wraps to the first, so a
    # frame that is not in ring order silently pairs the wrong BPMs and produces
    # a plausible-looking wrong answer rather than an error.
    if "s" in model.columns:
        positions = model["s"].to_numpy(dtype=float)
        if not np.all(np.diff(positions) > 0):
            raise ValueError(
                "Momentum dispersion needs a ring-ordered frame, but the model 's' "
                "column is not strictly increasing"
            )
    missing_model = [c for c in (*_OPTICS_COLUMNS, *_DISPERSION_COLUMNS) if c not in model.columns]
    if missing_model:
        raise KeyError(
            f"Model twiss is missing {missing_model}, required to derive momentum dispersion"
        )
    missing_measured = [c for c in ("dx", "dy") if c not in measured.columns]
    if missing_measured:
        raise KeyError(f"Measured dispersion columns {missing_measured} are required to derive D'")

    names = measured.index.to_numpy()
    nxt = np.roll(np.arange(len(measured)), -1)
    transfer = _pair_transfer_matrices(model, tunes)
    positions, angles = [0, 2], [1, 3]
    position_block = transfer[:, positions][:, :, positions]
    angle_block = transfer[:, positions][:, :, angles]
    _check_angle_sensitivity(angle_block, model, names)

    # The dispersion generated by bends between the two BPMs, taken from the
    # model. It is the only model-sourced term in the result.
    model_d = model[list(_DISPERSION_COLUMNS)].to_numpy(dtype=float)
    generated = model_d[nxt] - np.einsum("nij,nj->ni", transfer, model_d)

    measured_d = measured[["dx", "dy"]].to_numpy(dtype=float)
    measured_err = np.stack(
        [
            measured[c].to_numpy(dtype=float) if c in measured.columns else np.zeros(len(measured))
            for c in ("dx_err", "dy_err")
        ],
        axis=-1,
    )
    rhs = (
        measured_d[nxt]
        - np.einsum("nij,nj->ni", position_block, measured_d)
        - generated[:, positions]
    )
    inverse = np.linalg.inv(angle_block)
    momentum_dispersion = np.einsum("nij,nj->ni", inverse, rhs)
    # The measured dispersions come from independent BPMs, so their errors add
    # in quadrature through the linear solve. The model matrices are treated as
    # exact, consistent with how the model optics are treated everywhere else.
    covariance = np.einsum("nij,nj,nkj->nik", inverse, measured_err[nxt] ** 2, inverse) + np.einsum(
        "nij,njk,nk,nlk,nml->nim",
        inverse,
        position_block,
        measured_err**2,
        position_block,
        inverse,
    )
    errors = np.sqrt(np.einsum("nii->ni", covariance))
    return {
        "dpx": momentum_dispersion[:, 0],
        "dpx_err": errors[:, 0],
        "dpy": momentum_dispersion[:, 1],
        "dpy_err": errors[:, 1],
    }
