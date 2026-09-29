"""AC-dipole marker states from the global, coupled forced response.

The tracked method (:mod:`tmom_recon.acd.reconstruction`) builds each marker
state from the two BPMs next to the AC dipole and tracks it to the marker. Its
kick is a difference of two tracked momenta, so a few-percent local model error
between those BPMs becomes a tens-of-percent kick error.

Driven motion. The horizontal (vertical) AC dipole kicks ``px`` (``py``) at
the driven tune ``Q_1`` (``Q_2``). In steady state the linear ring answers each
drive with one forced mode, so at element ``s`` on turn ``n``

    X_n(s) = sum_k Re[ C_k g_k(s) exp(2 pi i Q_k n) ]

with ONE complex amplitude ``C_k`` per drive (the kick amplitude and phase) and
the exact forced response ``g_k(s)`` (components ``x, px, y, py``) of the
undriven reconstruction model. With ``b``/``a`` the marker ends before/after
the AC dipole, ``M_b`` the one-turn matrix there and ``e_k`` the kicked
momentum:

    before:          xi_k = (exp(2 pi i Q_k) I - M_b)^-1 M_b e_k
    after:           xi_k + e_k
    s downstream:    M(a -> s) (xi_k + e_k)
    s upstream:      M(s -> b)^-1 xi_k

"Upstream" means before the AC dipole in the sequence, so ``turn`` labels must
increment at the sequence start (as the tracked marker transport needs).

All matrices come from MAD-NG's linear normalizing map A
(``twiss{coupling=true, savemap=1}``, see ``madl_gphys.mad``: every optics
column is built from A, which is propagated without rephasing). The normal-mode
vectors are ``v_k = A[:, 2k] + i A[:, 2k+1]`` (:func:`normal_mode_responses`;
``sqrt(beta11) exp(2 pi i mu1)`` in x, and the ``ev1``/``ev2`` columns give their
cross-plane phase), transfer matrices are ``A(s') A(s)^-1`` and
``M_b = V diag(exp(+-2 pi i Q_nat)) V^-1``.

Why not the driven twiss: MAD-NG's driven optics model each AC dipole as a
lens acting on its own plane at all tunes. Without coupling this equals the
forced response exactly. With coupling, mode 1 has y motion at the AC dipole,
and the vertical lens kicks it at ``Q_1``, which the real vertical AC dipole
(driving only at ``Q_2``) never does.

``C_k`` is fitted by weighted least squares from the ``Q_k`` lines of x AND y at
all BPMs, and the marker states follow from ``g_k`` at the marker ends. The
kick is ``Re[C_k exp(2 pi i Q_k n)]`` itself rather than a difference of tracked
momenta, and the data enter only through ``C_1, C_2``, so rebuilding the
markers from a fitted model does not feed local errors back.

Only the driven lines are replaced. The static part of each marker coordinate
(the data-driven off-momentum orbit) and the variances are kept from the
tracked states.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from tmom_recon.acd.madng_driver import ACDipoleMadDriver

LOGGER = logging.getLogger(__name__)

GLOBAL_SUMMARY_ATTR_NAME = "acd_global_markers"
COORDS = ("x", "px", "y", "py")
_OWN_PLANE = ("x", "y")  # plane in which mode k is the drive

__all__ = [
    "COORDS",
    "GLOBAL_SUMMARY_ATTR_NAME",
    "GlobalModeFit",
    "apply_global_marker_states",
    "fit_mode_amplitude",
    "forced_mode_responses",
    "harmonic_lines",
    "normal_mode_responses",
]


@dataclass(frozen=True)
class GlobalModeFit:
    """Global fit of one AC-dipole drive.

    Attributes:
        mode: 1 (horizontal AC dipole) or 2 (vertical).
        tune: Driven tune of the mode.
        amplitude: Complex drive amplitude ``C`` (the AC-dipole kick in rad:
            ``Re[C exp(2 pi i Q n)]``).
        bpms: BPMs used, in data order.
        residuals: Per-BPM relative residual of the mode's own-plane line,
            ``|Z_i - C g_i| / |Z_i|``.
    """

    mode: int
    tune: float
    amplitude: complex
    bpms: tuple[str, ...]
    residuals: np.ndarray

    @property
    def median_residual(self) -> float:
        return float(np.median(self.residuals))


def harmonic_lines(
    turns: np.ndarray, values: np.ndarray, tunes: Sequence[float]
) -> tuple[np.ndarray, float]:
    """Complex amplitudes ``Z_j`` of the ``tunes`` lines and the offset ``c``.

    Joint least-squares fit of ``values ~ sum_j Re[Z_j exp(2 pi i Q_j n)] + c``.
    """
    turns = np.asarray(turns, dtype=float)
    columns = []
    for tune in tunes:
        phase = 2 * np.pi * float(tune) * turns
        columns += [np.cos(phase), np.sin(phase)]
    design = np.column_stack([*columns, np.ones_like(turns)])
    coef, *_ = np.linalg.lstsq(design, np.asarray(values, dtype=float), rcond=None)
    return coef[0:-1:2] - 1j * coef[1:-1:2], float(coef[-1])


def fit_mode_amplitude(lines: np.ndarray, model: np.ndarray, weights: np.ndarray) -> complex:
    """Weighted least-squares ``C`` of ``lines ~ C model`` (any shape, same for all)."""
    return complex(np.sum(weights * np.conj(model) * lines) / np.sum(weights * np.abs(model) ** 2))


def normal_mode_responses(
    model: ACDipoleMadDriver, pt: float
) -> tuple[dict[str, np.ndarray], tuple[float, float]]:
    """Normal-mode vectors of ``model`` at its observed elements, and its tunes.

    Runs a coupled twiss with ``savemap=1`` and reads the linear normalizing
    map A at each observed element (sequence order kept). Returns
    ``{NAME: v}`` with ``v`` of shape ``(4, 2)`` (rows ``x, px, y, py``, columns
    mode 1 and 2, ``v[:, k] = A[:4, 2k] + i A[:4, 2k+1]``) and the fractional
    natural tunes ``(q1, q2)``.
    """
    tws = model.run_twiss(observe=1, coupling=True, pt=pt, savemap=1)
    tunes = (float(tws.headers["q1"]) % 1.0, float(tws.headers["q2"]) % 1.0)
    model.mad.send(
        """
py:send(#tws)
for i = 1, #tws do
    py:send(tws[i].name)
    py:send(tws[i].__map)
end
"""
    )
    vectors = {}
    for _ in range(int(model.mad.recv())):
        name = str(model.mad.recv()).upper()
        a = np.asarray(model.mad.recv(), dtype=float)[:4, :4]
        vectors.setdefault(name, a[:, 0::2] + 1j * a[:, 1::2])
    return vectors, tunes


def _real_basis(v: np.ndarray) -> np.ndarray:
    """Real 4x4 A from the complex mode vectors (columns Re, Im per mode)."""
    return np.column_stack([v[:, 0].real, v[:, 0].imag, v[:, 1].real, v[:, 1].imag])


def forced_mode_responses(
    vectors: Mapping[str, np.ndarray],
    natural_tunes: tuple[float, float],
    driven_tunes: tuple[float, float],
    *,
    before: str,
    after: str,
) -> dict[str, np.ndarray]:
    """Exact steady-state response ``g_k`` to a unit kick in ``p_k`` at ``Q_k``.

    Args:
        vectors: :func:`normal_mode_responses` of the undriven model, in
            sequence order, including the marker ends ``before``/``after``.
        natural_tunes: The model's fractional natural tunes.
        driven_tunes: The AC-dipole drive tunes ``(Q_1, Q_2)``.
        before, after: Marker names on either side of the AC dipole.

    Returns:
        ``{NAME: g}`` with ``g`` of shape ``(4, 2)`` (column k: drive k).
    """
    names = list(vectors)
    before, after = before.upper(), after.upper()
    order = {name: i for i, name in enumerate(names)}
    a_before, a_after = _real_basis(vectors[before]), _real_basis(vectors[after])
    v = vectors[before]
    basis = np.column_stack([v[:, 0], v[:, 0].conj(), v[:, 1], v[:, 1].conj()])
    lam = np.exp(
        2j
        * np.pi
        * np.array([natural_tunes[0], -natural_tunes[0], natural_tunes[1], -natural_tunes[1]])
    )
    one_turn = np.real(basis @ np.diag(lam) @ np.linalg.inv(basis))
    xi, xi_after = np.zeros((4, 2), complex), np.zeros((4, 2), complex)
    for k, q in enumerate(driven_tunes):
        kick = np.zeros(4)
        kick[2 * k + 1] = 1.0
        xi[:, k] = np.linalg.solve(np.exp(2j * np.pi * q) * np.eye(4) - one_turn, one_turn @ kick)
        xi_after[:, k] = xi[:, k] + kick
    inv_before, inv_after = np.linalg.inv(a_before), np.linalg.inv(a_after)
    responses = {}
    for name in names:
        a_s = _real_basis(vectors[name])
        if name == before:
            responses[name] = xi
        elif order[name] > order[after]:
            responses[name] = a_s @ inv_after @ xi_after
        elif name == after:
            responses[name] = xi_after
        else:
            responses[name] = a_s @ inv_before @ xi
    return responses


def _fit_modes(
    data: pd.DataFrame, responses: Mapping[str, np.ndarray], tunes: tuple[float, float]
) -> list[GlobalModeFit]:
    lines, model, weights, bpms = [], [], [], []
    for name, rows in data.groupby("name", sort=False):
        key = str(name).upper()
        if key not in responses:
            continue
        var = np.array([np.mean(rows["var_x"]), np.mean(rows["var_y"])], dtype=float)
        if not np.all(np.isfinite(var) & (var > 0)):
            continue
        turns = rows["turn"].to_numpy(float)
        # lines[plane, mode]
        lines.append([harmonic_lines(turns, rows[p].to_numpy(float), tunes)[0] for p in _OWN_PLANE])
        model.append(responses[key][[0, 2], :])
        weights.append(1.0 / var)
        bpms.append(str(name))
    if len(bpms) < 2:
        raise ValueError(
            f"Global ACD markers need at least 2 BPMs with finite var_x/var_y; got {len(bpms)}"
        )
    lines, model, weights = np.array(lines), np.array(model), np.array(weights)[:, :, None]
    fits = []
    for k in range(2):
        amplitude = fit_mode_amplitude(lines[:, :, k], model[:, :, k], weights[:, :, 0])
        own = lines[:, k, k]
        residuals = np.abs(own - amplitude * model[:, k, k]) / np.abs(own)
        fits.append(GlobalModeFit(k + 1, float(tunes[k]), amplitude, tuple(bpms), residuals))
    return fits


def apply_global_marker_states(
    result: pd.DataFrame,
    data: pd.DataFrame,
    responses: Mapping[str, np.ndarray],
    *,
    driven_tunes: tuple[float, float],
) -> pd.DataFrame:
    """Replace the driven lines of the tracked marker states with the global solution.

    Args:
        result: Tracked ACD result (``row_type`` ``marker_before``/``marker_after``
            rows are replaced; all other rows are copied unchanged).
        data: BPM turn-by-turn data with ``name, turn, x, y, var_x, var_y``.
            Any static frame offset per BPM is irrelevant (fitted as a constant).
        responses: Forced responses ``{NAME: g (4, 2)}`` at the BPMs and marker
            ends (:func:`forced_mode_responses`).
        driven_tunes: Drive tunes ``(Q_1, Q_2)`` of the horizontal and vertical AC dipole.

    Returns:
        Copy of ``result`` with global marker states and the two
        :class:`GlobalModeFit` in ``attrs[GLOBAL_SUMMARY_ATTR_NAME]``.
    """
    responses = {str(k).upper(): np.asarray(v) for k, v in responses.items()}
    tunes = (float(driven_tunes[0]), float(driven_tunes[1]))
    fits = _fit_modes(data, responses, tunes)
    for fit in fits:
        LOGGER.info(
            "Global ACD markers mode %d: kick amplitude %.3e rad, median BPM residual %.2e over %d BPMs",
            fit.mode,
            abs(fit.amplitude),
            fit.median_residual,
            len(fit.bpms),
        )
    amplitudes = np.array([fit.amplitude for fit in fits])
    out = result.copy(deep=True)
    for row_type in ("marker_before", "marker_after"):
        sel = out["row_type"] == row_type
        if not sel.any():
            raise ValueError(f"ACD result has no {row_type} rows")
        g = responses[str(out.loc[sel, "name"].iloc[0]).upper()]
        turns = out.loc[sel, "turn"].to_numpy(float)
        rotation = np.exp(2j * np.pi * np.outer(turns, tunes))  # (n, 2)
        for i, col in enumerate(COORDS):
            offset = harmonic_lines(turns, out.loc[sel, col].to_numpy(float), tunes)[1]
            out.loc[sel, col] = offset + np.real(rotation @ (amplitudes * g[i]))
    out.attrs[GLOBAL_SUMMARY_ATTR_NAME] = fits
    return out
