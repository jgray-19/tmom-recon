"""BPM position-noise uncertainty of the raw AC-dipole kick.

The raw kick is ``p_after - p_before`` at the marker, each side the
selected BPM's state transported to the marker. A BPM state is
``(x, px, y, py)`` with ``px, py`` linear in the BPM and neighbour positions
(:mod:`tmom_recon.physics.momenta`), so the kick is a function of the measured
positions only. This module propagates their independent noise through that chain:
momentum reconstruction and the per-turn MAD-NG transport Jacobian, keeping the
correlations between the two sides and between shared neighbour BPMs.

Optics errors are systematic, not turn-to-turn noise, and are not included.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tmom_recon.data.schema import NEXT, PREV, SUFFIX_NEXT, SUFFIX_PREV
from tmom_recon.lattice.core import neighbour_plane_factors

if TYPE_CHECKING:
    from .madng_driver import ACDipoleMadDriver
    from .models import ACDipoleSide

PLANES = ("x", "y")
# Central-difference step of the transport Jacobian, in x/y [m] and px/py [rad].
JACOBIAN_STEP = 1e-6
# Output index of the momentum of each plane in (x, px, y, py).
MOMENTUM_INDEX = {"x": 1, "y": 3}
POSITION_INDEX = {"x": 0, "y": 2}

# (BPM name, plane, measurement turn minus kick turn)
MeasurementKey = tuple[str, str, int]
KickJacobian = dict[MeasurementKey, np.ndarray]


def transport_jacobians(
    model: ACDipoleMadDriver, side: ACDipoleSide, name: str, states: np.ndarray
) -> np.ndarray:
    """Per-state ``d(marker state) / d(BPM state)``, shape ``(n, 4, 4)`` as ``[out, in]``.

    Central finite differences through MAD-NG at each state.
    """
    probes = np.repeat(states[:, None, :], 8, axis=1)
    for k in range(4):
        probes[:, 2 * k, k] += JACOBIAN_STEP
        probes[:, 2 * k + 1, k] -= JACOBIAN_STEP
    marker = model.acd_before if side.direction == +1 else model.acd_after
    tracked = model.track_particles(name, marker, probes.reshape(-1, 4), direction=side.direction)
    tracked = tracked[:, :4].reshape(len(states), 8, 4)
    derivative = (tracked[:, 0::2, :] - tracked[:, 1::2, :]) / (2 * JACOBIAN_STEP)
    return derivative.transpose(0, 2, 1)


def momentum_coefficients(
    frame: pd.DataFrame, plane: str, *, is_prev: bool
) -> tuple[np.ndarray, np.ndarray]:
    """``d p_plane / d position`` of the BPM itself and of its neighbour, per row.

    From the momentum reconstruction of :mod:`tmom_recon.physics.momenta`; the
    dispersion terms are constant offsets and drop out.
    """
    names, suffix = (PREV, SUFFIX_PREV) if is_prev else (NEXT, SUFFIX_NEXT)
    delta = frame[names.delta_x if plane == "x" else names.delta_y].to_numpy(float)
    sign, alpha_sign, _, tan_phi, sec_phi = neighbour_plane_factors(
        2 * np.pi * delta, is_prev=is_prev
    )
    sqrt_beta = frame[f"sqrt_beta{plane}"].to_numpy(float)
    sqrt_beta_neighbour = frame[f"sqrt_beta{plane}_{suffix}"].to_numpy(float)
    alpha = frame[f"alfa{plane}"].to_numpy(float)
    own = sign * (tan_phi + alpha_sign * alpha) / sqrt_beta**2
    neighbour = sign * sec_phi / (sqrt_beta * sqrt_beta_neighbour)
    return own, neighbour


def _on_turns(values: np.ndarray, frame_turns: np.ndarray, turns: np.ndarray) -> np.ndarray:
    """Re-index per-row *values* (last axis = rows) onto *turns*."""
    return values[..., pd.Index(frame_turns).get_indexer(turns)]


def _add(jacobian: KickJacobian, key: MeasurementKey, coefficient: np.ndarray) -> None:
    jacobian[key] = jacobian.get(key, 0.0) + coefficient


def marker_momentum_jacobian(
    side: ACDipoleSide, model: ACDipoleMadDriver, turns: np.ndarray, plane: str
) -> tuple[KickJacobian, dict[tuple[str, str], float]]:
    """``d p_plane(marker, this side) / d measured position`` per turn, and position variances.

    Returns:
        ``(jacobian, variances)``; ``variances`` maps ``(bpm, plane)`` to the BPM
        position variance [m^2].
    """
    is_prev = side.direction == +1
    names, suffix = (PREV, SUFFIX_PREV) if is_prev else (NEXT, SUFFIX_NEXT)
    out = MOMENTUM_INDEX[plane]
    bpm, frame = side.bpm, side.frame
    states = frame[["x", "px", "y", "py"]].to_numpy(float)
    frame_turns = frame["turn"].to_numpy(int)
    transport = _on_turns(
        transport_jacobians(model, side, bpm, states).transpose(1, 2, 0), frame_turns, turns
    )

    variances: dict[tuple[str, str], float] = {}
    for p in PLANES:
        variances[(bpm, p)] = float(frame[f"var_{p}"].mean())
        neighbour = getattr(names, f"bpm_{p}")
        variances[(str(frame[neighbour].iat[0]), p)] = float(
            frame[getattr(names, f"var_{p}")].mean()
        )

    jacobian: KickJacobian = {}
    for p in PLANES:
        own, neighbour = momentum_coefficients(frame, p, is_prev=is_prev)
        own, neighbour = (_on_turns(c, frame_turns, turns) for c in (own, neighbour))
        offset_column = f"turn_{p}_{suffix}"
        offset = int(
            np.median(frame[offset_column].to_numpy(float) - frame["turn"].to_numpy(float))
        )
        pos, mom = POSITION_INDEX[p], MOMENTUM_INDEX[p]
        to_own = transport[out, pos] + transport[out, mom] * own
        to_neighbour = transport[out, mom] * neighbour
        neighbour_bpm = str(frame[getattr(names, f"bpm_{p}")].iat[0])
        _add(jacobian, (bpm, p, 0), to_own)
        _add(jacobian, (neighbour_bpm, p, offset), to_neighbour)
    return jacobian, variances


def kick_jacobian(
    upstream: ACDipoleSide,
    downstream: ACDipoleSide,
    model: ACDipoleMadDriver,
    turns: np.ndarray,
    plane: str,
) -> tuple[KickJacobian, dict[tuple[str, str], float]]:
    """``d(p_after - p_before) / d measured position`` per kick turn, and position variances.

    Key ``(bpm, plane, k)`` is the measurement of *bpm* at turn ``t + k`` for the kick
    at turn ``t``; the array is indexed by ``t`` (``turns``).
    """
    before, before_var = marker_momentum_jacobian(upstream, model, turns, plane)
    after, after_var = marker_momentum_jacobian(downstream, model, turns, plane)
    jacobian: KickJacobian = {}
    for key, coefficient in after.items():
        _add(jacobian, key, coefficient)
    for key, coefficient in before.items():
        _add(jacobian, key, -coefficient)
    return jacobian, {**before_var, **after_var}


def kick_std(jacobian: KickJacobian, variances: dict[tuple[str, str], float]) -> np.ndarray:
    """Per-turn standard deviation of the raw kick from BPM position noise [rad]."""
    variance = sum(
        coefficient**2 * variances[(bpm, p)] for (bpm, p, _), coefficient in jacobian.items()
    )
    return np.sqrt(variance)
