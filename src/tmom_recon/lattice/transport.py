from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlaneTransportMatrix:
    """Uncoupled 2x2 transport matrix for one transverse plane.

    Coordinates are ordered as ``(z, pz)``, with ``z`` equal to ``x`` or ``y``.
    """

    r11: float
    r12: float
    r21: float
    r22: float


def _require_twiss_columns(twiss: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(twiss.columns)
    if missing:
        raise KeyError(f"Twiss table is missing required columns: {sorted(missing)}")


def _phase_advance(mu_to: float, mu_from: float, tune: float | None) -> float:
    """Forward phase advance in radians from ``mu_from`` to ``mu_to``.

    If the target precedes the source, ``tune`` supplies the ring wrap. It is
    required for rings and omitted for beam lines.
    """
    delta = float(mu_to) - float(mu_from)
    if delta < 0.0:
        if tune is None:
            raise ValueError(
                "Target is upstream of the source, so forward transport wraps "
                "around the ring, but no tune was given to wrap with. Pass the "
                "plane's tune for a ring; a line cannot transport backwards."
            )
        delta += float(tune)
    return 2.0 * np.pi * delta


def transport_matrix_from_twiss(
    twiss: pd.DataFrame,
    *,
    source: str,
    target: str,
    plane: str,
    tune: float | None = None,
) -> PlaneTransportMatrix:
    r"""Build the uncoupled 2x2 transfer matrix between two lattice elements.

    For one transverse plane the linear map is

    .. math::

       \begin{pmatrix}
       z_1 \\
       p_{z,1}
       \end{pmatrix}
       =
       \mathbf{R}_{0 \to 1}
       \begin{pmatrix}
       z_0 \\
       p_{z,0}
       \end{pmatrix},

    with

    .. math::

       \mathbf{R}_{0 \to 1}
       =
       \begin{pmatrix}
       R_{11} & R_{12} \\
       R_{21} & R_{22}
       \end{pmatrix}.

    Using the Courant-Snyder parameterisation, the matrix elements are

    .. math::

       R_{11} = \sqrt{\frac{\beta_1}{\beta_0}}
       \left(\cos \Delta\mu + \alpha_0 \sin \Delta\mu\right),

    .. math::

       R_{12} = \sqrt{\beta_0 \beta_1}\sin \Delta\mu,

    .. math::

       R_{21} =
       \frac{(\alpha_0-\alpha_1)\cos \Delta\mu - (1+\alpha_0\alpha_1)\sin \Delta\mu}
       {\sqrt{\beta_0\beta_1}},

    .. math::

       R_{22} = \sqrt{\frac{\beta_0}{\beta_1}}
       \left(\cos \Delta\mu - \alpha_1 \sin \Delta\mu\right).

    This is the standard transfer-matrix form used in the implementation. The
    reference requested for this parameterisation is Bernhard Holzer,
    "Transverse beam dynamics", CERN Yellow Reports: School Proceedings,
    page 109:
    https://e-publishing.cern.ch/index.php/CYRSP/article/view/1585/1314
    """
    plane_map = {
        "x": ("beta11", "alfa11", "mu1"),
        "y": ("beta22", "alfa22", "mu2"),
    }
    try:
        beta_col, alpha_col, mu_col = plane_map[plane]
    except KeyError as exc:
        raise ValueError(f"Unsupported plane {plane!r}; expected 'x' or 'y'") from exc

    _require_twiss_columns(twiss, {beta_col, alpha_col, mu_col})
    if source not in twiss.index:
        raise KeyError(f"Source element {source!r} not found in twiss table")
    if target not in twiss.index:
        raise KeyError(f"Target element {target!r} not found in twiss table")

    beta_source = float(twiss.at[source, beta_col])
    alpha_source = float(twiss.at[source, alpha_col])
    beta_target = float(twiss.at[target, beta_col])
    alpha_target = float(twiss.at[target, alpha_col])
    delta_mu = _phase_advance(
        float(twiss.at[target, mu_col]), float(twiss.at[source, mu_col]), tune
    )

    sqrt_ratio = np.sqrt(beta_target / beta_source)
    sqrt_product = np.sqrt(beta_target * beta_source)
    cos_mu = float(np.cos(delta_mu))
    sin_mu = float(np.sin(delta_mu))

    r11 = sqrt_ratio * (cos_mu + alpha_source * sin_mu)
    r12 = sqrt_product * sin_mu
    r21 = (
        (alpha_source - alpha_target) * cos_mu - (1.0 + alpha_source * alpha_target) * sin_mu
    ) / sqrt_product
    r22 = np.sqrt(beta_source / beta_target) * (cos_mu - alpha_target * sin_mu)
    return PlaneTransportMatrix(r11=r11, r12=r12, r21=r21, r22=r22)


def transport_matrix_4d_from_twiss(
    twiss: pd.DataFrame,
    *,
    source: str,
    target: str,
    tunes: tuple[float, float] | None = None,
) -> np.ndarray:
    r"""Build the uncoupled block-diagonal 4x4 (x,px,y,py) transport matrix from Twiss.

    Assembles the two 2x2 Courant-Snyder matrices (one per transverse plane,
    computed by :func:`transport_matrix_from_twiss`) into the block-diagonal form

    .. math::

       \mathbf{M}_{4 \times 4} =
       \begin{pmatrix}
       \mathbf{R}^{(x)} & \mathbf{0} \\
       \mathbf{0}       & \mathbf{R}^{(y)}
       \end{pmatrix},

    using coordinate ordering ``(x, p_x, y, p_y)``.

    This Twiss-based approximation has zero transverse-coupling blocks.

    Args:
        twiss: Twiss table indexed by element name; must contain columns
            ``beta11``, ``alfa11``, ``mu1`` (horizontal) and ``beta22``,
            ``alfa22``, ``mu2`` (vertical).
        source: Index label of the source element (e.g. ``"kicker"``).
        target: Index label of the target element (e.g. a BPM name).
        tunes: ``(qx, qy)`` used to wrap the phase advance when the target is
            upstream of the source (see :func:`_phase_advance`). ``None`` for a
            line, where an upstream target is an error.

    Returns:
        ``(4, 4)`` numpy array in coordinate order ``(x, px, y, py)``.
    """
    tune_x, tune_y = tunes if tunes is not None else (None, None)
    mx = transport_matrix_from_twiss(twiss, source=source, target=target, plane="x", tune=tune_x)
    my = transport_matrix_from_twiss(twiss, source=source, target=target, plane="y", tune=tune_y)
    mat = np.zeros((4, 4))
    mat[0, 0] = mx.r11
    mat[0, 1] = mx.r12
    mat[1, 0] = mx.r21
    mat[1, 1] = mx.r22
    mat[2, 2] = my.r11
    mat[2, 3] = my.r12
    mat[3, 2] = my.r21
    mat[3, 3] = my.r22
    return mat


def solve_kick_from_positions(
    twiss: pd.DataFrame,
    *,
    source: str,
    target: str,
    x_source: float,
    y_source: float,
    x_target: float,
    y_target: float,
    tunes: tuple[float, float] | None = None,
) -> tuple[float, float]:
    r"""Solve the instantaneous kick momenta from source/target positions.

    In each plane, solve the first row of the transport equation

    .. math::

       z_1 = R_{11} z_0 + R_{12} p_{z,0}

    for the unknown kick momentum:

    .. math::

       p_{z,0} = \frac{z_1 - R_{11} z_0}{R_{12}}.

    With closed-orbit-subtracted data, ``x_source = y_source = 0``.
    """
    tune_x, tune_y = tunes if tunes is not None else (None, None)
    matrix_x = transport_matrix_from_twiss(
        twiss, source=source, target=target, plane="x", tune=tune_x
    )
    matrix_y = transport_matrix_from_twiss(
        twiss, source=source, target=target, plane="y", tune=tune_y
    )

    if abs(matrix_x.r12) < 1e-14:
        raise ValueError(
            f"Horizontal transport from {source!r} to {target!r} is singular for kick solving"
        )
    if abs(matrix_y.r12) < 1e-14:
        raise ValueError(
            f"Vertical transport from {source!r} to {target!r} is singular for kick solving"
        )

    px = (float(x_target) - matrix_x.r11 * float(x_source)) / matrix_x.r12
    py = (float(y_target) - matrix_y.r11 * float(y_source)) / matrix_y.r12
    return px, py


def solve_kick_4d_least_squares(
    matrices: dict[str, np.ndarray],
    x_measurements: dict[str, float],
    y_measurements: dict[str, float],
    x_source: float = 0.0,
    y_source: float = 0.0,
) -> tuple[float, float]:
    r"""Reconstruct kick momenta from multiple BPMs via 4D least-squares.

    Each 4x4 matrix maps ``[x_k, px_k, y_k, py_k]`` to a BPM state. With
    ``x_k = y_k = 0``, stacking the two position equations for each BPM gives
    an overdetermined system for ``(px_k, py_k)``:

    .. math::

       x_i = M^{(i)}_{02} \, p_{x,k} + M^{(i)}_{03} \, p_{y,k},

    .. math::

       y_i = M^{(i)}_{22} \, p_{x,k} + M^{(i)}_{23} \, p_{y,k},

    Args:
        matrices: Dict mapping BPM name → (4, 4) numpy transport matrix.
        x_measurements: Dict mapping BPM name → measured x displacement.
        y_measurements: Dict mapping BPM name → measured y displacement.
        x_source: Closed-orbit-subtracted source x position (default 0).
        y_source: Closed-orbit-subtracted source y position (default 0).

    Returns:
        ``(px_kick, py_kick)`` — the reconstructed kick momenta.
    """
    rows_a: list[list[float]] = []
    rows_b: list[float] = []

    for bpm, mat in matrices.items():
        if bpm not in x_measurements or bpm not in y_measurements:
            continue
        # Remove known source-position contribution (mat[:,0]*x_k + mat[:,2]*y_k)
        x_rhs = x_measurements[bpm] - mat[0, 0] * x_source - mat[0, 2] * y_source
        y_rhs = y_measurements[bpm] - mat[2, 0] * x_source - mat[2, 2] * y_source
        # Coefficients for [px_k, py_k]
        rows_a.append([mat[0, 1], mat[0, 3]])
        rows_b.append(x_rhs)
        rows_a.append([mat[2, 1], mat[2, 3]])
        rows_b.append(y_rhs)

    a_mat = np.array(rows_a)
    b = np.array(rows_b)
    result, _, _, _ = np.linalg.lstsq(a_mat, b, rcond=None)
    return float(result[0]), float(result[1])
