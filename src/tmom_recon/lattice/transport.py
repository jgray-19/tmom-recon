from __future__ import annotations

import numpy as np
import pandas as pd

#: Coupled MAD-NG Twiss columns (``twiss{coupling=true}``) the kick response needs.
_COLUMNS = ("beta11", "beta12", "beta21", "beta22", "mu1", "mu2", "ev1", "ev2")


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
                "mode's tune for a ring; a line cannot transport backwards."
            )
        delta += float(tune)
    return 2.0 * np.pi * delta


def kick_response_from_twiss(
    twiss: pd.DataFrame,
    *,
    source: str,
    target: str,
    tunes: tuple[float, float] | None = None,
) -> np.ndarray:
    r"""Position response at ``target`` to a thin kick at ``source``, including x-y coupling.

    Returns the ``(2, 2)`` matrix :math:`R` with

    .. math::

       \begin{pmatrix} x_1 \\ y_1 \end{pmatrix}
       = R \begin{pmatrix} \Delta p_{x,0} \\ \Delta p_{y,0} \end{pmatrix},
       \qquad
       R_{ab} = \sum_{k=1}^{2} \sqrt{\beta_{ak,1}\,\beta_{bk,0}}\,
       \sin\!\left(\Delta\mu_k + \phi_{ak,1} - \phi_{bk,0}\right),

    in the Mais-Ripken parameterisation MAD-NG uses: ``beta<a><k>`` is the beta
    of mode ``k`` seen in plane ``a``, ``mu<k>`` the mode phase, and each mode is
    in phase with its own plane (:math:`\phi_{kk} = 0`). Its phase in the other
    plane is :math:`\phi_{ak} = -\arg(\mathrm{ev}_k)`, where
    :math:`\mathrm{ev}_k = e^{i\nu_k}` is the eigenvector phase factor of
    Lebedev & Bogacz (2010), eq. 4.11 (MAD-NG ``madl_gphys.mad``). Without coupling
    ``beta12 = beta21 = 0`` and this is the Courant-Snyder
    :math:`R_{12} = \sqrt{\beta_0\beta_1}\sin\Delta\mu` in each plane.

    The ``betx``/``bety`` columns are not used: on a coupled lattice they are not
    the mode betas, and a plane-by-plane matrix built from them has no cross terms.

    Args:
        twiss: Coupled MAD-NG Twiss indexed by element name, with the columns in
            :data:`_COLUMNS`.
        source: Index label of the kick point (e.g. the kicker marker).
        target: Index label of the observation point (e.g. a BPM).
        tunes: ``(q1, q2)`` used to wrap the phase advance when the target is
            upstream of the source (see :func:`_phase_advance`). ``None`` for a
            line, where an upstream target is an error.

    Returns:
        ``R`` with rows ``(x, y)`` and columns ``(delta_px, delta_py)``.
    """
    missing = set(_COLUMNS).difference(twiss.columns)
    if missing:
        raise KeyError(
            f"Twiss table is missing coupled-optics columns {sorted(missing)}; "
            "run the twiss with coupling=True"
        )
    for role, element in (("Source", source), ("Target", target)):
        if element not in twiss.index:
            raise KeyError(f"{role} element {element!r} not found in twiss table")

    tune = tunes if tunes is not None else (None, None)
    response = np.zeros((2, 2))
    for mode in (1, 2):
        delta_mu = _phase_advance(
            twiss.at[target, f"mu{mode}"], twiss.at[source, f"mu{mode}"], tune[mode - 1]
        )
        amplitude = {}
        phase = {}
        for element in (source, target):
            amplitude[element] = np.sqrt(
                [float(twiss.at[element, f"beta{plane}{mode}"]) for plane in (1, 2)]
            )
            # Complex on a coupled lattice; MAD-NG writes 0 when uncoupled, where the
            # matching beta12/beta21 are 0 too. complex() also reads object columns.
            other = -np.angle(complex(twiss.at[element, f"ev{mode}"]))
            phase[element] = np.array([other, other])
            phase[element][mode - 1] = 0.0
        response += np.outer(amplitude[target], amplitude[source]) * np.sin(
            delta_mu + phase[target][:, None] - phase[source][None, :]
        )
    return response
