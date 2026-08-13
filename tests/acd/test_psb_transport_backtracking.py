"""Is MAD-NG state transport through the PSB lattice consistent in both directions?

The AC-dipole reconstruction moves phase-space states between BPMs and the
``<acd>_before`` / ``<acd>_after`` markers in both directions, and a downstream
optimiser then tracks those states onward. If the *backward* transport is not the
exact inverse of the forward one, every reconstructed initial condition inherits
that error, and an optics fit run on them is fitting the transport rather than
the quadrupoles.

This is a **MAD-NG-only** check. xsuite is used solely to generate a set of
realistic phase-space states to transport; it is never itself back-tracked, and
no result here is compared against it. That keeps the test measuring one thing —
whether MAD-NG's backward map inverts its forward map — rather than folding in a
two-code agreement floor that has nothing to do with directionality.

Both checks cover the **whole ring**, not just the AC-dipole neighbourhood:

1. :func:`test_transport_round_trip_closes_on_every_ring_leg` — every consecutive
   ``BPM_i -> BPM_i+1 -> BPM_i`` leg, including the one containing the AC dipole,
   so all 32 half-bends and all 48 quadrupoles are crossed in both directions.
   Localises any failure to a specific leg.
2. :func:`test_full_ring_chained_transport_closes` — the same legs chained: all
   the way forward around the ring BPM by BPM, then all the way back. Errors that
   are individually below tolerance but systematic would accumulate here, and a
   per-leg check would not see them.
3. :func:`test_backward_transport_jacobian_inverts_the_forward_one` — the
   *first-order* statement. Closing on a state is not enough: an optimiser
   propagates sensitivities through these transports, so ``J_backward`` must be
   ``J_forward^-1``, and both must be symplectic. Also per ring leg.

Both run with and without a distorted closed orbit (bend field errors mirrored
into the model), since a closed orbit is where directionality would most
plausibly break.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.psb_tracking import ACD_ELEMENT, DRIVEN_TUNES, build_psb_tracking_setup

LOGGER = logging.getLogger(__name__)

BEND_ERROR_RMS = 8e-4
BEND_ERROR_SEED = 7
ACD_DRIVEN_TUNES = (0.18, DRIVEN_TUNES[1])
COORDS = ("x", "px", "y", "py")

# Transport runs through the same MAD-NG maps in both directions, so the round
# trip should close at double precision.
ROUNDTRIP_TOL = 1e-12
# The chained round trip crosses every ring leg twice, so it accumulates over ~30
# transports rather than 2; allow for that growth while staying far below anything
# physically meaningful.
CHAINED_TOL = 1e-11


def _setup(data_dir, *, with_orbit: bool):
    return build_psb_tracking_setup(
        data_dir,
        delta_p=0.0,
        driven_tunes=ACD_DRIVEN_TUNES,
        bend_error_rms=BEND_ERROR_RMS if with_orbit else 0.0,
        bend_error_seed=BEND_ERROR_SEED,
        apply_bend_errors_to_model=True,
    )


def _states_at(tracking_df, name: str) -> np.ndarray:
    """Per-turn ``(x, px, y, py)`` at *name*, turn-sorted."""
    rows = tracking_df.loc[tracking_df["name"] == name].sort_values("turn")
    assert not rows.empty, f"no tracked rows for {name}"
    return rows[list(COORDS)].to_numpy(dtype=float)


def _ring_bpms(setup) -> list[str]:
    """BPM names present in both the model twiss and the tracking data, by ``s``."""
    tws = setup["tws"]
    tracked = {str(name).upper() for name in setup["tracking_df"]["name"].unique()}
    bpms = tws.loc[tws.index.str.contains("BPM", case=False)].sort_values("s")
    names = [str(name) for name in bpms.index if str(name).upper() in tracked]
    assert len(names) >= 8, f"expected the full PSB BPM set, got {len(names)}"
    return names


def _acd_leg(setup, bpms: list[str]) -> int:
    """Index of the leg ``bpms[i] -> bpms[i+1]`` that contains the AC dipole."""
    tws = setup["tws"]
    acd_s = float(setup["model"].twiss_elements.loc[ACD_ELEMENT.upper(), "s"])
    positions = [float(tws.loc[name, "s"]) for name in bpms]
    for index in range(len(bpms) - 1):
        if positions[index] <= acd_s < positions[index + 1]:
            return index
    # The AC dipole sits outside [first BPM, last BPM], i.e. on the wrap-around leg.
    return len(bpms) - 1


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    return {coord: float(np.abs(a[:, i] - b[:, i]).max()) for i, coord in enumerate(COORDS)}


def _worst(diffs: dict[str, float]) -> tuple[str, float]:
    coord = max(diffs, key=lambda key: diffs[key])
    return coord, diffs[coord]


@pytest.mark.slow
@pytest.mark.parametrize("with_orbit", [False, True], ids=["flat_orbit", "distorted_orbit"])
def test_transport_round_trip_closes_on_every_ring_leg(with_orbit, data_dir) -> None:
    """``BPM_i -> BPM_i+1 -> BPM_i`` returns the original state, all the way round.

    Covers every leg of the ring including the one containing the AC dipole, so
    every bend and quadrupole is crossed forward and backward. Only the four
    transverse coordinates are carried between legs: ``track_particles`` requires
    ``t == 0`` on input, so the longitudinal coordinate cannot be fed back and is
    not part of the round trip.
    """
    setup = _setup(data_dir, with_orbit=with_orbit)
    model = setup["model"]
    bpms = _ring_bpms(setup)
    worst_overall = ("", 0.0, "")

    for index in range(len(bpms) - 1):
        source, target = bpms[index], bpms[index + 1]
        original = _states_at(setup["tracking_df"], source.upper())
        forward = model.track_particles(source, target, original, direction=+1)
        back = model.track_particles(target, source, forward[:, :4], direction=-1)
        diffs = _max_abs_diff(original, back[:, :4])
        coord, value = _worst(diffs)
        if value > worst_overall[1]:
            worst_overall = (f"{source} -> {target}", value, coord)
        for name, diff in diffs.items():
            assert diff < ROUNDTRIP_TOL, (
                f"round trip {source} -> {target} -> {source} did not close in {name}: "
                f"max|diff|={diff:.3e}. The backward transport is not the inverse of "
                "the forward one."
            )

    LOGGER.info(
        "[orbit=%s] %d ring legs round-tripped; worst %s in %s at %.2e",
        with_orbit,
        len(bpms) - 1,
        worst_overall[2],
        worst_overall[0],
        worst_overall[1],
    )


@pytest.mark.slow
@pytest.mark.parametrize("with_orbit", [False, True], ids=["flat_orbit", "distorted_orbit"])
def test_full_ring_chained_transport_closes(with_orbit, data_dir) -> None:
    """Forward around the whole ring BPM by BPM, then all the way back, is the identity.

    The per-leg check above would pass even if every leg carried the same small
    systematic bias; chaining the legs makes such a bias accumulate. Nothing is
    compared against xsuite — the starting states merely have to be realistic,
    and the assertion is purely that MAD-NG's backward transport undoes its own
    forward transport over the entire lattice.
    """
    setup = _setup(data_dir, with_orbit=with_orbit)
    model = setup["model"]
    bpms = _ring_bpms(setup)
    original = _states_at(setup["tracking_df"], bpms[0].upper())

    state = original
    for index in range(len(bpms) - 1):
        state = model.track_particles(bpms[index], bpms[index + 1], state[:, :4], direction=+1)
    for index in range(len(bpms) - 1, 0, -1):
        state = model.track_particles(bpms[index], bpms[index - 1], state[:, :4], direction=-1)

    diffs = _max_abs_diff(original, state[:, :4])
    coord, value = _worst(diffs)
    LOGGER.info(
        "[orbit=%s] chained over %d legs each way; worst %s=%.2e",
        with_orbit,
        len(bpms) - 1,
        coord,
        value,
    )
    for name, diff in diffs.items():
        assert diff < CHAINED_TOL, (
            f"chaining {len(bpms) - 1} forward legs and {len(bpms) - 1} backward legs "
            f"around the ring did not close in {name}: max|diff|={diff:.3e}"
        )


# Central-difference steps for the transfer-matrix probe: large enough that the
# state difference is far above the 1e-15 tracking noise, small enough that the
# non-linear remainder over one BPM-to-BPM leg stays negligible.
FD_STEP = np.array([1e-6, 1e-7, 1e-6, 1e-7])
# Jacobian entries are O(1)-O(10) for these legs; 1e-7 is ~1e-8 relative, which is
# the central-difference floor at these step sizes, not a physics limit.
JACOBIAN_TOL = 1e-7


def _transfer_matrix(
    model, source: str, target: str, *, direction: int, about: np.ndarray | None = None
) -> np.ndarray:
    """Central-difference Jacobian d(state at target)/d(state at source), 4x4.

    Column ``j`` is the response of all four coordinates to a step in coordinate
    ``j``, evaluated about the reference point *about* (default: the origin).

    The expansion point matters. PSB bends use exact maps, so the transport is
    genuinely non-linear and its Jacobian varies across phase space; a state
    riding a mm-scale closed orbit does not see the same first-order matrix as
    one on axis.
    """
    reference = np.zeros(4) if about is None else np.asarray(about, dtype=float)
    probes = []
    for index in range(4):
        for sign in (+1, -1):
            state = reference.copy()
            state[index] += sign * FD_STEP[index]
            probes.append(state)
    out = model.track_particles(source, target, np.array(probes), direction=direction)[:, :4]
    return np.column_stack([(out[2 * i] - out[2 * i + 1]) / (2.0 * FD_STEP[i]) for i in range(4)])


@pytest.mark.slow
@pytest.mark.parametrize("with_orbit", [False, True], ids=["flat_orbit", "distorted_orbit"])
def test_backward_transport_jacobian_inverts_the_forward_one(with_orbit, data_dir) -> None:
    """The *derivative* of the backward map is the inverse of the forward derivative.

    Closing the round trip on a state is necessary but not sufficient: an
    optimiser propagates first-order sensitivities through these transports, so
    what matters is that ``J_backward == J_forward^-1``. A backward map that
    happened to return the right state while carrying a transposed or otherwise
    inconsistent Jacobian would leave the states right and every gradient wrong.

    Checked on every ring leg, in both directions, together with symplecticity of
    each matrix — the property an optimiser's covariance propagation relies on,
    and the one a sign error in the backward map would break first.

    The two Jacobians must be evaluated at *corresponding* points: the forward one
    about a reference state ``P``, the backward one about ``f(P)``, its forward
    image. PSB bends use exact maps, so the transport is non-linear and probing
    both about the origin compares matrices from different points in phase space.
    On a flat orbit that distinction is invisible (the origin maps to itself), but
    with a 5.5 mm closed orbit it produces a spurious ~6e-5 residual — an artefact
    of the probe, not of the backward map.
    """
    setup = _setup(data_dir, with_orbit=with_orbit)
    model = setup["model"]
    bpms = _ring_bpms(setup)
    identity = np.eye(4)
    symplectic_form = np.array(
        [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0]]
    )
    worst_inverse = 0.0
    worst_symplectic = 0.0
    worst_det = 0.0

    for index in range(len(bpms) - 1):
        source, target = bpms[index], bpms[index + 1]
        # Expand about a real trajectory: the mean tracked state at the source,
        # and its forward image at the target.
        reference = _states_at(setup["tracking_df"], source.upper()).mean(axis=0)
        image = model.track_particles(source, target, reference.reshape(1, 4), direction=+1)[0, :4]
        forward = _transfer_matrix(model, source, target, direction=+1, about=reference)
        backward = _transfer_matrix(model, target, source, direction=-1, about=image)

        residual = float(np.abs(backward @ forward - identity).max())
        worst_inverse = max(worst_inverse, residual)
        assert residual < JACOBIAN_TOL, (
            f"J_backward @ J_forward is not the identity on {source} -> {target}: "
            f"max|residual|={residual:.3e}. The backward transport returns the right "
            "state but the wrong first-order sensitivity."
        )

        for label, matrix in (("forward", forward), ("backward", backward)):
            violation = float(np.abs(matrix.T @ symplectic_form @ matrix - symplectic_form).max())
            worst_symplectic = max(worst_symplectic, violation)
            assert violation < JACOBIAN_TOL, (
                f"the {label} transfer matrix on {source} -> {target} is not "
                f"symplectic: max|J^T S J - S|={violation:.3e}"
            )
            determinant = float(np.linalg.det(matrix))
            worst_det = max(worst_det, abs(determinant - 1.0))
            assert determinant == pytest.approx(1.0, abs=JACOBIAN_TOL), (
                f"the {label} transfer matrix on {source} -> {target} has "
                f"determinant {determinant:.12f}"
            )

    LOGGER.info(
        "[orbit=%s] %d legs: worst |J_back @ J_fwd - I|=%.2e, |J^T S J - S|=%.2e, |det-1|=%.2e",
        with_orbit,
        len(bpms) - 1,
        worst_inverse,
        worst_symplectic,
        worst_det,
    )
