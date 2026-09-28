"""PSB single-kick reconstruction against MAD-NG tracking of the *same* lattice.

The xsuite contract (``tests/contracts/test_kicker_reconstruction.py``) measures
the whole chain, but its tolerance also absorbs the xsuite/MAD-NG formulation
difference.  Here the tracked machine and the reconstruction model are one MAD-NG
lattice built twice, in two separate MAD-NG processes, with identical strengths
and the same integrator (``method=6``, the ``run_twiss`` default).  Any error left
is therefore the reconstruction's own: the kick-response formula, the phase
wrapping, the first-pass split, the frame and the closed-orbit handling.

The kick is applied by the tracker itself -- the particle is stopped at the
kicker on the kick turn and ``(px, py)`` incremented -- so the truth is exact and
does not depend on any element convention.

Machine states cover what the real reconstruction meets: another working point,
gradient errors (beta beating), quadrupole tilts (x-y coupling), bend and
quadrupole-offset errors (a closed orbit in both planes), off-momentum beams, and
all of them together.  Two kicker locations give two different source phases.

Nonlinearity is the only thing a linear transport cannot reproduce, and it is
measured rather than tolerated: the kick error must fall with the kick amplitude
(:func:`test_residual_error_is_nonlinear_only`), so the absolute bound at a small
kick really is a bound on the formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import numpy as np
import pandas as pd
import pytest
from pymadng_utils.accelerators import PSB

from tmom_recon import KickerConfig, ModelDetails, calculate_kicker_pz
from tmom_recon.acd.madng_driver import ACDipoleMadDriver

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.psb]

_KICK_TURN = 5
_N_TURNS = 12
_KICK = 1e-6
_PLANES = {"horizontal": (1.0, 0.0), "vertical": (0.0, 1.0), "diagonal": (1.0, -0.7)}
_KICKERS = ("BR3.DHZ12L4", "BR3.DVT5L4")

#: Relative kick error from the first BPM alone; worst measured 5.8e-7.
_KICK_REL_TOL = 2e-6
#: Reported state at the kicker [m, rad]; worst measured ~6e-13, the MAD-NG twiss/track floor.
_ORBIT_TOL = 2e-12


@dataclass(frozen=True)
class MachineState:
    """One PSB configuration, applied identically to tracker and model."""

    delta_p: float = 0.0
    tune_knob_scale: tuple[float, float] = (1.0, 1.0)
    quad_dk1l_rms: float = 0.0
    quad_tilt_rms: float = 0.0
    quad_dxdy_rms: float = 0.0
    bend_dk0l_rms: float = 0.0
    seed: int = 1
    notes: str = field(default="", compare=False)


STATES = {
    "nominal": MachineState(),
    "working_point": MachineState(tune_knob_scale=(1.02, 0.985)),
    "gradient_errors": MachineState(quad_dk1l_rms=2e-3),
    "coupled": MachineState(quad_tilt_rms=5e-3),
    "closed_orbit": MachineState(bend_dk0l_rms=2e-4, quad_dxdy_rms=2e-4),
    "off_momentum_plus": MachineState(delta_p=1e-3),
    "all_off_momentum_minus": MachineState(
        delta_p=-1e-3,
        tune_knob_scale=(0.99, 1.01),
        quad_dk1l_rms=2e-3,
        quad_tilt_rms=5e-3,
        quad_dxdy_rms=2e-4,
        bend_dk0l_rms=2e-4,
        seed=7,
    ),
    "all_on_momentum": MachineState(
        quad_dk1l_rms=2e-3,
        quad_tilt_rms=5e-3,
        quad_dxdy_rms=2e-4,
        bend_dk0l_rms=2e-4,
        seed=11,
    ),
}


@pytest.fixture(scope="module")
def accelerator(seq_psb3) -> PSB:
    return PSB(sequence_file=seq_psb3, ring=3, kinetic_energy=0.160)


@cache
def _lattice_names(sequence_file) -> tuple[tuple[str, ...], tuple[str, ...]]:
    driver = ACDipoleMadDriver(
        accelerator=PSB(sequence_file=sequence_file, ring=3, kinetic_energy=0.160), pt=0.0
    )
    names = [str(n).upper() for n in driver.run_twiss(observe=0).index]
    quads = tuple(dict.fromkeys(n for n in names if n.startswith("BR.Q")))
    bends = tuple(dict.fromkeys(n for n in names if n.startswith("BR.BHZ")))
    return quads, bends


def _model_inputs(accelerator: PSB, state: MachineState):
    """The strengths and tune knobs that define *state*, for both lattices."""
    quads, bends = _lattice_names(accelerator.sequence_file)
    rng = np.random.default_rng(state.seed)
    strengths: dict[str, float] = {}
    for quad in quads:
        if state.quad_dk1l_rms:
            strengths[f"{quad}.dk1l"] = float(rng.normal(0.0, state.quad_dk1l_rms))
        if state.quad_tilt_rms:
            strengths[f"{quad}.tilt"] = float(rng.normal(0.0, state.quad_tilt_rms))
        if state.quad_dxdy_rms:
            strengths[f"{quad}.dx"] = float(rng.normal(0.0, state.quad_dxdy_rms))
            strengths[f"{quad}.dy"] = float(rng.normal(0.0, state.quad_dxdy_rms))
    if state.bend_dk0l_rms:
        for bend in bends:
            strengths[f"{bend}.dk0l"] = float(rng.normal(0.0, state.bend_dk0l_rms))
    probe = ACDipoleMadDriver(accelerator=accelerator, pt=0.0)
    base = probe.mad.recv_vars(*[f"MADX['{v}']" for v in accelerator.tune_variables])
    tune_knobs = {
        name: float(value) * scale
        for name, value, scale in zip(accelerator.tune_variables, base, state.tune_knob_scale)
    }
    return strengths or None, tune_knobs


def _track(accelerator, strengths, tune_knobs, *, pt, kicker, kick):
    """Track one particle on its closed orbit; kick it at *kicker* on turn ``_KICK_TURN``.

    Returns the BPM readings as a ``name/turn/x/y`` frame and the particle's
    phase-space state at the kicker just before the kick.  Turn labels follow the
    BPM acquisition: the kick turn is the turn whose ``#s -> #e`` pass contains
    the kick.
    """
    driver = ACDipoleMadDriver(
        accelerator=accelerator,
        pt=pt,
        magnet_strengths=strengths,
        tune_knobs=tune_knobs,
        observed_elements=kicker,
    )
    # The particle must start on its closed orbit, or it oscillates before the kick.
    start = driver.run_twiss(observe=0, coupling=True, pt=pt, cotol=1e-12).iloc[0]
    driver.mad.send(f"""
local seq = loaded_sequence
local ik = seq:index_of('{kicker}')
local names, turns, xs, ys = {{}}, {{}}, {{}}, {{}}
local function keep(mt, label)
  for k = 1, #mt do
    local row = mt[k]
    if row.name:upper():match('^BR3%.BPM.*3$') then
      names[#names + 1] = row.name
      turns[#turns + 1] = label(row.turn)
      xs[#xs + 1] = row.x
      ys[#ys + 1] = row.y
    end
  end
end
local function go(range, X, nturn)
  local mt, flw = MAD.track{{sequence=seq, range=range, X0=X, nturn=nturn,
                              observe=0, method=6, beam=seq.beam}}
  local p = flw[1]
  return mt, {{x=p.x, px=p.px, y=p.y, py=p.py, t=p.t, pt=p.pt}}
end
local X = {{x={float(start.x)!r}, px={float(start.px)!r}, y={float(start.y)!r}, py={float(start.py)!r}, t=0, pt={float(pt)!r}}}
local mt
mt, X = go('#s/#e', X, {_KICK_TURN - 1}); keep(mt, function(t) return t end)
mt, X = go('#s/{kicker}', X, 1);          keep(mt, function() return {_KICK_TURN} end)
local before = {{X.x, X.px, X.y, X.py}}
X.px = X.px + {float(kick[0])!r}; X.py = X.py + {float(kick[1])!r}
mt, X = go('{kicker}/#e', X, 1);          keep(mt, function() return {_KICK_TURN} end)
mt, X = go('#s/#e', X, {_N_TURNS - _KICK_TURN}); keep(mt, function(t) return t + {_KICK_TURN} end)
py:send(table.concat(names, ','))
py:send(turns, true); py:send(xs, true); py:send(ys, true); py:send(before, true)
""")
    names = driver.mad.recv().upper().split(",")
    turns = np.asarray(driver.mad.recv(), dtype=int)
    x = np.asarray(driver.mad.recv(), dtype=float)
    y = np.asarray(driver.mad.recv(), dtype=float)
    before = np.asarray(driver.mad.recv(), dtype=float)
    data = pd.DataFrame({"name": names, "turn": turns, "x": x, "y": y})
    return data, dict(zip(("x", "px", "y", "py"), before))


@cache
def _first_bpm_s(sequence_file) -> dict[str, float]:
    driver = ACDipoleMadDriver(
        accelerator=PSB(sequence_file=sequence_file, ring=3, kinetic_energy=0.160),
        pt=0.0,
        observed_elements=list(_KICKERS),
    )
    tws = driver.run_twiss(observe=0)
    return {str(n).upper(): float(v) for n, v in tws["s"].items()}


def _first_bpm_after(accelerator: PSB, kicker: str, data: pd.DataFrame) -> str:
    s = _first_bpm_s(accelerator.sequence_file)
    after = [n for n in data["name"].unique() if s[n] > s[kicker]]
    return min(after, key=s.__getitem__)


def _reconstruct(accelerator, state: MachineState, kicker: str, kick: tuple[float, float]):
    strengths, tune_knobs = _model_inputs(accelerator, state)
    pt = accelerator.dp2pt(state.delta_p)
    data, before = _track(accelerator, strengths, tune_knobs, pt=pt, kicker=kicker, kick=kick)
    # The operator's setting-zero orbit: the same machine, on momentum, unkicked.
    zero, _ = _track(accelerator, strengths, tune_knobs, pt=0.0, kicker=kicker, kick=(0.0, 0.0))
    orbit_zero = zero.groupby("name", sort=False)[["x", "y"]].mean()
    result = calculate_kicker_pz(
        data,
        ModelDetails(accelerator, pt=pt, magnet_strengths=strengths, tune_knobs=tune_knobs),
        KickerConfig(kicker=kicker, n_turns_free=_KICK_TURN),
        closed_orbit_at_zero=orbit_zero,
        orbit_mode="dynamic",
    )
    # The frame restores the model's zero-momentum orbit, so with a model that
    # matches the machine the reported state is the absolute pre-kick orbit.
    return result, data, before


@pytest.mark.parametrize("plane", tuple(_PLANES))
@pytest.mark.parametrize("kicker", _KICKERS)
@pytest.mark.parametrize("state", tuple(STATES), ids=str)
def test_psb_kick_is_recovered_exactly(accelerator, state: str, kicker: str, plane: str) -> None:
    kick = tuple(_KICK * c for c in _PLANES[plane])
    result, data, truth = _reconstruct(accelerator, STATES[state], kicker, kick)
    row, fitted = result.iloc[0], result.attrs["kick"]

    assert int(row["turn"]) == _KICK_TURN
    assert fitted.bpm == _first_bpm_after(accelerator, kicker, data)
    error = max(abs(fitted.delta_px - kick[0]), abs(fitted.delta_py - kick[1]))
    assert error / _KICK < _KICK_REL_TOL, f"relative kick error {error / _KICK:.3e}"

    # The reported state is the closed orbit the beam rode, in the dynamic frame,
    # plus the kick -- all four coordinates, from the tracked particle itself.
    for coordinate, kicked in (("x", 0.0), ("px", kick[0]), ("y", 0.0), ("py", kick[1])):
        assert row[coordinate] - kicked == pytest.approx(truth[coordinate], abs=_ORBIT_TOL), (
            coordinate
        )


@pytest.mark.parametrize("state", ("nominal", "all_off_momentum_minus"), ids=str)
def test_residual_error_is_nonlinear_only(accelerator, state: str) -> None:
    """Scaling the kick by 10 must scale the relative error by ~10: a linear
    formulation error would leave the relative error unchanged."""
    relative = []
    for amplitude in (1e-5, 1e-4):
        kick = (amplitude, -0.7 * amplitude)
        result, _data, _truth = _reconstruct(accelerator, STATES[state], _KICKERS[0], kick)
        fitted = result.attrs["kick"]
        relative.append(
            max(abs(fitted.delta_px - kick[0]), abs(fitted.delta_py - kick[1])) / amplitude
        )
    assert relative[1] > 5 * relative[0], relative
