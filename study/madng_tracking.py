"""MAD-NG tracking layer for the limitations study.

Everything that touches MAD-NG lives here: building/loading lattices, running
twiss, and free-betatron tracking of a single particle over many turns. The
reconstruction (:func:`tmom_recon.calculate_pz`) and scoring live in
:mod:`study.metrics`; plotting in :mod:`study.plotting`.

All lattices are driven through ``pymadng_utils``' :class:`KnobMadInterface`,
which gives sequence loading, ``run_twiss`` (indexed by element name with
``q1/q2`` headers), pattern-based observation, seeded magnet perturbations and
tune matching for free.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pymadng_utils.accelerators import LHC, PSB
from pymadng_utils.mad import KnobMadInterface
from xtrack_tools.coordinates import create_initial_conditions

from study.fodo import FODO, write_fodo_seq

REPO = Path(__file__).resolve().parents[1]
SEQ_DIR = REPO / "study" / "sequences"
RING_SEQ_DIR = REPO / "tests" / "data" / "sequences"

PROTON_MASS_GEV = 0.93827208816  # proton rest mass [GeV/c^2]
C_GEV_PER_TM = 0.299792458  # p[GeV/c] = C * B[T] * rho[m]


_LUA_CLASS = {
    "a": "[A-Za-z]",
    "d": r"\d",
    "w": "[A-Za-z0-9]",
    "s": r"\s",
    "l": "[a-z]",
    "u": "[A-Z]",
}


def lua_to_regex(pattern: str) -> str:
    """Translate a MAD-NG (Lua) string pattern to an equivalent Python regex.

    ``accelerator.bpm_pattern`` is a Lua pattern (e.g. the PSB ``^BR3%.BPM.*3$``),
    fine for MAD-NG ``observe`` but wrong if fed straight to :mod:`re` — ``%.``
    means a literal dot in Lua but a literal ``%`` in regex. We only need the few
    tokens the accelerator patterns use: ``%`` escapes (``%.`` -> ``\\.``) and the
    character classes ``%d/%a/%w/%s/%l/%u``.
    """
    out, i = [], 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "%" and i + 1 < len(pattern):
            nxt = pattern[i + 1]
            out.append(_LUA_CLASS.get(nxt, "\\" + nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def bpm_regex(iface: KnobMadInterface) -> re.Pattern:
    """Compiled Python regex matching the accelerator's BPM names."""
    return re.compile(lua_to_regex(iface.accelerator.bpm_pattern))


def energy_from_rho(rho: float, *, b_field: float = 1.0) -> float:
    """Kinetic energy [GeV] of a proton bent at radius ``rho`` by field ``b_field``.

    A real magnet of fixed field bends a higher-momentum beam on a larger radius
    (``p = 0.3 * B * rho``), so when the FODO bend radius is swept the beam energy
    must follow for the dispersion/chromatic physics to stay realistic (the LHC
    sits at large rho / high energy, the PSB at small rho / low energy).
    """
    p = C_GEV_PER_TM * b_field * rho
    return float((p**2 + PROTON_MASS_GEV**2) ** 0.5 - PROTON_MASS_GEV)


# --------------------------------------------------------------------------- #
# FODO
# --------------------------------------------------------------------------- #
def make_fodo(
    n_cells: int = 12,
    *,
    qx: float = 3.18,
    qy: float = 3.22,
    kinetic_energy: float = 2.0,
    seq_path: Path | None = None,
    quad_knobs: dict[str, float] | None = None,
) -> KnobMadInterface:
    """Build the parametric FODO, load it and set ~90 deg/cell tunes.

    ``qx, qy`` near ``n_cells/4`` give ~90 deg phase advance per cell. Bends and
    sextupoles start OFF (knobs ``bangle, ksf, ksd`` = 0); use
    :func:`set_fodo_knobs` to turn them on.

    ``quad_knobs`` (``{"kqf": .., "kqd": ..}``) sets the matched quad strengths
    directly and skips the tune match — pass the dict returned by
    :meth:`KnobMadInterface.match_tunes` to rebuild an identical lattice cheaply
    (used by the many-sample error studies).
    """
    seq_path = seq_path or (SEQ_DIR / "fodo.seq")
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    write_fodo_seq(seq_path, n_cells=n_cells)
    iface = KnobMadInterface(accelerator=FODO(seq_path, kinetic_energy=kinetic_energy))
    if quad_knobs is None:
        iface.match_tunes(target_qx=qx, target_qy=qy)
    else:
        iface.set_madx_variables(**quad_knobs)
    return iface


def matched_quad_knobs(iface: KnobMadInterface) -> dict[str, float]:
    """Read the current FODO quad knob values (for cheap lattice rebuilds)."""
    return {
        "kqf": float(iface.mad["MADX['kqf']"]),
        "kqd": float(iface.mad["MADX['kqd']"]),
    }


def set_fodo_knobs(
    iface: KnobMadInterface,
    *,
    bangle: float | None = None,
    ksf: float | None = None,
    ksd: float | None = None,
) -> None:
    """Set FODO knobs (bend angle per dipole, focusing/defocusing sextupole k2)."""
    updates = {k: v for k, v in (("bangle", bangle), ("ksf", ksf), ("ksd", ksd)) if v is not None}
    if updates:
        iface.set_madx_variables(**updates)


# --------------------------------------------------------------------------- #
# Realistic rings
# --------------------------------------------------------------------------- #
def load_ring(name: str) -> KnobMadInterface:
    """Load a realistic ring (``"lhc"`` or ``"psb"``) for free-oscillation tracking."""
    if name == "lhc":
        acc = LHC(beam=1, sequence_file=RING_SEQ_DIR / "lhcb1.seq", kinetic_energy=6800.0)
    elif name == "psb":
        acc = PSB(sequence_file=RING_SEQ_DIR / "psb3_saved.seq", ring=3, kinetic_energy=0.160)
    else:
        raise ValueError(f"unknown ring {name!r}; expected 'lhc' or 'psb'")
    return KnobMadInterface(accelerator=acc)


# Per-lattice free-betatron tracking defaults (action scaled to typical beta so
# the amplitude is ~mm; nturn a few hundred). Used by the multi-lattice studies.
LATTICES = ("fodo", "lhc", "psb")
LATTICE_LABEL = {"fodo": "FODO", "lhc": "LHC B1", "psb": "PSB ring 3"}
LATTICE_TRACK = {
    "fodo": {"action": 3e-7, "nturn": 256},
    "lhc": {"action": 5e-9, "nturn": 128},
    "psb": {"action": 5e-6, "nturn": 256},
}


def build_lattice(
    name: str,
    *,
    quad_knobs: dict[str, float] | None = None,
    bangle: float | None = None,
    ksf: float | None = None,
    ksd: float | None = None,
) -> KnobMadInterface:
    """Return a fresh tracking-ready interface for ``"fodo"``/``"lhc"``/``"psb"``.

    For the FODO the optional knobs (matched quads, bend angle, sextupoles) are
    applied; the real rings ignore them (their lattice is fixed).
    """
    if name == "fodo":
        iface = make_fodo(quad_knobs=quad_knobs)
        set_fodo_knobs(iface, bangle=bangle, ksf=ksf, ksd=ksd)
        return iface
    return _ensure_perturbation_families(load_ring(name))


# Default per-family perturbation metadata for rings that don't define their own
# (e.g. PSB): perturb every quad/dipole/sextupole of the relevant MAD kind. The
# RMS is supplied explicitly by the caller, so only the family keys matter here.
_DEFAULT_FAMILIES = {
    "q": {"default_rel_std": 1e-4},
    "d": {"default_rel_std": 1e-4},
    "s": {"default_rel_std": 1e-4},
}


def _ensure_perturbation_families(iface: KnobMadInterface) -> KnobMadInterface:
    """Give an accelerator default perturbation families if it defines none."""
    acc = iface.accelerator
    if not acc.get_perturbation_families():
        acc.get_perturbation_families = lambda: dict(_DEFAULT_FAMILIES)
    return iface


# --------------------------------------------------------------------------- #
# Twiss + tracking
# --------------------------------------------------------------------------- #
def bpm_twiss(iface: KnobMadInterface, tws: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the model twiss reduced to BPM rows (indexed by name, with q1/q2)."""
    if tws is None:
        tws = iface.run_twiss(observe=0, chrom=True)
    pattern = bpm_regex(iface)
    mask = tws.index.to_series().astype(str).map(lambda s: bool(pattern.match(s)))
    return tws.loc[mask].copy()


def track_free(
    iface: KnobMadInterface,
    *,
    action: float = 2e-7,
    angle: float = 0.1,
    nturn: int = 256,
    dp: float = 0.0,
) -> pd.DataFrame:
    """Track one particle for ``nturn`` turns and return per-BPM turn-by-turn data.

    The particle is launched as a betatron oscillation of Courant-Snyder action
    ``action`` (same in both planes) *around the closed orbit* at the sequence
    start, using :func:`xtrack_tools.coordinates.create_initial_conditions`. For
    ``dp != 0`` the twiss (hence the closed orbit) is computed at that momentum
    deviation, so the particle sits on the dispersive off-momentum closed orbit
    rather than being artificially displaced from it.

    Returns a DataFrame with ``name, turn, x, px, y, py, pt`` at every observed
    BPM. ``px, py`` are the tracked truth used to score the reconstruction.
    """
    pt = iface.dp2pt(dp)
    # Off-momentum closed orbit + optics at the launch point.
    co_tws = iface.run_twiss(observe=0, chrom=True, deltap=dp)
    ic = create_initial_conditions(
        action=action, angle=angle, twiss_data=co_tws, kick_plane="xy", starting_bpm="$start"
    )
    x0 = [float(ic["x"]), float(ic["px"]), float(ic["y"]), float(ic["py"]), 0.0, pt]

    iface.observe(iface.accelerator.bpm_pattern)
    # Raw Lua so only the first return (the table) is captured: track may return a
    # variable number of values (e.g. extra lost-particle info on a perturbed/
    # unstable lattice), which would break a fixed multi-key unpack.
    x0s = ", ".join(repr(float(v)) for v in x0)
    iface.mad.send(
        f"py_trk_tbl = track {{sequence=loaded_sequence, X0={{{x0s}}}, nturn={nturn}, observe=1}}"
    )
    trk = iface.mad.py_trk_tbl.to_df()
    pattern = bpm_regex(iface)
    mask = trk["name"].astype(str).map(lambda s: bool(pattern.match(s)))
    cols = ["name", "turn", "x", "px", "y", "py", "pt"]
    out = trk.loc[mask, cols].reset_index(drop=True)
    out["name"] = out["name"].astype(str)
    out["turn"] = out["turn"].astype(int)
    return out
