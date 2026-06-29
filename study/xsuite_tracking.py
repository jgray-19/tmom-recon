"""xsuite tracking layer for the limitations study (drop-in for ``madng_tracking``).

MAD-NG tracking is correct but slow, which caps how many turns each study can
afford. xsuite tracks the same lattices ~100x faster, so the noise / SVD studies
can use thousands of turns. This module mirrors the public surface of
:mod:`study.madng_tracking` (``make_fodo``, ``build_lattice``, ``bpm_twiss``,
``track_free``, ``load_ring`` ...) so a study switches backend by changing only
its import line.

Lattices are MAD-X sequences loaded into an :class:`xtrack.Line` via
``xtrack_tools`` (FODO written by :func:`study.fodo.write_fodo_seq`; LHC/PSB from
``tests/data/sequences``). Tracking is free betatron motion (no AC dipole) around
the (off-momentum) closed orbit using
:func:`xtrack_tools.tracking.run_tracking_without_ac_dipole`.

Two conventions matter for parity with the reconstruction:

* **Full-tune twiss headers.** :func:`xtrack_tools.monitors.xsuite_tws_to_ng`
  stores the *fractional* tune in ``q1/q2``, but the neighbour-pair phase matrix
  (:func:`tmom_recon.physics.bpm_phases.phase_advance_matrix_from_tws`) needs the
  total betatron phase, which is the *full* tune (matching the cumulative ``mu``
  range). :func:`bpm_twiss` overwrites the headers with the full tune.
* **delta, not pt.** xsuite works in ``delta = dp/p`` and its twiss dispersion
  (``dx`` ...) is per ``delta``. The reconstruction is self-consistent in that
  same convention, so :meth:`XLattice.dp2pt` returns ``dp`` unchanged — the value
  handed to ``pt_override`` is the tracked ``delta``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xpart as xp
import xtrack as xt
from xtrack_tools.env import create_xsuite_environment
from xtrack_tools.monitors import process_tracking_data, xsuite_tws_to_ng
from xtrack_tools.tracking import run_tracking_without_ac_dipole

from study.fodo import write_fodo_seq
from study.madng_tracking import (  # reuse pure-Python constants/helpers (no MAD)
    LATTICE_LABEL,
    LATTICES,
    energy_from_rho,
)

__all__ = [
    "LATTICES",
    "LATTICE_LABEL",
    "LATTICE_TRACK",
    "XLattice",
    "bpm_twiss",
    "build_lattice",
    "energy_from_rho",
    "load_ring",
    "make_fodo",
    "matched_quad_knobs",
    "set_fodo_knobs",
    "track_free",
]

REPO = Path(__file__).resolve().parents[1]
SEQ_DIR = REPO / "study" / "sequences"
RING_SEQ_DIR = REPO / "tests" / "data" / "sequences"
PROTON_MASS_GEV = 0.93827208816

# xsuite tracks fast, so afford many more turns than the MAD-NG defaults. The LHC
# (563 BPMs) is the only slow one (~0.1 s/turn), so it gets fewer.
LATTICE_TRACK = {
    "fodo": {"action": 3e-7, "nturn": 1024},
    "lhc": {"action": 5e-9, "nturn": 256},
    "psb": {"action": 5e-6, "nturn": 1024},
}

# Per-lattice BPM selection: (regex on UPPER-case twiss/track names,
# regex on lower-case line element names for the monitor selection).
_BPM_PATTERNS = {
    "fodo": (r"^BPM", r"bpm"),
    "lhc": (r"^BPM", r"bpm"),
    "psb": (r"^BR3\.BPM.*3$", r"br3\.bpm.*3"),
}

# Family -> (xsuite element class, strength attribute) for seeded perturbations.
_PERTURB_KIND = {
    "q": (xt.Quadrupole, "k1"),
    "d": (xt.Bend, "k0"),
    "s": (xt.Sextupole, "k2"),
}


def _particle_ref(kinetic_energy: float) -> xt.Particles:
    return xt.Particles(mass=xp.PROTON_MASS_EV, kinetic_energy0=kinetic_energy * 1e9)


class XLattice:
    """A tracking-ready xsuite line plus the metadata the studies expect.

    Exposes the small slice of the ``KnobMadInterface`` API the studies call
    (``dp2pt``, ``apply_magnet_perturbations``, ``close``) so it is a drop-in for
    the MAD-NG ``iface`` object.
    """

    def __init__(self, line: xt.Line, name: str, kinetic_energy: float) -> None:
        self.line = line
        self.name = name
        self.kinetic_energy = kinetic_energy
        up, low = _BPM_PATTERNS[name]
        self.bpm_pattern_upper = re.compile(up)
        self.bpm_pattern_line = low

    # -- API parity with KnobMadInterface ---------------------------------- #
    def dp2pt(self, dp: float) -> float:
        """Momentum coordinate the reconstruction should override with.

        xsuite is in the ``delta = dp/p`` convention (its dispersion is per
        ``delta``), so the reconstruction is self-consistent using ``delta``
        directly — no MAD-NG ``pt`` transform is applied.
        """
        return float(dp)

    def apply_magnet_perturbations(
        self,
        *,
        rel_error: float,
        seed: int,
        magnet_type: str | list[str],
    ) -> int:
        """Apply seeded Gaussian relative errors to a magnet family, in place.

        Each magnet of the requested family (``"q"`` quad ``k1``, ``"d"`` bend
        ``k0``, ``"s"`` sextupole ``k2``) is scaled by ``1 + N(0, rel_error)``, so
        ``rel_error`` is the relative-error RMS. Returns the number perturbed.
        """
        rng = np.random.default_rng(seed)
        families = [magnet_type] if isinstance(magnet_type, str) else list(magnet_type)
        n = 0
        for fam in families:
            cls, attr = _PERTURB_KIND[fam]
            for nm in self.line.element_names:
                el = self.line[nm]
                if isinstance(el, cls):
                    base = float(getattr(el, attr))
                    if base == 0.0:
                        continue
                    setattr(el, attr, base * (1.0 + float(rng.normal(0.0, rel_error))))
                    n += 1
        return n

    def close(self) -> None:  # pragma: no cover - no external process to free
        """No-op: xsuite holds no child process (kept for API parity)."""


# --------------------------------------------------------------------------- #
# FODO
# --------------------------------------------------------------------------- #
def _write_fodo_if_changed(path: Path, n_cells: int) -> Path:
    """Write the FODO sequence only when its content changes (keeps JSON cache)."""
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    new = path.with_suffix(".seq.tmp")
    write_fodo_seq(new, n_cells=n_cells)
    text = new.read_text()
    new.unlink()
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return path


def make_fodo(
    n_cells: int = 12,
    *,
    qx: float = 3.18,
    qy: float = 3.22,
    kinetic_energy: float = 2.0,
    quad_knobs: dict[str, float] | None = None,
) -> XLattice:
    """Build the parametric FODO as an xsuite line and set ~90 deg/cell tunes.

    ``quad_knobs`` (``{"kqf": .., "kqd": ..}``) sets the matched quad strengths
    directly and skips the tune match — pass the dict from
    :func:`matched_quad_knobs` to rebuild an identical lattice cheaply.
    """
    seq = _write_fodo_if_changed(SEQ_DIR / f"fodo_{n_cells}.seq", n_cells)
    env = create_xsuite_environment(
        sequence_file=seq, kinetic_energy=kinetic_energy, seq_name="fodo"
    )
    line = env["fodo"]
    if quad_knobs is None:
        line.match(
            method="4d",
            vary=[xt.Vary("kqf", step=1e-4), xt.Vary("kqd", step=1e-4)],
            targets=[xt.Target("qx", qx, tol=1e-7), xt.Target("qy", qy, tol=1e-7)],
        )
    else:
        for k, v in quad_knobs.items():
            line.vars[k] = float(v)
    return XLattice(line, "fodo", kinetic_energy)


def matched_quad_knobs(xl: XLattice) -> dict[str, float]:
    """Read the current FODO quad knob values (for cheap lattice rebuilds)."""
    return {
        "kqf": float(xl.line.vars.val["kqf"]),
        "kqd": float(xl.line.vars.val["kqd"]),
    }


def set_fodo_knobs(
    xl: XLattice,
    *,
    bangle: float | None = None,
    ksf: float | None = None,
    ksd: float | None = None,
) -> None:
    """Set FODO knobs (bend angle per dipole, focusing/defocusing sextupole k2)."""
    for k, v in (("bangle", bangle), ("ksf", ksf), ("ksd", ksd)):
        if v is not None:
            xl.line.vars[k] = float(v)


# --------------------------------------------------------------------------- #
# Realistic rings
# --------------------------------------------------------------------------- #
def load_ring(name: str) -> XLattice:
    """Load a realistic ring (``"lhc"`` or ``"psb"``) as an xsuite line."""
    if name == "lhc":
        seq, ke, seq_name = RING_SEQ_DIR / "lhcb1.seq", 6800.0, "lhcb1"
    elif name == "psb":
        seq, ke, seq_name = RING_SEQ_DIR / "psb3_saved.seq", 0.160, "psb3"
    else:
        raise ValueError(f"unknown ring {name!r}; expected 'lhc' or 'psb'")
    env = create_xsuite_environment(sequence_file=seq, kinetic_energy=ke, seq_name=seq_name)
    return XLattice(env[seq_name.lower()], name, ke)


def build_lattice(
    name: str,
    *,
    quad_knobs: dict[str, float] | None = None,
    bangle: float | None = None,
    ksf: float | None = None,
    ksd: float | None = None,
) -> XLattice:
    """Return a fresh tracking-ready :class:`XLattice` for fodo/lhc/psb."""
    if name == "fodo":
        xl = make_fodo(quad_knobs=quad_knobs)
        set_fodo_knobs(xl, bangle=bangle, ksf=ksf, ksd=ksd)
        return xl
    return load_ring(name)


# --------------------------------------------------------------------------- #
# Twiss + tracking
# --------------------------------------------------------------------------- #
def _ng_twiss(line: xt.Line, *, delta: float = 0.0) -> pd.DataFrame:
    """xsuite twiss -> MAD-NG-format DataFrame with FULL-tune headers."""
    tws = line.twiss(method="4d", **({"delta0": delta} if delta else {}))
    ng = xsuite_tws_to_ng(tws)
    ng.headers["q1"] = float(tws.qx)  # full tune, matches cumulative mu range
    ng.headers["q2"] = float(tws.qy)
    return ng


def bpm_twiss(xl: XLattice, tws: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the on-momentum model twiss reduced to BPM rows (indexed by name)."""
    ng = _ng_twiss(xl.line) if tws is None else tws
    mask = ng.index.to_series().astype(str).map(lambda s: bool(xl.bpm_pattern_upper.match(s)))
    return ng.loc[mask].copy()


def track_free(
    xl: XLattice,
    *,
    action: float = 2e-7,
    angle: float = 0.1,
    nturn: int = 256,
    dp: float = 0.0,
) -> pd.DataFrame:
    """Track one particle for ``nturn`` turns and return per-BPM turn-by-turn data.

    The particle is launched as a Courant-Snyder action-angle betatron oscillation
    *around the (off-momentum) closed orbit*: for ``dp != 0`` the optics used to
    build the initial conditions are the off-momentum twiss (``delta0=dp``) so the
    particle sits on the dispersive closed orbit, matching the MAD-NG path.

    Returns a DataFrame with ``name, turn, x, px, y, py, pt`` at every BPM;
    ``px, py`` are the tracked truth used to score the reconstruction.
    """
    ic_tws = xl.line.twiss(method="4d", **({"delta0": dp} if dp else {}))
    mon = run_tracking_without_ac_dipole(
        xl.line,
        ic_tws,
        flattop_turns=nturn,
        bpm_pattern=xl.bpm_pattern_line,
        action_list=[action],
        angle_list=[angle],
        use_diagonal_kicks=True,
        deltas=float(dp),
    )
    trk = process_tracking_data(mon, ramp_turns=0, flattop_turns=nturn, add_variance_columns=False)
    mask = trk["name"].astype(str).map(lambda s: bool(xl.bpm_pattern_upper.match(s)))
    out = trk.loc[mask, ["name", "turn", "x", "px", "y", "py"]].reset_index(drop=True)
    out["name"] = out["name"].astype(str)
    out["turn"] = out["turn"].astype(int)
    out["pt"] = float(dp)
    return out
