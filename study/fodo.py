"""Parametric FODO lattice + ``Accelerator`` descriptor for the limitations study.

The basic momentum reconstruction (:func:`tmom_recon.calculate_pz`) assumes a
perfectly linear, perfectly known lattice. To probe that assumption in a clean,
controllable way we generate a FODO ring as a MAD-X sequence with global knobs so
the same lattice can be re-used with bends / sextupoles turned on or off, and we
drive it through the ``pymadng_utils`` accelerator machinery (sequence loading,
twiss, observation, tracking and seeded magnet perturbations).

Element naming follows the family conventions ``pymadng_utils`` perturbations
expect: ``QF*/QD*`` (quadrupoles, family ``q``), ``MB*`` (bends, family ``d``),
``SF*/SD*`` (sextupoles, family ``s``), ``BPM*`` (monitors).
"""

from __future__ import annotations

from pathlib import Path

from pymadng_utils.accelerators.base import Accelerator

# Cell geometry (metres). One BPM monitor sits immediately after every quad.
CELL_LEN = 10.0
QUAD_LEN = 0.5
SEXT_LEN = 0.4
BEND_LEN = 2.0
# Default knob values: quads tuned later by ``match_tunes`` to ~90 deg/cell,
# bends and sextupoles start OFF so the baseline ring is linear and straight.
DEFAULT_KQF = 0.34
DEFAULT_KQD = -0.34


def write_fodo_seq(
    path: Path | str,
    n_cells: int = 12,
    *,
    seq_name: str = "fodo",
) -> Path:
    """Write a parametric FODO MAD-X sequence with global knobs.

    The lattice exposes the knobs ``kqf, kqd`` (quad strengths), ``bangle`` (bend
    angle per dipole, 0 = straight), and ``ksf, ksd`` (sextupole ``k2``, 0 = off).
    Strengths use deferred (``:=``) expressions so changing a knob in MAD-NG
    updates every element of that family. Returns the resolved path.
    """
    path = Path(path)
    lines: list[str] = [
        "! Auto-generated parametric FODO lattice (limitations study).",
        f"lq = {QUAD_LEN}; lb = {BEND_LEN}; ls = {SEXT_LEN}; lcell = {CELL_LEN};",
        f"kqf = {DEFAULT_KQF}; kqd = {DEFAULT_KQD};",
        "bangle = 0; ksf = 0; ksd = 0;",
        "",
        "qfclass: quadrupole, l=lq, k1:=kqf;",
        "qdclass: quadrupole, l=lq, k1:=kqd;",
        "sfclass: sextupole,  l=ls, k2:=ksf;",
        "sdclass: sextupole,  l=ls, k2:=ksd;",
        "mbclass: sbend, l=lb, angle:=bangle, k0:=bangle/lb;",
        "bpmclass: monitor, l=0;",
        "",
        f"{seq_name}: sequence, l={n_cells * CELL_LEN}, refer=entry;",
    ]
    for i in range(1, n_cells + 1):
        base = (i - 1) * CELL_LEN
        lines += [
            f"  QF.{i}:    qfclass,  at={base + 0.0:.6f};",
            f"  BPM.F.{i}: bpmclass, at={base + QUAD_LEN:.6f};",
            f"  SF.{i}:    sfclass,  at={base + QUAD_LEN:.6f};",
            f"  MB1.{i}:   mbclass,  at={base + 2.0:.6f};",
            f"  QD.{i}:    qdclass,  at={base + 5.0:.6f};",
            f"  BPM.D.{i}: bpmclass, at={base + 5.0 + QUAD_LEN:.6f};",
            f"  SD.{i}:    sdclass,  at={base + 5.0 + QUAD_LEN:.6f};",
            f"  MB2.{i}:   mbclass,  at={base + 7.0:.6f};",
        ]
    lines += ["endsequence;", ""]
    path.write_text("\n".join(lines))
    return path


class FODO(Accelerator):
    """Accelerator descriptor for the generated FODO ring.

    Only the pieces the limitations study touches are meaningful; the
    AC-dipole-related abstract members are stubbed since no AC dipole is used.
    """

    BPM_PATTERN = "^BPM"

    def __init__(
        self,
        sequence_file: Path | str,
        kinetic_energy: float = 2.0,
        bpm_pattern: str | None = None,
        particle: str = "proton",
        **kwargs,
    ) -> None:
        super().__init__(
            sequence_file=sequence_file,
            kinetic_energy=kinetic_energy,
            bpm_pattern=bpm_pattern or self.BPM_PATTERN,
            particle=particle,
            **kwargs,
        )

    @property
    def seq_name(self) -> str:
        """MAD sequence name (matches :func:`write_fodo_seq` default)."""
        return "fodo"

    @property
    def ac_dipole_name(self) -> str:  # pragma: no cover - unused, no AC dipole
        """Unused: the basic reconstruction study installs no AC dipole."""
        raise NotImplementedError("FODO study uses free oscillation, no AC dipole")

    @property
    def tune_variables(self) -> tuple[str, str]:
        """Knob names matched by :meth:`AcceleratorMadInterface.match_tunes`."""
        return "kqf", "kqd"

    @property
    def tune_integers(self) -> tuple[int, int]:
        """Integer tunes are absorbed into the matched fractional targets."""
        return 0, 0

    def get_perturbation_families(self) -> dict[str, dict[str, str | float | dict]]:
        """Family metadata so seeded magnet perturbations target the right magnets."""
        return {
            "d": {"default_rel_std": 1e-4, "pattern": r"^MB"},
            "q": {"default_rel_std": 1e-4, "pattern": r"^Q"},
            "s": {"default_rel_std": 1e-4, "pattern": r"^S"},
        }
