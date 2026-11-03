"""Real OMC3 optics measurement from in-memory simulated BPM acquisitions.

This is the small, test-owned form of the production pipeline in
``psb_md.hio_analysis`` / ``psb_md.acd_workflow``, and it mirrors that
pipeline's momentum handling deliberately.

Each momentum acquisition is analysed by Harpy against **its own** model,
natural tunes and known momentum, and then gets **its own optics run**, so the
phase it produces describes the lattice at that momentum. Dispersion is the one
quantity that cannot be measured that way -- ``D = dCO/ddelta`` needs the
momentum spread -- so it comes from a single combined run over every lin file
and is copied into each per-momentum directory afterwards.

Pooling all momenta into one optics run instead makes omc3 average the phase
across them (``omc3.optics_measurements.phase`` falls back to every input file
when ``analyse_dpp`` is unset, and warns that it is doing so). The resulting
phase belongs to no single momentum, and using it off momentum is a systematic
the reconstruction cannot see. ``psb_md.acd_workflow`` runs phase analysis per
orbit group for exactly this reason; this module now does the same.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from omc3.hole_in_one import hole_in_one_entrypoint
from turn_by_turn.structures import TbtData, TransverseData

#: Measured by the combined run and copied into every per-momentum directory,
#: mirroring ``psb_md.acd_workflow._copy_combined_dispersion_to_phase_dir``.
DISPERSION_MEASUREMENT_FILES = ("dispersion_x.tfs", "dispersion_y.tfs")


@dataclass(frozen=True)
class Acquisition:
    """One momentum group, with the model and tunes that describe it.

    The analogue of ``psb_md.hio_analysis.OrbitHarpyInput``: the AC-dipole drive
    is a fixed frequency, so ``driven_tunes`` is common, but the natural tune and
    the model both move with momentum and must follow the acquisition.
    """

    data: pd.DataFrame
    pt: float
    model_dir: Path
    natural_tunes: tuple[float, float]
    driven_tunes: tuple[float, float]


def _tbt_data(data: pd.DataFrame, source: Path) -> TbtData:
    names = list(dict.fromkeys(data["name"].astype(str)))
    turns = sorted(int(turn) for turn in data["turn"].unique())

    def matrix(plane: str) -> pd.DataFrame:
        return data.pivot(index="name", columns="turn", values=plane).reindex(
            index=names, columns=turns
        )

    return TbtData(
        matrices=[TransverseData(X=matrix("x"), Y=matrix("y"))],
        nturns=len(turns),
        bunch_ids=[0],
        meta={"file": str(source)},
    )


@contextmanager
def _known_momenta(values: list[float]):
    """Pin OMC3's per-file momentum instead of re-estimating from its model."""
    from omc3.optics_measurements import dpp

    original = dpp.calculate_dpoverp

    def known(input_files, meas_input):
        if len(input_files["X"]) != len(values):
            raise ValueError("OMC3 input count differs from the known momentum count")
        return values

    dpp.calculate_dpoverp = known
    try:
        yield
    finally:
        dpp.calculate_dpoverp = original


def _run_optics(
    *,
    lin_bases: list[Path],
    momenta: list[float],
    model_dir: Path,
    output_dir: Path,
    compensation: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with _known_momenta(momenta):
        hole_in_one_entrypoint(
            harpy=False,
            optics=True,
            files=lin_bases,
            outputdir=output_dir,
            compensation=compensation,
            three_bpm_method=True,
            accel="psbooster",
            model_dir=model_dir,
            ring=3,
        )


def run_fake_measurement(
    acquisitions: list[Acquisition],
    *,
    output_dir: Path,
) -> dict[str, dict[float, Path]]:
    """Harpy and optics per momentum, with dispersion from a combined run.

    Returns:
        ``{workflow: {pt: measurement_dir}}`` for the driven (``"acd"``) and
        free (``"all"``) workflows. Each directory holds phase measured at that
        momentum alone and dispersion measured across all of them.
    """
    lin_dir = output_dir / "lin_files"
    lin_dir.mkdir(parents=True)
    lin_bases: list[Path] = []

    for index, acquisition in enumerate(acquisitions):
        source = output_dir / f"fake_dp_{index}.sdds"
        tbt = _tbt_data(acquisition.data, source)
        hole_in_one_entrypoint(
            harpy=True,
            optics=False,
            files=[tbt],
            outputdir=lin_dir,
            tbt_datatype="tbt_data",
            unit="m",
            turns=[0, tbt.nturns],
            clean=False,
            keep_exact_zeros=True,
            peak_to_peak=1e-10,
            max_peak=0.02,
            tunes=[*acquisition.driven_tunes, 0.0],
            nattunes=[*acquisition.natural_tunes, 0.0],
            output_bits=10,
            turn_bits=14,
            tolerance=1e-3,
            tune_clean_limit=1e-3,
            to_write=["lin"],
        )
        lin_bases.append(lin_dir / source.name)

    momenta = [acquisition.pt for acquisition in acquisitions]
    # The combined run only has to produce dispersion, so it uses the momentum
    # closest to zero as its model, the way an on-momentum reference would.
    reference = min(acquisitions, key=lambda acquisition: abs(acquisition.pt))

    results: dict[str, dict[float, Path]] = {}
    for workflow, compensation in (("acd", "none"), ("all", "equation")):
        workflow_dir = output_dir / workflow
        combined_dir = workflow_dir / "combined"
        _run_optics(
            lin_bases=lin_bases,
            momenta=momenta,
            model_dir=reference.model_dir,
            output_dir=combined_dir,
            compensation=compensation,
        )

        per_momentum: dict[float, Path] = {}
        for index, acquisition in enumerate(acquisitions):
            momentum_dir = workflow_dir / f"pt_{index}"
            # One file, so its momentum is the group's reference: pin it to zero
            # rather than letting omc3 re-derive an offset from the model
            # dispersion. Only phase and beta are consumed from here; the
            # degenerate single-momentum dispersion is overwritten below.
            _run_optics(
                lin_bases=[lin_bases[index]],
                momenta=[0.0],
                model_dir=acquisition.model_dir,
                output_dir=momentum_dir,
                compensation=compensation,
            )
            for filename in DISPERSION_MEASUREMENT_FILES:
                source = combined_dir / filename
                if not source.exists():
                    raise FileNotFoundError(f"Combined HIO dispersion file is missing: {source}")
                shutil.copy2(source, momentum_dir / filename)
            per_momentum[acquisition.pt] = momentum_dir
        results[workflow] = per_momentum
    return results
