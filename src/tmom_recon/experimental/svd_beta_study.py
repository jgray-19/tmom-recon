"""Why does weighted-SVD cleaning help beta-from-amplitude but hurt beta-from-phase?

Background
----------
On real LHC fill data (see ``noise_investigation/svd_vs_harpy_clean.py``) a
*weighted* SVD pre-clean -- each BPM column whitened by its blank-acquisition
noise variance before the truncated SVD, then re-scaled -- gave
beta-from-*amplitude* as good as or better than plain harpy cleaning, but
beta-from-*phase* that was noticeably worse. This module reproduces and
diagnoses that split in a controlled AC-dipole simulation that uses the *same*
omc3 ``hole_in_one`` pipeline (harpy + optics) as the real measurement.

Pipeline (mirrors operations)
-----------------------------
1. ``prepare_model_dir`` builds a real omc3 LHC model directory via
   ``pymadng_utils`` (omc3 model creator + MAD-NG). This yields ``twiss.dat``,
   ``twiss_ac.dat`` (the AC-dipole driven model needed for compensation),
   ``twiss_elements.dat`` and -- crucially -- ``lhcb1_saved.seq``, the *same*
   lattice the model describes.
2. ``build_acd_scene`` loads that sequence, applies quadrupole errors with
   ``apply_magnet_perturbations`` (genuine beta-beating vs the model),
   re-matches tunes / corrects orbit, builds the xsuite line with
   ``initialise_env``, AC-dipole-tracks it through a ramp + flat-top, and keeps
   only the flat-top (post-ramp) turns. Truth beating is
   ``(beta_machine - beta_model) / beta_model`` from the twiss tables.
3. Heteroscedastic per-BPM Gaussian noise is added. (With *uniform* noise,
   weighted and plain SVD are identical -- the column scales cancel -- so a
   spread is what makes the comparison meaningful, and it mirrors real BPMs.)
4. Each branch is cleaned and pushed through ``hole_in_one``:

   - ``none``     : raw data, harpy ``clean=True`` (its full internal SVD).
   - ``svd``      : plain truncated-SVD pre-clean, then harpy with its internal
                    SVD denoising disabled (``sing_val`` huge,
                    ``svd_dominance_limit=1``), keeping only the cut-cleaning.
   - ``weighted`` : variance-weighted truncated-SVD pre-clean, same harpy
                    settings as ``svd``.

   This mirrors the real script: the only difference between ``svd`` and
   ``weighted`` is the per-column whitening.
5. omc3 optics (``compensation="equation"``) produces ``beta_phase`` and
   ``beta_amplitude`` against the model. We score ``DELTABET - truth_beat`` RMS
   per BPM, so a lower number is a more accurate beating measurement.

The deliverable is :func:`run_study`; :func:`summarise` prints the verdict.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tmom_recon.svd import svd_clean_measurements, weighted_svd_clean_measurements

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

PLANES = ("x", "y")
AC_DIPOLE_ELEMENT = "MKQA.6L4.B1"
SEQ_NAME = "lhcb1"
NAT_TUNES = (0.28, 0.31)
DRV_TUNES = (0.27, 0.322)
# harpy settings that switch off harpy's own internal SVD denoising, so a
# pre-clean branch is judged on the pre-clean alone (kept cut-cleaning aside).
_NO_INTERNAL_SVD = {"sing_val": 10_000, "svd_dominance_limit": 1.0}


# --------------------------------------------------------------------------- #
# Model directory (omc3 + MAD-NG, via pymadng_utils)
# --------------------------------------------------------------------------- #
def prepare_model_dir(
    workdir: Path,
    *,
    year: str = "2025",
    modifiers: str = "R2025aRP_A18cmC18cmA10mL200cm_Flat.madx",
    nat_tunes: tuple[float, float] = NAT_TUNES,
    drv_tunes: tuple[float, float] = DRV_TUNES,
    energy: float = 6800.0,
    fetch: str = "afs",
    path: str | None = None,
) -> Path:
    """Create (or reuse) an omc3 LHC beam-1 model directory.

    Returns the directory containing ``twiss.dat``, ``twiss_ac.dat``,
    ``twiss_elements.dat`` and ``lhcb1_saved.seq``.
    """
    model_dir = workdir / f"model_b1__t{nat_tunes[0]}_{nat_tunes[1]}"
    if (model_dir / "twiss.dat").is_file() and (model_dir / f"{SEQ_NAME}_saved.seq").is_file():
        logger.info("Reusing existing model dir %s", model_dir)
        return model_dir

    from pymadng_utils.model_creator.create_models import create_lhc_model

    create_lhc_model(
        beam=1,
        output_dir=model_dir,
        year=year,
        modifiers=modifiers,
        fetch=fetch,
        path=path,
        nat_tunes=list(nat_tunes),
        drv_tunes=list(drv_tunes),
        energy=energy,
    )
    return model_dir


# --------------------------------------------------------------------------- #
# Scene: perturbed machine + tracked flat-top + truth beating
# --------------------------------------------------------------------------- #
@dataclass
class ACDScene:
    """Driven flat-top data plus the model to recover.

    Attributes:
        flat_tbt: Noiseless flat-top frame (``name``, ``turn``, ``x``, ``y``).
        bpms: Name-indexed model optics with machine truth betas and truth
            beating columns, sorted by ``s``.
        model_dir: omc3 model directory (passed to harpy + optics).
        workdir: Scratch directory for SDDS / harpy / optics outputs.
    """

    flat_tbt: pd.DataFrame
    bpms: pd.DataFrame
    model_dir: Path
    workdir: Path


def _model_bpm_frame(model_dir: Path) -> pd.DataFrame:
    """Read the model BPM optics (betx/bety/mux/muy) from ``twiss.dat``."""
    import tfs

    model = tfs.read(model_dir / "twiss.dat", index="NAME")
    cols = {c.upper(): c for c in model.columns}
    frame = pd.DataFrame(
        {
            "s": model[cols["S"]],
            "betx": model[cols["BETX"]],
            "bety": model[cols["BETY"]],
        },
        index=model.index.str.upper(),
    )
    return frame[frame.index.str.contains("BPM")].sort_values("s")


def _machine_bpm_betas(tws, bpm_index: pd.Index) -> pd.DataFrame:
    """Pull machine betas at the model BPMs from an xsuite twiss table."""
    names = np.array([str(n).upper() for n in tws.name])
    df = pd.DataFrame(
        {"betx": np.asarray(tws.betx), "bety": np.asarray(tws.bety)}, index=names
    ).loc[~pd.Index(names).duplicated()]
    return df.reindex(bpm_index)


def build_acd_scene(
    model_dir: Path,
    *,
    rel_k1_std: float = 5e-4,
    perturb_seed: int = 1,
    ramp_turns: int = 2000,
    flattop_turns: int = 2000,
    workdir: Path | None = None,
) -> ACDScene:
    """Perturb the machine, AC-dipole-track it, and assemble the truth beating."""
    from typing import Any, cast

    import tfs
    from pymadng_utils.accelerators import LHC
    from pymadng_utils.mad.accelerator_mad_interface import AcceleratorMadInterface
    from xtrack_tools.acd import run_ac_dipole_tracking
    from xtrack_tools.env import initialise_env
    from xtrack_tools.monitors import process_tracking_data

    workdir = workdir or Path(tempfile.mkdtemp(prefix="svd_beta_study_"))
    workdir.mkdir(parents=True, exist_ok=True)
    seq_file = model_dir / f"{SEQ_NAME}_saved.seq"

    mad = cast(
        "Any", AcceleratorMadInterface(LHC(beam=1, sequence_file=seq_file, kinetic_energy=6800))
    )
    try:
        magnet_strengths = mad.apply_magnet_perturbations(
            rel_error=rel_k1_std, seed=perturb_seed, magnet_type="q"
        )
        if isinstance(magnet_strengths, tuple):
            magnet_strengths = magnet_strengths[0]

        corrector_file = workdir / "correctors.tfs"
        matched_tunes = mad.perform_orbit_correction(
            machine_deltap=0.0,
            target_qx=NAT_TUNES[0],
            target_qy=NAT_TUNES[1],
            corrector_file=corrector_file,
        )
    finally:
        mad.close()

    corrector_table = cast("pd.DataFrame", tfs.read(corrector_file))
    corrector_table = corrector_table.loc[
        ~corrector_table["kind"].astype(str).str.lower().isin({"monitor", "hmonitor", "vmonitor"})
    ]

    env = initialise_env(
        matched_tunes,
        magnet_strengths,
        corrector_table,
        sequence_file=seq_file,
        seq_name=SEQ_NAME,
    )
    baseline_line = env[SEQ_NAME].copy()
    xsuite_tws = baseline_line.twiss(method="4d")

    monitored_line = run_ac_dipole_tracking(
        line=baseline_line,
        acd_marker=AC_DIPOLE_ELEMENT,
        sequence_name=SEQ_NAME,
        tws=xsuite_tws,
        deltap=0.0,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=list(DRV_TUNES),
        bpm_pattern=r"(?i)bpm.*",
        horizontal_excitation=2 * 0.042 / 180.0**0.5,
        vertical_excitation=2 * 0.042 / 177.0**0.5,
    )
    tracking_df = process_tracking_data(
        monitored_line,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        add_variance_columns=False,
    )
    flat_tbt = tracking_df.loc[:, ["name", "turn", "x", "y"]].copy()
    flat_tbt["name"] = flat_tbt["name"].str.upper()

    bpms = _model_bpm_frame(model_dir)
    machine = _machine_bpm_betas(xsuite_tws, bpms.index)
    bpms["betx_true"] = machine["betx"]
    bpms["bety_true"] = machine["bety"]
    bpms["beatx_true"] = (bpms["betx_true"] - bpms["betx"]) / bpms["betx"]
    bpms["beaty_true"] = (bpms["bety_true"] - bpms["bety"]) / bpms["bety"]
    bpms = bpms.dropna(subset=["betx_true", "bety_true"])

    logger.info(
        "ACD scene: %d BPMs, truth beating rms x=%.4f y=%.4f",
        len(bpms),
        float(np.sqrt(np.mean(bpms["beatx_true"] ** 2))),
        float(np.sqrt(np.mean(bpms["beaty_true"] ** 2))),
    )
    return ACDScene(flat_tbt=flat_tbt, bpms=bpms, model_dir=model_dir, workdir=workdir)


# --------------------------------------------------------------------------- #
# Noise + cleaning
# --------------------------------------------------------------------------- #
def add_noise(
    tbt: pd.DataFrame, sigma: float, seed: int, sigma_spread: float = 2.0
) -> pd.DataFrame:
    """Add heteroscedastic zero-mean Gaussian noise and variance columns.

    Each BPM's sigma is drawn log-uniformly in ``[sigma/(1+spread), sigma*(1+spread)]``.
    ``sigma_spread=0`` gives uniform noise, for which weighted and plain SVD
    coincide exactly.
    """
    rng = np.random.default_rng(seed)
    out = tbt.copy()
    names = out["name"].to_numpy()
    unique = np.unique(names)
    if sigma_spread > 0.0:
        logsig = rng.uniform(
            np.log(sigma / (1.0 + sigma_spread)),
            np.log(sigma * (1.0 + sigma_spread)),
            size=len(unique),
        )
        per_bpm = dict(zip(unique, np.exp(logsig), strict=True))
    else:
        per_bpm = dict.fromkeys(unique, sigma)
    sig = np.array([per_bpm[n] for n in names])
    out["x"] = out["x"].to_numpy() + rng.normal(0.0, sig)
    out["y"] = out["y"].to_numpy() + rng.normal(0.0, sig)
    out["var_x"] = sig**2
    out["var_y"] = sig**2
    return out


def clean(tbt: pd.DataFrame, method: str, rank: int | str = "auto") -> pd.DataFrame:
    """Dispatch to a cleaning method: ``none``, ``svd`` or ``weighted``."""
    if method == "none":
        return tbt
    if method == "svd":
        return svd_clean_measurements(tbt, rank=rank)
    if method == "weighted":
        return weighted_svd_clean_measurements(tbt, rank=rank)
    raise ValueError(f"unknown cleaning method: {method!r}")


# --------------------------------------------------------------------------- #
# hole_in_one: harpy + optics
# --------------------------------------------------------------------------- #
def _write_sdds(tbt: pd.DataFrame, bpm_order: Sequence[str], path: Path) -> Path:
    """Write a long-format frame to an LHC-style SDDS for harpy."""
    from turn_by_turn import write_tbt
    from turn_by_turn.structures import TbtData, TransverseData

    turns = np.sort(tbt["turn"].unique())
    x = tbt.pivot(index="name", columns="turn", values="x").reindex(
        index=list(bpm_order), columns=turns
    )
    y = tbt.pivot(index="name", columns="turn", values="y").reindex(
        index=list(bpm_order), columns=turns
    )
    write_tbt(
        path,
        TbtData(matrices=[TransverseData(X=x, Y=y)], bunch_ids=[0], nturns=len(turns)),
        datatype="lhc",
    )
    return path


def run_hole_in_one(
    scene: ACDScene, tbt: pd.DataFrame, method: str, tag: str
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run harpy + optics for one branch and return the beta tfs tables.

    Returns ``{plane: {"phase": df, "amplitude": df}}`` with ``NAME``-indexed
    optics frames (the omc3 ``beta_phase_<plane>.tfs`` / ``beta_amplitude``).
    """
    import tfs
    from omc3.hole_in_one import hole_in_one_entrypoint

    bpm_order = list(scene.bpms.index)
    sdds_dir = scene.workdir / "sdds" / tag
    sdds_dir.mkdir(parents=True, exist_ok=True)
    sdds = _write_sdds(tbt, bpm_order, sdds_dir / "sim.sdds")

    harpy_dir = scene.workdir / "harpy" / tag
    harpy_extra = {} if method == "none" else _NO_INTERNAL_SVD
    hole_in_one_entrypoint(
        harpy=True,
        clean=True,
        files=[sdds],
        outputdir=harpy_dir,
        model=scene.model_dir / "twiss.dat",
        tunes=[*DRV_TUNES, 0.0],
        nattunes=[*NAT_TUNES, 0.0],
        unit="m",
        to_write=["lin"],
        tbt_datatype="lhc",
        **harpy_extra,
    )

    optics_dir = scene.workdir / "optics" / tag
    lin_base = next(harpy_dir.glob("*.linx")).with_suffix("")  # strip .linx -> *.sdds
    hole_in_one_entrypoint(
        optics=True,
        accel="lhc",
        beam=1,
        year="2025",
        model_dir=scene.model_dir,
        compensation="equation",
        files=[lin_base],
        outputdir=optics_dir,
    )

    out: dict[str, dict[str, pd.DataFrame]] = {}
    for plane in PLANES:
        out[plane] = {
            kind: tfs.read(optics_dir / f"beta_{kind}_{plane}.tfs", index="NAME")
            for kind in ("phase", "amplitude")
        }
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def analyse(scene: ACDScene, betas: dict[str, dict[str, pd.DataFrame]]) -> dict[str, float]:
    """Score one branch's DELTABET against the truth beating, per plane/method."""
    bpms = scene.bpms
    result: dict[str, float] = {}
    for plane in PLANES:
        truth = bpms[f"beat{plane}_true"]
        for kind in ("phase", "amplitude"):
            df = betas[plane][kind]
            col = f"DELTABET{plane.upper()}"
            common = df.index.intersection(truth.index)
            residual = df.loc[common, col].to_numpy() - truth.loc[common].to_numpy()
            result[f"{kind}_err_{plane}"] = _rms(residual)
            result[f"{kind}_rms_{plane}"] = _rms(df.loc[common, col].to_numpy())
            result[f"n_{kind}_{plane}"] = int(len(common))
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class StudyConfig:
    """Inputs for :func:`run_study`."""

    workdir: Path
    sigma: float = 3e-4
    sigma_spread: float = 2.0
    rel_k1_std: float = 5e-4
    perturb_seed: int = 1
    ramp_turns: int = 2000
    flattop_turns: int = 2000
    rank: int | str = "auto"
    methods: tuple[str, ...] = ("none", "svd", "weighted")
    noise_seeds: tuple[int, ...] = (0, 1, 2)
    scene: ACDScene | None = field(default=None, repr=False)


def run_study(config: StudyConfig) -> pd.DataFrame:
    """Run the comparison; one row per (method, noise seed), scored per plane."""
    if config.scene is not None:
        scene = config.scene
    else:
        model_dir = prepare_model_dir(config.workdir)
        scene = build_acd_scene(
            model_dir,
            rel_k1_std=config.rel_k1_std,
            perturb_seed=config.perturb_seed,
            ramp_turns=config.ramp_turns,
            flattop_turns=config.flattop_turns,
            workdir=config.workdir,
        )

    rows = []
    for seed in config.noise_seeds:
        noisy = add_noise(scene.flat_tbt, config.sigma, seed, config.sigma_spread)
        for method in config.methods:
            cleaned = clean(noisy, method, rank=config.rank)
            betas = run_hole_in_one(scene, cleaned, method, tag=f"{method}_s{seed}")
            rows.append({"method": method, "seed": seed, **analyse(scene, betas)})
            logger.info("seed=%d method=%s done", seed, method)
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate :func:`run_study` rows into per-method mean errors."""
    err_cols = [c for c in results.columns if c.endswith(("_err_x", "_err_y"))]
    return results.groupby("method")[err_cols].mean()
