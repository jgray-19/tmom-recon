"""Regenerate deterministic external-strength fixtures.

This development helper is not imported by the tests or by :mod:`tmom_recon`.
Run it from the repository root with the CERN ``accpy`` interpreter.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pymadng_utils.accelerators import LHC
from pymadng_utils.mad import AcceleratorMadInterface
from xtrack_tools.env import create_xsuite_environment
from xtrack_tools.errors import apply_relative_bend_field_errors

from tests.psb_tracking import (
    KINETIC_ENERGY_GEV,
    MAIN_BEND_PREFIX,
    QUAD_PREFIX,
    SEQ_NAME,
    _apply_relative_quad_gradient_errors,
    create_psb_model_dir,
)
from tests.support.external_strengths import SCHEMA_VERSION, strength_fingerprint

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent
GENERATED_UTC = "2026-08-20T00:00:00Z"


def _git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(machine: str, metadata: dict, strengths: dict[str, float]) -> None:
    parsed = {str(name): float(value) for name, value in sorted(strengths.items())}
    metadata["strength_fingerprint_sha256"] = strength_fingerprint(parsed)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "machine": machine,
        "metadata": metadata,
        "strengths": parsed,
    }
    destination = OUTPUT_DIR / f"{machine}.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _lhc(machine: str) -> None:
    sequence = ROOT / "tests" / "data" / "sequences" / f"{machine}.seq"
    accelerator = LHC(beam=1, sequence_file=sequence, kinetic_energy=6800)
    interface = AcceleratorMadInterface(accelerator)
    try:
        strengths, _ = interface.apply_magnet_perturbations(
            rel_error=1e-4, seed=12, magnet_type="all"
        )
    finally:
        interface.close()
    _write(
        machine,
        {
            "generated_utc": GENERATED_UTC,
            "generator": "pymadng-utils AcceleratorMadInterface.apply_magnet_perturbations",
            "pymadng_utils_revision": _git_revision(
                Path(__import__("pymadng_utils").__file__).resolve().parents[2]
            ),
            "sequence": {
                "path": str(sequence.relative_to(ROOT)),
                "sha256": _sha256(sequence),
            },
            "perturbations": [{"family": "all", "relative_rms": 1e-4, "seed": 12}],
            "strength_semantics": "integrated MAD-NG delta knobs",
        },
        strengths,
    )


def _psb() -> None:
    source = ROOT / "tests" / "data" / "acc-models-psb"
    model_dir = create_psb_model_dir(source)
    sequence = model_dir / "psb3_saved.seq"
    env = create_xsuite_environment(
        sequence_file=sequence,
        kinetic_energy=KINETIC_ENERGY_GEV,
        seq_name=SEQ_NAME,
        json_file=model_dir / "psb3_saved.json",
    )
    line = env[SEQ_NAME].copy()
    bends = apply_relative_bend_field_errors(line, rms=8e-4, seed=7, name_prefix=MAIN_BEND_PREFIX)
    quads = _apply_relative_quad_gradient_errors(line, rms=1e-3, seed=11, name_prefix=QUAD_PREFIX)
    strengths = {f"{name.upper()}.k0": value for name, value in bends.items()}
    strengths.update({f"{name.upper()}.k1": value for name, value in quads.items()})
    source_files = sorted(path for path in source.rglob("*") if path.is_file())
    source_digest = hashlib.sha256()
    for path in source_files:
        source_digest.update(str(path.relative_to(source)).encode())
        source_digest.update(path.read_bytes())
    _write(
        "psb",
        {
            "generated_utc": GENERATED_UTC,
            "generator": "xtrack-tools seeded PSB perturbation helpers",
            "pymadng_utils_revision": _git_revision(
                Path(__import__("pymadng_utils").__file__).resolve().parents[2]
            ),
            "sequence": {
                "path": "tests/data/acc-models-psb",
                "sha256": source_digest.hexdigest(),
            },
            "perturbations": [
                {"family": "bend", "relative_rms": 8e-4, "seed": 7},
                {"family": "quadrupole", "relative_rms": 1e-3, "seed": 11},
            ],
            "strength_semantics": "absolute MAD-NG k0/k1 values",
        },
        strengths,
    )


if __name__ == "__main__":
    # Keep the timestamp stable; fixture identity is content/provenance based.
    assert datetime.fromisoformat(GENERATED_UTC.replace("Z", "+00:00")).tzinfo is UTC
    _psb()
    _lhc("lhcb1")
    _lhc("b1_120cm_crossing")
