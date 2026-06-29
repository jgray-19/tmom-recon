"""Run every limitations study in order.

Each ``study/0X_*.py`` is runnable standalone; this just executes them in
sequence (their module names start with a digit, so we run them by path rather
than importing). Run from the repo root::

    PYTHONPATH=. python study/run_all.py

Figures land in ``study/plots/`` and result CSVs in ``study/results/``.
"""

from __future__ import annotations

import runpy
import time
from pathlib import Path

STUDIES = [
    "01_fodo_nonlinearities.py",
    "02_bend_radius.py",
    "03_realistic_lattices.py",
    "04_bpm_noise.py",
    "05_magnetic_errors.py",
    "06_model_optics_distortion.py",
    "07_phase_advance.py",
    "08_weighted_svd.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for name in STUDIES:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        t0 = time.time()
        runpy.run_path(str(here / name), run_name="__main__")
        print(f"-- {name} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
