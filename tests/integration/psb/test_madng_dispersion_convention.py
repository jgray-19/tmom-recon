"""MAD-NG second-order dispersion convention contract for PSB."""

from __future__ import annotations

import numpy as np
import pytest
from pymadng_utils.accelerators import PSB

from tests.psb_tracking import ACD_ELEMENT, KINETIC_ENERGY_GEV, RING, SEQ_FILE
from tmom_recon.acd.madng_driver import ACDipoleMadDriver

DELTA = 3.0e-3


@pytest.mark.psb
@pytest.mark.integration
@pytest.mark.slow
def test_madng_second_order_dispersion_is_per_pt_with_half_folded_in(psb_model_dir) -> None:
    seq = psb_model_dir / SEQ_FILE
    accelerator = PSB(sequence_file=seq, ring=RING, kinetic_energy=KINETIC_ENERGY_GEV)
    model = ACDipoleMadDriver(accelerator=accelerator, pt=0.0, observed_elements=ACD_ELEMENT)

    tw0 = model.run_twiss(observe=0, chrom=True)
    for column in ("dx", "dpx", "ddx", "ddpx"):
        assert column in tw0.columns, f"chrom=true did not produce {column}"

    bpms = [name for name in tw0.index if "BPM" in str(name).upper()]
    assert len(bpms) > 4

    pt = accelerator.dp2pt(DELTA)
    tw_pt = model.run_twiss(observe=0, pt=pt)

    for coord, first, second in (("x", "dx", "ddx"), ("px", "dpx", "ddpx")):
        ref = tw0.loc[bpms, coord].to_numpy(float)
        exact = tw_pt.loc[bpms, coord].to_numpy(float)
        d1 = tw0.loc[bpms, first].to_numpy(float)
        d2 = tw0.loc[bpms, second].to_numpy(float)

        design = np.column_stack([pt * d1, pt**2 * d2])
        c1, c2 = np.linalg.lstsq(design, exact - ref, rcond=None)[0]

        assert c1 == pytest.approx(1.0, abs=1e-3), f"{coord}: dispersion is not per unit pt"
        assert c2 == pytest.approx(1.0, abs=0.02), f"{coord}: unexpected ddx normalisation"

        res_1st = np.abs(ref + pt * d1 - exact).max()
        res_2nd = np.abs(ref + pt * d1 + pt**2 * d2 - exact).max()
        assert res_2nd < res_1st / 10.0, (
            f"{coord}: second order ({res_2nd:.3e}) did not improve on first order ({res_1st:.3e})"
        )
