"""The kick uncertainty is the BPM position noise propagated through the real reconstruction.

Reference: central finite differences of the full ``calculate_acd_pz`` raw kick
(``p_after - p_before`` from ``attrs["raw_marker_states"]``) with respect to the
measured positions. Perturbing every third turn of one BPM/plane at once separates the
turn offsets -1, 0, +1 of the neighbour-turn dependence, so three runs per BPM/plane
and sign give the whole Jacobian. Real MAD-NG transport, no mocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.reference_co import measured_zero_reference_for_simulation
from tmom_recon import ACDipoleConfig, ModelDetails, calculate_acd_pz

from .acd_test_helpers import AC_DIPOLE_ELEMENT, _ac_dipole_segment_around_element, _get_driver

SEQ_FILE = "lhcb1.seq"
DRIVEN_TUNES = (0.27, 0.322)
POSITION_SIGMA = 3e-5
FD_STEP = 1e-5
# Keys below this fraction of the largest coefficient are not finite-differenced.
KEY_CUT = 1e-2
# Central differences of the same smooth map: only rounding and O(h^2) remain.
RTOL = 1e-4
ATOL_FRACTION = 1e-6
EDGE_TURNS = 3

pytestmark = [pytest.mark.lhc, pytest.mark.integration, pytest.mark.slow]


def _run(data, model_details, config):
    return calculate_acd_pz(
        data,
        model_details,
        config,
        closed_orbit_at_zero=measured_zero_reference_for_simulation(data),
        orbit_mode="dynamic",
    )


def _raw_kick(result, plane: str) -> np.ndarray:
    raw = result.attrs["raw_marker_states"].sort_values("turn")
    after = raw[raw["row_type"] == "marker_after"][f"p{plane}"].to_numpy()
    before = raw[raw["row_type"] == "marker_before"][f"p{plane}"].to_numpy()
    return after - before


@pytest.fixture(scope="module")
def reconstruction(data_dir, acd_tracking_setup):
    sequence_file = data_dir / "sequences" / SEQ_FILE
    data = acd_tracking_setup(sequence_file, delta_p=0.0, flattop_turns=100).data.copy()
    data["var_x"] = POSITION_SIGMA**2
    data["var_y"] = POSITION_SIGMA**2
    driver = _get_driver(sequence_file)
    upstream, downstream = _ac_dipole_segment_around_element(
        driver.twiss_elements,
        available_bpms=data["name"].unique().tolist(),
        element_name=AC_DIPOLE_ELEMENT,
    )
    config = ACDipoleConfig(
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        driven_tunes=DRIVEN_TUNES,
        bpm_upstream=upstream,
        bpm_downstream=downstream,
    )
    details = ModelDetails(accelerator=driver.accelerator, pt=driver.pt)
    return data, details, config, _run(data, details, config)


@pytest.mark.parametrize("plane", ["x", "y"])
def test_kick_jacobian_matches_finite_differences(reconstruction, plane) -> None:
    data, details, config, result = reconstruction
    jacobian, _ = result.attrs["kick_jacobian"][plane]
    turns = result.attrs["summary"]["turn"].to_numpy(int)
    scale = max(np.abs(c[EDGE_TURNS:-EDGE_TURNS]).max() for c in jacobian.values())
    keys = [k for k, c in jacobian.items() if np.abs(c).max() > KEY_CUT * scale]
    assert keys, "no significant Jacobian entries"

    groups = sorted({(bpm, p) for bpm, p, _ in keys})
    for bpm, p in groups:
        for residue in range(3):
            raw = []
            for sign in (+1, -1):
                shifted = data.copy()
                rows = (shifted["name"] == bpm) & (shifted["turn"] % 3 == residue)
                shifted.loc[rows, p] += sign * FD_STEP
                raw.append(_raw_kick(_run(shifted, details, config), plane))
            derivative = (raw[0] - raw[1]) / (2 * FD_STEP)
            for key in (k for k in keys if k[:2] == (bpm, p)):
                offset = key[2]
                picked = (turns + offset) % 3 == residue
                picked[:EDGE_TURNS] = picked[-EDGE_TURNS:] = False
                np.testing.assert_allclose(
                    jacobian[key][picked],
                    derivative[picked],
                    rtol=RTOL,
                    atol=ATOL_FRACTION * scale,
                    err_msg=f"{plane} kick vs {key}",
                )


@pytest.mark.parametrize("plane", ["x", "y"])
def test_kick_std_is_the_propagated_position_noise(reconstruction, plane) -> None:
    _, _, _, result = reconstruction
    jacobian, variances = result.attrs["kick_jacobian"][plane]
    std = result.attrs["summary"][f"dp{plane}_err_rad"].to_numpy()
    expected = np.sqrt(sum(c**2 * POSITION_SIGMA**2 for c in jacobian.values()))
    assert variances and set(variances.values()) == {POSITION_SIGMA**2}
    np.testing.assert_allclose(std, expected, rtol=1e-12)
    assert np.isfinite(std).all()
    assert (std > 0).all()
