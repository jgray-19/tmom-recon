"""MAD-NG TPSA-based 4D transport matrix computation for kicker reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pymadng import MAD


def get_transport_matrices_4d(
    sequence_file: Path,
    sequence_name: str,
    kicker: str,
    bpms: list[str],
    beam_energy: float = 6800.0,
) -> dict[str, np.ndarray]:
    """Compute first-order 4x4 (x,px,y,py) transport matrices from kicker to each BPM.

    Uses MAD-NG TPSA with order 1 in all six phase-space coordinates, then
    truncates the resulting 6x6 Jacobian to the transverse 4x4 block.

    Args:
        sequence_file: Path to the MAD-X sequence file (.seq).
        sequence_name: Name of the sequence inside the file (e.g. "lhcb1").
        kicker: Element name of the kicker (must match MAD-NG naming, e.g. "MKD.O5L6.B1").
        bpms: List of BPM element names to compute matrices to.
        beam_energy: Beam total energy in GeV.

    Returns:
        Dict mapping each BPM name to a (4, 4) numpy array, where the matrix M
        maps ``[x, px, y, py]`` at the kicker to ``[x, px, y, py]`` at the BPM.
    """
    sequence_file = Path(sequence_file)
    mad_cache = sequence_file.with_suffix(".mad")

    script = f"""
local damap, track in MAD
MADX:load("{sequence_file}", "{mad_cache}", {{rbarc=false}})
local seq = MADX.{sequence_name}
seq.beam = beam {{particle="proton", energy={beam_energy:.15e}}}

local bpms = py:recv()
for _, bpm in ipairs(bpms) do
    local x0 = damap {{nv=6, no={{1,1,1,1,1,1}}}}
    local trk, flw = track {{sequence=seq, range="{kicker}/"..bpm, X0=x0}}
    py:send(flw[1]:get1())
end
"""

    matrices: dict[str, np.ndarray] = {}
    with MAD() as mad:
        mad.send(script)
        mad.send(bpms)
        for bpm in bpms:
            r6 = np.array(mad.recv())  # (6, 6) Jacobian from TPSA get1()
            matrices[bpm] = r6[:4, :4]

    return matrices
