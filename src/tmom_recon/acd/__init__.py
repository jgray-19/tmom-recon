"""AC-dipole reconstruction helpers.

This subpackage groups the marker-side AC-dipole workflow:

- selecting BPMs around a driven element,
- reconstructing local BPM states,
- transporting those states to the AC-dipole marker with MAD-NG,
- fitting harmonic ``dpx``/``dpy`` kick waveforms at the marker, and
- transporting cleaned pre-/post-kick states back to the adjacent BPMs.

The most useful public entry points are:

- :class:`ACDipoleConfig` for higher-level integration paths,
- :func:`calculate_ac_dipole_momentum` for direct ACD reconstruction, and
- :class:`ACDipoleMadDriver` for model-backed state transport.
"""

from .integration import (
    ACDipoleConfig,
    apply_ac_dipole_bpm_overrides_inplace,
    apply_precomputed_ac_dipole_bpm_overrides_inplace,
    run_ac_dipole_reconstruction,
)
from .madng_driver import ACDipoleMadDriver, ACDipoleTrackingError
from .models import (
    ACDipoleBPMSelection,
    ACDipoleBPMWindow,
    ACDipoleHarmonicFit,
    ACDipoleStateEstimate,
    ACDipoleStateSeries,
)
from .reconstruction import calculate_ac_dipole_momentum
from .selection import select_ac_dipole_bpm_window, select_ac_dipole_bpms

__all__ = [
    "ACDipoleBPMSelection",
    "ACDipoleBPMWindow",
    "ACDipoleConfig",
    "ACDipoleHarmonicFit",
    "ACDipoleMadDriver",
    "ACDipoleStateEstimate",
    "ACDipoleStateSeries",
    "ACDipoleTrackingError",
    "apply_ac_dipole_bpm_overrides_inplace",
    "apply_precomputed_ac_dipole_bpm_overrides_inplace",
    "calculate_ac_dipole_momentum",
    "run_ac_dipole_reconstruction",
    "select_ac_dipole_bpm_window",
    "select_ac_dipole_bpms",
]
