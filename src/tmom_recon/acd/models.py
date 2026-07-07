"""Dataclasses shared across the AC-dipole reconstruction pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ACDipoleBPMSelection:
    """A single upstream/downstream BPM pair bracketing the AC dipole.

    Attributes:
        upstream: Name of the upstream BPM.
        downstream: Name of the downstream BPM.
    """

    upstream: str
    downstream: str


@dataclass(frozen=True)
class ACDipoleBPMWindow:
    """An ordered window of BPMs on each side of the AC dipole.

    Attributes:
        upstream: Ordered tuple of upstream BPM names (closest first).
        downstream: Ordered tuple of downstream BPM names (closest first).
    """

    upstream: tuple[str, ...]
    downstream: tuple[str, ...]

    @property
    def primary(self) -> ACDipoleBPMSelection:
        """The closest upstream/downstream BPM pair."""
        return ACDipoleBPMSelection(
            upstream=self.upstream[0],
            downstream=self.downstream[0],
        )


@dataclass(frozen=True)
class ACDipoleSide:
    """Reconstructed BPM momenta on one side of the AC-dipole marker.

    Bundles the per-BPM reconstructed-momentum frames for the window of BPMs on
    a single side of the marker together with the orientation conventions that
    tell the pipeline how to transport those states to the marker
    (:attr:`direction`) and how to label the resulting output. Representing the
    two sides as two of these lets the reconstruction treat upstream and
    downstream symmetrically instead of threading mirror-image arguments.

    The window ordering (closest BPM to the marker first) and the primary BPM are
    read straight off :attr:`bpm_frames`, so the window has a single source.

    Attributes:
        label: Side identity, ``"upstream"`` or ``"downstream"``. Appears in the
            per-turn summary column names (e.g. ``px_bpm_upstream``).
        marker_end: The marker end this side reconstructs: ``"before"`` (pre-kick,
            upstream) or ``"after"`` (post-kick, downstream).
        direction: MAD-NG tracking direction from a BPM *to* the marker: ``+1``
            forward (upstream) or ``-1`` backward (downstream). The reverse
            transport, marker to BPM, uses ``-direction``.
        bpm_frames: Ordered mapping from BPM name to its reconstructed-momentum
            DataFrame, closest BPM to the marker first.
    """

    label: str
    marker_end: str
    direction: int
    bpm_frames: dict[str, pd.DataFrame]

    @property
    def bpm_names(self) -> tuple[str, ...]:
        """Window BPM names, closest to the marker first."""
        return tuple(self.bpm_frames)

    @property
    def primary(self) -> str:
        """Name of the primary (closest) BPM."""
        return next(iter(self.bpm_frames))

    @property
    def primary_frame(self) -> pd.DataFrame:
        """Reconstructed-momentum DataFrame for the primary BPM."""
        return self.bpm_frames[self.primary]

    @property
    def bpm_row_type(self) -> str:
        """``row_type`` value tagging this side's BPM output rows."""
        return f"{self.label}_bpm"

    @property
    def marker_row_type(self) -> str:
        """``row_type`` value tagging this side's marker output rows."""
        return f"marker_{self.marker_end}"


@dataclass(frozen=True)
class ACDipoleHarmonicFit:
    """Result of a single-harmonic least-squares fit at the AC-dipole marker.

    The fitted waveform is ``amplitude * sin(2π * tune * n + phase) + offset``
    where ``n`` is the turn number.

    Attributes:
        tune: Fractional tune used for the fit (0 < tune < 0.5).
        amplitude: Peak amplitude of the fitted harmonic [rad].
        phase: Phase of the fitted harmonic [rad].
        offset: DC offset of the fitted harmonic [rad].
        fitted: Per-turn fitted values, shape ``(n_turns,)`` [rad].
    """

    tune: float
    amplitude: float
    phase: float
    offset: float
    fitted: np.ndarray


@dataclass(frozen=True)
class ACDipoleStateSeries:
    """Turn-by-turn phase-space state at a single lattice location.

    Attributes:
        x: Horizontal position per turn [m].
        px: Horizontal normalised momentum per turn [rad].
        y: Vertical position per turn [m].
        py: Vertical normalised momentum per turn [rad].
        t: Longitudinal time coordinate per turn [s], or ``None``.
        pt: Longitudinal momentum deviation per turn, or ``None``.
    """

    x: np.ndarray
    px: np.ndarray
    y: np.ndarray
    py: np.ndarray
    t: np.ndarray
    pt: np.ndarray


@dataclass(frozen=True)
class ACDipoleStateEstimate:
    """Phase-space state estimate with per-coordinate variances.

    Attributes:
        state: Central-value turn-by-turn state.
        var_x: Variance of ``state.x`` per turn.
        var_px: Variance of ``state.px`` per turn.
        var_y: Variance of ``state.y`` per turn.
        var_py: Variance of ``state.py`` per turn.
    """

    state: ACDipoleStateSeries
    var_x: np.ndarray
    var_px: np.ndarray
    var_y: np.ndarray
    var_py: np.ndarray


@dataclass(frozen=True)
class ACDipoleCleaningResult:
    """Output of the single-pass AC-dipole state cleaning step.

    Attributes:
        upstream: Cleaned pre-kick state estimate at the marker.
        downstream: Cleaned post-kick state estimate at the marker.
        dpx_fit: Harmonic fit of the horizontal kick waveform.
        dpy_fit: Harmonic fit of the vertical kick waveform.
    """

    upstream: ACDipoleStateEstimate
    downstream: ACDipoleStateEstimate
    dpx_fit: ACDipoleHarmonicFit
    dpy_fit: ACDipoleHarmonicFit


@dataclass(frozen=True)
class ACDipoleFitResult:
    """Aggregate output of the full AC-dipole fit and cleaning pipeline.

    Attributes:
        summary: Per-turn summary DataFrame with BPM momenta and raw kick columns.
        turns: Turn numbers array, shape ``(n_turns,)``.
        raw_upstream: Raw (un-cleaned) upstream state estimate at the marker.
        raw_downstream: Raw (un-cleaned) downstream state estimate at the marker.
        cleaned_upstream: Cleaned pre-kick state estimate at the marker.
        cleaned_downstream: Cleaned post-kick state estimate at the marker.
        dpx_raw: Raw horizontal kick per turn (downstream.px - upstream.px) [rad].
        dpy_raw: Raw vertical kick per turn [rad].
        dpx_fit: Harmonic fit of the horizontal kick waveform.
        dpy_fit: Harmonic fit of the vertical kick waveform.
        dpx_r2: R² of the horizontal harmonic fit.
        dpy_r2: R² of the vertical harmonic fit.
    """

    summary: pd.DataFrame
    turns: np.ndarray
    raw_upstream: ACDipoleStateEstimate
    raw_downstream: ACDipoleStateEstimate
    cleaned_upstream: ACDipoleStateEstimate
    cleaned_downstream: ACDipoleStateEstimate
    dpx_raw: np.ndarray
    dpy_raw: np.ndarray
    dpx_fit: ACDipoleHarmonicFit
    dpy_fit: ACDipoleHarmonicFit
    dpx_r2: float
    dpy_r2: float
