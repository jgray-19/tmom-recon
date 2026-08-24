"""Unified momentum-reconstruction entry point.

:func:`calculate_pz` is the single public API: it resolves the optics sources
(model twiss and/or omc3 measurement directory, per category), runs the
neighbour-pair reconstruction, and optionally refines the BPMs around an AC
dipole with the ACD kick reconstruction.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from tmom_recon.acd.integration import (
    ACDipoleConfig,
    ResolvedACDipoleConfig,
    apply_precomputed_ac_dipole_bpm_overrides,
    resolve_ac_dipole_config,
)
from tmom_recon.acd.reconstruction import (
    calculate_ac_dipole_momentum,
    prepare_ac_dipole_inputs,
    reconstruct_from_prepared,
)
from tmom_recon.frame import ReconstructionFrame
from tmom_recon.lattice.core import validate_input
from tmom_recon.model import (
    ModelDetails,
    ResolvedModel,
    resolve_model_details,
)
from tmom_recon.optics import (
    LoadedMeasurement,
    ModelOpticsErrors,
    OpticsCategory,
    load_measurement,
    resolve_optics,
)
from tmom_recon.physics.pt_calculation import _estimate_pt_from_prepared
from tmom_recon.physics.transverse import reconstruct_momenta

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Collection, Mapping
    from pathlib import Path

    import pandas as pd
    import tfs

    from tmom_recon.acd.madng_driver import ACDipoleMadDriver
    from tmom_recon.acd.reconstruction import PreparedACDInputs


LOGGER = logging.getLogger(__name__)

__all__ = [
    "ACDipoleConfig",
    "ACDipolePzGenerator",
    "ModelDetails",
    "ModelOpticsErrors",
    "ReconstructionFrame",
    "PzGenerator",
    "calculate_pz",
]


def calculate_pz(
    data: pd.DataFrame,
    model_details: ModelDetails,
    *,
    frame: ReconstructionFrame,
    measurement_dir: str | Path | None = None,
    model_optics: Collection[OpticsCategory] = (),
    use_dispersion: bool = True,
    model_errors: ModelOpticsErrors | None = None,
    reverse_meas_tws: bool = False,
    measurement_pt_offset: float | None = None,
    info: bool = True,
    acd: ACDipoleConfig | None = None,
    acd_only: bool = False,
    generator: bool = False,
    barrier_s: float | None,
) -> pd.DataFrame | PzGenerator | ACDipolePzGenerator:
    """Reconstruct transverse momenta at every BPM from turn-by-turn data.

    Two conventions this entry point is strict about, because both have been got
    wrong in practice:

    * **Momentum is relative.** ``measurement_pt_offset`` is measured from the
      orbit-zero frame. When omitted it is estimated after that frame is applied.
    * **Phase.** Any phase in the returned/consumed intermediate frames named
      ``delta`` is the deviation of a neighbour advance from a quarter turn
      (:math:`\\phi_{\\mathrm{code}} = \\phi_x - \\pi/2`, in turns), *not* the
      phase advance and *not* the twiss ``mu1``/``mu2``. See
      :mod:`tmom_recon.physics.bpm_phases`.

    The model optics are always generated from *model_details*. The user never
    provides a twiss to this entry point. When *measurement_dir* is supplied,
    measured optics can override selected categories, but the model-side optics
    and closed-orbit references are still generated here.

    Args:
        data: Turn-by-turn BPM data with columns ``name, turn, x, y`` and
            position variances ``var_x, var_y``.
        model_details: Accelerator, tunes, momentum and optional strengths used
            to generate the MAD-NG model optics.
        frame: Measured orbit-zero positions, dynamic planes, and fitted momenta
            for retained planes. Required on every reconstruction path.
        measurement_dir: omc3 optics measurement directory.
        model_optics: Optics categories forced to come from the model.
        use_dispersion: If ``False``, run a pure transverse reconstruction
            (no dispersion terms, pt taken as 0 unless overridden).
        model_errors: Rough uncertainties for model-sourced optics.
        reverse_meas_tws: Reverse BPM ordering in the measured phase chain
            (Beam 2 / reverse-direction data).
        measurement_pt_offset: MAD-NG ``pt`` relative to orbit zero. Omit it to
            estimate the offset after dynamic-plane orbit subtraction.
        info: Whether to log diagnostics.
        acd: AC-dipole configuration. When given, the BPMs bracketing the AC
            dipole are refined with the ACD kick reconstruction.
        acd_only: Selects the ACD-only behaviour (requires *acd*). When
            ``True``, skip the all-BPM reconstruction and return only the ACD
            result — long-form rows for the upstream BPM, ``<acd>_before``,
            ``<acd>_after`` and the downstream BPM, with the per-turn summary
            in ``attrs["summary"]``.
        generator: Return a generator object instead of a DataFrame. With
            ``acd_only=True`` this returns :class:`ACDipolePzGenerator`;
            otherwise it returns :class:`PzGenerator`.
        barrier_s: Explicit longitudinal position of a localised element (e.g.
            an AC dipole) that the all-BPM neighbour-pair reconstruction must
            not transport across, because the free model optics do not contain
            the kick that element imparts. Pass ``None`` explicitly only when
            the reconstruction has no such localised element. This is required
            even when ``acd`` is supplied, so the all-BPM and ACD paths cannot
            silently disagree about the barrier location. Ignored for
            ``acd_only`` paths.

    Returns:
        Momentum DataFrame (all BPMs) when *generator* is ``False`` and
        *acd_only* is ``False``; the small ACD ``TfsDataFrame`` when
        ``acd_only=True``; otherwise a generator object. With *acd* and
        ``acd_only=False`` the full ACD result is attached as
        ``attrs["acd_result"]``.

    Raises:
        ValueError: If *model_details* is missing, a source request cannot be
            satisfied, or *acd_only* is set without *acd*.
    """
    if acd_only and acd is None:
        raise ValueError("acd_only requires an ACDipoleConfig via acd=")
    if not use_dispersion and measurement_pt_offset not in (None, 0.0):
        raise ValueError("A non-zero measurement_pt_offset requires dispersion")
    if generator and acd is not None and not acd_only:
        raise ValueError("generator=True with an ACDipoleConfig requires acd_only=True")
    validate_input(data)
    data = frame.prepare_data(data)
    bpm_names = [str(name) for name in data["name"].unique()]

    # Resolve once at the caller's probe momentum. If momentum was not supplied,
    # estimate it from orbit-zero-relative coordinates and regenerate the model at
    # that physical offset before either reconstruction path runs.
    if acd is not None:
        resolved_acd = resolve_ac_dipole_config(model_details, acd)
        optics_tws = resolved_acd.optics_tws
    else:
        resolved_model = resolve_model_details(model_details)
        optics_tws = resolved_model.tws
    if measurement_pt_offset is None and use_dispersion:
        measurement_pt_offset = _estimate_pt_from_prepared(data, optics_tws, frame=frame, info=info)
        model_details = replace(model_details, pt=float(measurement_pt_offset))
        if acd is not None:
            resolved_acd = resolve_ac_dipole_config(model_details, acd)
            optics_tws = resolved_acd.optics_tws
        else:
            resolved_model = resolve_model_details(model_details)
            optics_tws = resolved_model.tws

    if generator and acd_only:
        assert acd is not None
        return ACDipolePzGenerator._build(
            data=data,
            resolved_acd=resolved_acd,
            frame=frame,
            measurement_dir=measurement_dir,
            model_optics=tuple(model_optics),
            use_dispersion=use_dispersion,
            model_errors=model_errors,
            reverse_meas_tws=reverse_meas_tws,
            bpm_names=bpm_names,
        )

    if generator:
        return PzGenerator._build(
            data=data,
            resolved_model=resolved_model,
            frame=frame,
            measurement_dir=measurement_dir,
            model_optics=tuple(model_optics),
            use_dispersion=use_dispersion,
            model_errors=model_errors,
            reverse_meas_tws=reverse_meas_tws,
            measurement_pt_offset=measurement_pt_offset,
            info=info,
            barrier_s=barrier_s,
            bpm_names=bpm_names,
        )

    optics = resolve_optics(
        optics_tws=optics_tws,
        frame=frame,
        measurement_dir=measurement_dir,
        model_optics=model_optics,
        use_dispersion=use_dispersion,
        model_errors=model_errors,
        reverse_meas_tws=reverse_meas_tws,
        bpm_names=bpm_names,
    )

    if acd is not None:
        data_for_acd = data.copy(deep=True)
        if "var_x" not in data_for_acd.columns:
            data_for_acd["var_x"] = 1.0
        if "var_y" not in data_for_acd.columns:
            data_for_acd["var_y"] = 1.0
        acd_result = calculate_ac_dipole_momentum(
            data_for_acd,
            resolved_acd.optics_tws,
            ac_dipole_marker=resolved_acd.config.ac_dipole_marker,
            model=resolved_acd.model,
            dpx_tune=resolved_acd.config.driven_tunes[0],
            dpy_tune=resolved_acd.config.driven_tunes[1],
            bpm_upstream=resolved_acd.config.bpm_upstream,
            bpm_downstream=resolved_acd.config.bpm_downstream,
            smooth_lambda=resolved_acd.config.smooth_lambda,
            frame=frame,
            tracking_orbit_tws=resolved_acd.tracking_tws,
            orbit_zero_model_tws=resolved_acd.closed_orbit_tws,
            resolved_tws=optics.tws,
        )

    if acd_only:
        assert acd is not None
        return acd_result

    result = reconstruct_momenta(
        data,
        optics,
        measurement_pt_offset=measurement_pt_offset,
        info=info,
        barrier_s=barrier_s,
    )

    if acd is not None:
        result = apply_precomputed_ac_dipole_bpm_overrides(result=result, acd_result=acd_result)
        result.attrs["acd_result"] = acd_result

    return result


class ACDipolePzGenerator:
    """Fast repeated AC-dipole reconstruction for a fixed dataset.

    Built by ``calculate_pz(..., acd_only=True, generator=True)``. The
    measurement data, generated optics, generated closed orbit and BPM-window
    selection are frozen at construction; each :meth:`update` re-runs the
    reconstruction with those generated model inputs.

    Attributes:
        latest: The most recent :meth:`update` result, or ``None`` before the
            first call.
    """

    def __init__(
        self,
        *,
        prepared: PreparedACDInputs,
        resolved_acd: ResolvedACDipoleConfig,
        frame: ReconstructionFrame,
        measured: LoadedMeasurement | None,
        model_optics: Collection[OpticsCategory],
        use_dispersion: bool,
        model_errors: ModelOpticsErrors | None,
        reverse_meas_tws: bool,
        bpm_names: Collection[str],
    ) -> None:
        self._prepared = prepared
        self._resolved_acd = resolved_acd
        self._optics_tws = resolved_acd.optics_tws
        self._tracking_tws = resolved_acd.tracking_tws
        self._closed_orbit_tws = resolved_acd.closed_orbit_tws
        self._frame = frame
        self._measured = measured
        self._model_optics = tuple(model_optics)
        self._use_dispersion = use_dispersion
        self._model_errors = model_errors
        self._reverse_meas_tws = reverse_meas_tws
        self._bpm_names = list(bpm_names)
        self.latest: tfs.TfsDataFrame | None = None

    @classmethod
    def _build(
        cls,
        *,
        data: pd.DataFrame,
        resolved_acd: ResolvedACDipoleConfig,
        frame: ReconstructionFrame,
        measurement_dir: str | Path | None,
        model_optics: Collection[OpticsCategory],
        use_dispersion: bool,
        model_errors: ModelOpticsErrors | None,
        reverse_meas_tws: bool,
        bpm_names: Collection[str],
    ) -> ACDipolePzGenerator:
        """Freeze the data side of the pipeline and return a generator."""
        acd = resolved_acd.config
        measured = (
            load_measurement(
                measurement_dir,
                reverse_meas_tws=reverse_meas_tws,
                bpm_names=bpm_names,
            )
            if measurement_dir is not None
            else None
        )
        data_for_acd = data.copy(deep=True)
        if "var_x" not in data_for_acd.columns:
            data_for_acd["var_x"] = 1.0
        if "var_y" not in data_for_acd.columns:
            data_for_acd["var_y"] = 1.0
        prepared = prepare_ac_dipole_inputs(
            data_for_acd,
            resolved_acd.optics_tws,
            ac_dipole_marker=acd.ac_dipole_marker,
            model=resolved_acd.model,
            dpx_tune=acd.driven_tunes[0],
            dpy_tune=acd.driven_tunes[1],
            bpm_upstream=acd.bpm_upstream,
            bpm_downstream=acd.bpm_downstream,
            smooth_lambda=acd.smooth_lambda,
        )
        return cls(
            prepared=prepared,
            resolved_acd=resolved_acd,
            frame=frame,
            measured=measured,
            model_optics=model_optics,
            use_dispersion=use_dispersion,
            model_errors=model_errors,
            reverse_meas_tws=reverse_meas_tws,
            bpm_names=bpm_names,
        )

    @property
    def model(self) -> ACDipoleMadDriver:
        """The MAD-NG driver used for state transport (mutate magnets here)."""
        return self._resolved_acd.model

    def update(
        self,
        *,
        magnet_strengths: Mapping[str, float] | None = None,
        measurement_pt_offset: float | None = None,
    ) -> tfs.TfsDataFrame:
        """Recompute the ACD reconstruction from the generated model inputs.

        Args:
            magnet_strengths: New magnet strengths to apply to the persisted
                driver before reconstructing. When given, the driven optics and
                the undriven ``dp/p=0`` closed-orbit reference are regenerated
                from the mutated model (see
                :meth:`ACDipoleMadDriver.apply_strengths`). Tunes are *not*
                re-matched, so the strength change is observed directly.
            measurement_pt_offset: New MAD-NG ``pt`` offset from the orbit-zero
                frame. Because that frame is defined as zero, this is also the
                physical tracking ``pt``. When given, the driver's energy
                coordinate is updated before reconstructing, so the
                marker-state transport and BPM momenta re-track at this energy.
                The reconstruction reads ``self.model.pt`` live, so no rebuild is
                needed. When ``None`` the driver's current ``pt`` is kept.

        Returns:
            The small 4-point ACD ``TfsDataFrame`` (summary in
            ``attrs["summary"]``). Also stored in :attr:`latest`.
        """
        pt_changed = measurement_pt_offset is not None
        if pt_changed:
            updated_pt = float(measurement_pt_offset)
            self.model.pt = updated_pt
            self._resolved_acd.optics_model.pt = updated_pt
        if magnet_strengths is not None:
            # Transport (undriven) and optics (driven) are separate models; both
            # must see the new strengths.
            self.model.apply_strengths(magnet_strengths)
            optics_model = self._resolved_acd.optics_model
            optics_model.apply_strengths(magnet_strengths)
        if magnet_strengths is not None or pt_changed:
            optics_model = self._resolved_acd.optics_model
            self._closed_orbit_tws = self.model.run_twiss(
                observe=1, coupling=True, chrom=True, deltap=0.0
            )
            self._tracking_tws = self.model.run_twiss(
                observe=1, coupling=True, chrom=True, pt=self.model.pt
            )
            self._optics_tws = optics_model.run_twiss(
                observe=1,
                coupling=True,
                chrom=True,
                pt=self.model.pt,
            )
        optics = resolve_optics(
            optics_tws=self._optics_tws,
            frame=self._frame,
            measured=self._measured,
            model_optics=self._model_optics,
            use_dispersion=self._use_dispersion,
            model_errors=self._model_errors,
            reverse_meas_tws=self._reverse_meas_tws,
            bpm_names=self._bpm_names,
        )
        self.latest = reconstruct_from_prepared(
            self._prepared,
            self._optics_tws,
            frame=self._frame,
            tracking_orbit_tws=self._tracking_tws,
            orbit_zero_model_tws=self._closed_orbit_tws,
            resolved_tws=optics.tws,
        )
        return self.latest


class PzGenerator:
    """Fast repeated all-BPM momentum reconstruction for fixed turn data.

    Built by ``calculate_pz(..., generator=True)``. The tracking data, generated
    model optics and any measurement directory are cached at construction; each
    :meth:`update` resolves optics from those generated inputs and reconstructs
    either all BPMs or the requested subset.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        resolved_model: ResolvedModel,
        frame: ReconstructionFrame,
        measured: LoadedMeasurement | None,
        model_optics: Collection[OpticsCategory],
        use_dispersion: bool,
        model_errors: ModelOpticsErrors | None,
        reverse_meas_tws: bool,
        measurement_pt_offset: float | None,
        info: bool,
        barrier_s: float | None,
        bpm_names: Collection[str],
    ) -> None:
        self._data = data.copy(deep=True)
        self._resolved_model = resolved_model
        self._optics_tws = self._resolved_model.tws
        self._frame = frame
        self._measured = measured
        self._model_optics = tuple(model_optics)
        self._use_dispersion = use_dispersion
        self._model_errors = model_errors
        self._reverse_meas_tws = reverse_meas_tws
        self._measurement_pt_offset = measurement_pt_offset
        self._info = info
        self._barrier_s = barrier_s
        self._bpm_names = list(bpm_names)
        self.latest: pd.DataFrame | None = None

    @classmethod
    def _build(
        cls,
        *,
        data: pd.DataFrame,
        resolved_model: ResolvedModel,
        frame: ReconstructionFrame,
        measurement_dir: str | Path | None,
        model_optics: Collection[OpticsCategory],
        use_dispersion: bool,
        model_errors: ModelOpticsErrors | None,
        reverse_meas_tws: bool,
        measurement_pt_offset: float | None,
        info: bool,
        barrier_s: float | None,
        bpm_names: Collection[str],
    ) -> PzGenerator:
        measured = (
            load_measurement(
                measurement_dir,
                reverse_meas_tws=reverse_meas_tws,
                bpm_names=bpm_names,
            )
            if measurement_dir is not None
            else None
        )
        return cls(
            data=data,
            resolved_model=resolved_model,
            frame=frame,
            measured=measured,
            model_optics=model_optics,
            use_dispersion=use_dispersion,
            model_errors=model_errors,
            reverse_meas_tws=reverse_meas_tws,
            measurement_pt_offset=measurement_pt_offset,
            info=info,
            barrier_s=barrier_s,
            bpm_names=bpm_names,
        )

    @property
    def model(self) -> ACDipoleMadDriver:
        """The MAD-NG driver used to generate the optics (mutate magnets here)."""
        return self._resolved_model.model

    def update(
        self,
        *,
        magnet_strengths: Mapping[str, float] | None = None,
        bpm_names: Collection[str] | None = None,
        measurement_pt_offset: float | None = None,
    ) -> pd.DataFrame:
        """Recompute momentum for an optional BPM subset.

        When *magnet_strengths* is given they are applied to the persisted driver
        and the model optics are regenerated (a new closed orbit and new optics),
        without re-matching the tunes. *measurement_pt_offset* is relative to
        the orbit-zero frame (whose coordinate is exactly zero); it overrides
        the build-time value for this call and all subsequent ones. When
        ``None`` the build-time value is kept.
        """
        if measurement_pt_offset is not None:
            self._measurement_pt_offset = float(measurement_pt_offset)
        if magnet_strengths is not None:
            model = self._resolved_model.model
            model.apply_strengths(magnet_strengths)
            self._optics_tws = model.run_twiss(observe=1, coupling=True, chrom=True, pt=model.pt)
        optics = resolve_optics(
            optics_tws=self._optics_tws,
            frame=self._frame,
            measured=self._measured,
            model_optics=self._model_optics,
            use_dispersion=self._use_dispersion,
            model_errors=self._model_errors,
            reverse_meas_tws=self._reverse_meas_tws,
            bpm_names=self._bpm_names,
        )
        self.latest = reconstruct_momenta(
            self._data,
            optics,
            measurement_pt_offset=self._measurement_pt_offset,
            info=self._info,
            barrier_s=self._barrier_s,
            bpm_names=bpm_names,
        )
        return self.latest
