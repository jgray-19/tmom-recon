"""Focused all-BPM and AC-dipole momentum reconstruction workflows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tmom_recon.acd.global_markers import (
    apply_global_marker_states,
    cross_mode_bases,
    forced_mode_responses,
    normal_mode_responses,
)
from tmom_recon.acd.integration import (
    ACDipoleConfig,
    ResolvedACDipoleConfig,
    resolve_ac_dipole_config,
)
from tmom_recon.acd.reconstruction import (
    _calculate_ac_dipole_momentum,
    prepare_ac_dipole_inputs,
    reconstruct_from_prepared,
)
from tmom_recon.data.checks import validate_input
from tmom_recon.model import (
    ModelDetails,
    ResolvedModel,
    resolve_model_details,
)
from tmom_recon.optics import (
    LoadedMeasurement,
    ModelOpticsErrors,
    OpticsInput,
    load_measurement,
    resolve_optics,
)
from tmom_recon.orbit_reference import build_orbit_reference
from tmom_recon.physics.transverse import reconstruct_momenta

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Collection, Mapping

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
    "OpticsInput",
    "PzGenerator",
    "calculate_acd_global_pz",
    "calculate_acd_pz",
    "calculate_pz",
]


def calculate_pz(
    data: pd.DataFrame,
    model_details: ModelDetails,
    *,
    closed_orbit_at_zero: pd.DataFrame,
    orbit_mode: str,
    optics: OpticsInput = OpticsInput(),
    model_errors: ModelOpticsErrors | None = None,
    info: bool = True,
    acd: ACDipoleConfig | None = None,
    barrier_s: float | None,
) -> pd.DataFrame:
    """Reconstruct all BPM momenta from explicit model and optics inputs.

    *acd* is not an AC-dipole reconstruction -- that is
    :func:`calculate_acd_pz`. It states that *data* was taken with the AC dipole
    driving, so the neighbour-pair optics must be the **driven** optics. Free
    optics against driven data is a modelling error, not a small one: the driven
    beta beating is the whole reason the driven twiss exists. The AC dipole is
    also a localised kick the free transport does not contain, so *barrier_s*
    must still be given, exactly as for a free reconstruction.
    """
    validate_input(data)
    if acd is None:
        resolved_model = resolve_model_details(model_details)
        optics_tws = resolved_model.tws
        zero_tws = resolved_model.zero_tws
    else:
        resolved_acd = resolve_ac_dipole_config(model_details, acd)
        optics_tws = resolved_acd.optics_tws
        zero_tws = resolved_acd.closed_orbit_tws
    reference = build_orbit_reference(
        closed_orbit_at_zero, orbit_mode, zero_tws, _reference_names(data, zero_tws)
    )
    resolved_optics = _resolve(
        optics_tws,
        zero_tws,
        data,
        optics,
        model_errors,
        dpp=_dpp(model_details.accelerator, model_details.pt),
    )
    return reconstruct_momenta(
        data, resolved_optics, reference, pt=model_details.pt, info=info, barrier_s=barrier_s
    )


def calculate_acd_pz(
    data: pd.DataFrame,
    model_details: ModelDetails,
    config: ACDipoleConfig,
    *,
    closed_orbit_at_zero: pd.DataFrame,
    orbit_mode: str,
    optics: OpticsInput = OpticsInput(),
    model_errors: ModelOpticsErrors | None = None,
) -> tfs.TfsDataFrame:
    """Reconstruct the driven kick and adjacent BPM states around an AC dipole."""
    validate_input(data)
    resolved_acd = resolve_ac_dipole_config(model_details, config)
    reference = build_orbit_reference(
        closed_orbit_at_zero,
        orbit_mode,
        resolved_acd.closed_orbit_tws,
        _reference_names(data, resolved_acd.closed_orbit_tws),
    )
    resolved_optics = _resolve(
        resolved_acd.optics_tws,
        resolved_acd.closed_orbit_tws,
        data,
        optics,
        model_errors,
        dpp=_dpp(model_details.accelerator, model_details.pt),
    )
    return _calculate_acd(data, resolved_acd, reference, resolved_optics)


def calculate_acd_global_pz(
    data: pd.DataFrame,
    model_details: ModelDetails,
    config: ACDipoleConfig,
    *,
    closed_orbit_at_zero: pd.DataFrame,
    orbit_mode: str,
    optics: OpticsInput = OpticsInput(),
    model_errors: ModelOpticsErrors | None = None,
    fit_cross_modes: bool = False,
) -> tfs.TfsDataFrame:
    """AC-dipole reconstruction with marker states from the global driven optics.

    Same inputs and output layout as :func:`calculate_acd_pz`. The tracked
    reconstruction runs first; the driven lines of the ``marker_before`` /
    ``marker_after`` rows are then replaced by the exact coupled forced response
    to the two AC dipoles, one amplitude each, fitted over x and y at all BPMs
    (:func:`tmom_recon.acd.global_markers.apply_global_marker_states`). The
    static offsets, the variances and the BPM rows stay tracked. The per-mode
    fits are in ``attrs["acd_global_markers"]``.

    The kick is then the fitted drive amplitude itself instead of
    a difference of tracked momenta, which is far less sensitive to local model
    errors between the two neighbouring BPMs.

    ``fit_cross_modes`` also fits each drive's other-mode content at the markers
    from the BPM data (:func:`tmom_recon.acd.global_markers.cross_mode_bases`)
    instead of taking it from the model's coupling alone, so a model with too
    little coupling does not hand the fit markers with too little.
    """
    validate_input(data)
    resolved_acd = resolve_ac_dipole_config(model_details, config)
    reference = build_orbit_reference(
        closed_orbit_at_zero,
        orbit_mode,
        resolved_acd.closed_orbit_tws,
        _reference_names(data, resolved_acd.closed_orbit_tws),
    )
    resolved_optics = _resolve(
        resolved_acd.optics_tws,
        resolved_acd.closed_orbit_tws,
        data,
        optics,
        model_errors,
        dpp=_dpp(model_details.accelerator, model_details.pt),
    )
    tracked = _calculate_acd(data, resolved_acd, reference, resolved_optics)
    return _global_marker_states(
        tracked,
        _acd_data(data),
        resolved_acd.model,
        config.driven_tunes,
        fit_cross_modes=fit_cross_modes,
    )


def _global_marker_states(
    tracked: tfs.TfsDataFrame,
    acd_data: pd.DataFrame,
    model: ACDipoleMadDriver,
    driven_tunes: tuple[float, float],
    *,
    fit_cross_modes: bool,
) -> tfs.TfsDataFrame:
    """Replace the tracked marker drive lines with the global forced response."""
    # The undriven transport model: the forced response is exact with coupling,
    # the driven (lens) optics are not.
    vectors, natural_tunes = normal_mode_responses(model, model.pt)
    before = model.accelerator.acd_marker_name("before")
    after = model.accelerator.acd_marker_name("after")
    responses = forced_mode_responses(
        vectors, natural_tunes, driven_tunes, before=before, after=after
    )
    cross_bases = cross_mode_bases(vectors, before=before, after=after) if fit_cross_modes else None
    return apply_global_marker_states(
        tracked,
        acd_data,
        responses,
        driven_tunes=driven_tunes,
        cross_bases=cross_bases,
    )


def _names(data: pd.DataFrame) -> list[str]:
    return [str(name) for name in data["name"].unique()]


def _reference_names(data: pd.DataFrame, zero_tws: pd.DataFrame) -> list[str]:
    model_names = {str(name).upper() for name in zero_tws.index}
    return [name for name in _names(data) if name.upper() in model_names]


def _resolve(
    tws: pd.DataFrame,
    zero_tws: pd.DataFrame,
    data: pd.DataFrame,
    optics: OpticsInput,
    model_errors: ModelOpticsErrors | None,
    measured: LoadedMeasurement | None = None,
    *,
    dpp: float = 0.0,
):
    return resolve_optics(
        optics_tws=tws,
        zero_tws=zero_tws,
        optics=optics,
        model_errors=model_errors,
        bpm_names=_names(data),
        measured=measured,
        dpp=dpp,
    )


def _dpp(accelerator, pt: float) -> float:
    """dp/p of *pt* (MAD-NG's own conversion); the model betas need it."""
    return float(accelerator.pt2dp(pt)) if pt else 0.0


def _acd_data(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy(deep=True)
    for column in ("var_x", "var_y"):
        if column not in result:
            result[column] = 1.0
    return result


def _calculate_acd(data, resolved_acd, reference, resolved_optics):
    config = resolved_acd.config
    return _calculate_ac_dipole_momentum(
        _acd_data(data),
        resolved_acd.optics_tws,
        ac_dipole_marker=config.ac_dipole_marker,
        model=resolved_acd.model,
        dpx_tune=config.driven_tunes[0],
        dpy_tune=config.driven_tunes[1],
        bpm_upstream=config.bpm_upstream,
        bpm_downstream=config.bpm_downstream,
        smooth_lambda=config.smooth_lambda,
        reject_inconsistent_state=config.reject_inconsistent_state,
        reference=reference,
        resolved_optics=resolved_optics,
    )


class ACDipolePzGenerator:
    """Fast repeated AC-dipole reconstruction for a fixed dataset.

    Build with :meth:`ACDipolePzGenerator.build`. The measurement data,
    generated optics, generated closed orbit and BPM-window selection are
    frozen at construction; each :meth:`update` re-runs the reconstruction
    with those generated model inputs.

    Attributes:
        latest: The most recent :meth:`update` result, or ``None`` before the
            first call.
    """

    def __init__(
        self,
        *,
        prepared: PreparedACDInputs,
        resolved_acd: ResolvedACDipoleConfig,
        closed_orbit_at_zero: pd.DataFrame,
        orbit_mode: str,
        measured: LoadedMeasurement | None,
        optics: OpticsInput,
        model_errors: ModelOpticsErrors | None,
        global_markers: bool = False,
        fit_cross_modes: bool = False,
        acd_data: pd.DataFrame | None = None,
    ) -> None:
        if global_markers and acd_data is None:
            raise ValueError("global_markers needs the ACD data the lines are fitted to")
        self._prepared = prepared
        self._resolved_acd = resolved_acd
        self._optics_tws = resolved_acd.optics_tws
        self._closed_orbit_tws = resolved_acd.closed_orbit_tws
        self._closed_orbit_at_zero = closed_orbit_at_zero.copy(deep=True)
        self._orbit_mode = orbit_mode
        self._measured = measured
        self._optics = optics
        self._model_errors = model_errors
        self._global_markers = global_markers
        self._fit_cross_modes = fit_cross_modes
        self._acd_data = acd_data
        self.latest: tfs.TfsDataFrame | None = None

    @classmethod
    def build(
        cls,
        *,
        data: pd.DataFrame,
        model_details: ModelDetails,
        config: ACDipoleConfig,
        closed_orbit_at_zero: pd.DataFrame,
        orbit_mode: str,
        optics: OpticsInput = OpticsInput(),
        model_errors: ModelOpticsErrors | None = None,
        global_markers: bool = False,
        fit_cross_modes: bool = False,
    ) -> ACDipolePzGenerator:
        """Freeze the data side of the pipeline and return a generator.

        ``global_markers`` makes every :meth:`update` return the
        :func:`calculate_acd_global_pz` marker states instead of the tracked ones.
        """
        validate_input(data)
        resolved_acd = resolve_ac_dipole_config(model_details, config)
        acd = resolved_acd.config
        measured = (
            load_measurement(
                optics.measurement_dir,
                reverse_meas_tws=optics.reverse_measurement_order,
                bpm_names=_names(data),
            )
            if optics.measurement_dir is not None
            else None
        )
        prepared = prepare_ac_dipole_inputs(
            _acd_data(data),
            resolved_acd.optics_tws,
            ac_dipole_marker=acd.ac_dipole_marker,
            model=resolved_acd.model,
            dpx_tune=acd.driven_tunes[0],
            dpy_tune=acd.driven_tunes[1],
            bpm_upstream=acd.bpm_upstream,
            bpm_downstream=acd.bpm_downstream,
            smooth_lambda=acd.smooth_lambda,
            reject_inconsistent_state=acd.reject_inconsistent_state,
        )
        return cls(
            prepared=prepared,
            resolved_acd=resolved_acd,
            closed_orbit_at_zero=closed_orbit_at_zero,
            orbit_mode=orbit_mode,
            measured=measured,
            optics=optics,
            model_errors=model_errors,
            global_markers=global_markers,
            fit_cross_modes=fit_cross_modes,
            acd_data=_acd_data(data) if global_markers else None,
        )

    @property
    def model(self) -> ACDipoleMadDriver:
        """The MAD-NG driver used for state transport (mutate magnets here)."""
        return self._resolved_acd.model

    def update(
        self,
        *,
        magnet_strengths: Mapping[str, float] | None = None,
        pt: float | None = None,
        knobs: Mapping[str, float] | None = None,
    ) -> tfs.TfsDataFrame:
        """Recompute the ACD reconstruction from the generated model inputs.

        Args:
            knobs: Knob variables (e.g. tune-correction knobs) to set on both
                models before reconstructing. A caller that holds the natural
                tunes to a measurement passes the knobs it rematched with
                *magnet_strengths* applied; the AC dipole stays installed at the
                driven tunes it was built with.
            magnet_strengths: New magnet strengths to apply to the persisted
                driver before reconstructing. When given, the driven optics and
                the undriven ``dp/p=0`` closed-orbit reference are regenerated
                from the mutated model (see
                :meth:`ACDipoleMadDriver.apply_strengths`). Tunes are *not*
                re-matched, so the strength change is observed directly.
            pt: New MAD-NG ``pt`` value. When given, the driver's energy
                coordinate is updated before reconstructing, so the
                marker-state transport and BPM momenta re-track at this energy.
                The reconstruction reads ``self.model.pt`` live, so no rebuild is
                needed. When ``None`` the driver's current ``pt`` is kept.

        Returns:
            The small 4-point ACD ``TfsDataFrame`` (summary in
            ``attrs["summary"]``). Also stored in :attr:`latest`.
        """
        pt_changed = pt is not None
        if pt_changed:
            updated_pt = float(pt)
            self.model.pt = updated_pt
            self._resolved_acd.optics_model.pt = updated_pt
        if knobs is not None:
            self.model.set_knobs(knobs)
            self._resolved_acd.optics_model.set_knobs(knobs)
        if magnet_strengths is not None:
            # Transport (undriven) and optics (driven) are separate models; both
            # must see the new strengths.
            self.model.apply_strengths(magnet_strengths)
            optics_model = self._resolved_acd.optics_model
            optics_model.apply_strengths(magnet_strengths)
        if magnet_strengths is not None or pt_changed or knobs is not None:
            optics_model = self._resolved_acd.optics_model
            self._closed_orbit_tws = self.model.run_twiss(
                observe=1, coupling=True, chrom=True, deltap=0.0
            )
            self._optics_tws = optics_model.run_twiss(
                observe=1,
                coupling=True,
                chrom=True,
                pt=self.model.pt,
            )
        resolved_optics = _resolve(
            self._optics_tws,
            self._closed_orbit_tws,
            self._prepared.data,
            self._optics,
            self._model_errors,
            self._measured,
            dpp=_dpp(self.model.accelerator, self.model.pt),
        )
        reference = build_orbit_reference(
            self._closed_orbit_at_zero,
            self._orbit_mode,
            self._closed_orbit_tws,
            _reference_names(self._prepared.data, self._closed_orbit_tws),
        )
        result = reconstruct_from_prepared(
            self._prepared,
            self._optics_tws,
            reference=reference,
            resolved_optics=resolved_optics,
        )
        if self._global_markers:
            result = _global_marker_states(
                result,
                self._acd_data,
                self.model,
                self._resolved_acd.config.driven_tunes,
                fit_cross_modes=self._fit_cross_modes,
            )
        self.latest = result
        return self.latest


class PzGenerator:
    """Repeated all-BPM reconstruction for fixed data and frame."""

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        resolved_model: ResolvedModel,
        closed_orbit_at_zero: pd.DataFrame,
        orbit_mode: str,
        measured: LoadedMeasurement | None,
        optics: OpticsInput,
        model_errors: ModelOpticsErrors | None,
        info: bool,
        barrier_s: float | None,
    ) -> None:
        self._data = data.copy(deep=True)
        self._resolved_model = resolved_model
        self._optics_tws = self._resolved_model.tws
        self._closed_orbit_at_zero = closed_orbit_at_zero.copy(deep=True)
        self._orbit_mode = orbit_mode
        self._zero_tws = resolved_model.zero_tws
        self._measured = measured
        self._optics = optics
        self._model_errors = model_errors
        self._info = info
        self._barrier_s = barrier_s
        self.latest: pd.DataFrame | None = None

    @classmethod
    def build(
        cls,
        *,
        data: pd.DataFrame,
        model_details: ModelDetails,
        closed_orbit_at_zero: pd.DataFrame,
        orbit_mode: str,
        optics: OpticsInput = OpticsInput(),
        model_errors: ModelOpticsErrors | None = None,
        info: bool = True,
        barrier_s: float | None,
    ) -> PzGenerator:
        validate_input(data)
        resolved_model = resolve_model_details(model_details)
        measured = (
            load_measurement(
                optics.measurement_dir,
                reverse_meas_tws=optics.reverse_measurement_order,
                bpm_names=_names(data),
            )
            if optics.measurement_dir is not None
            else None
        )
        return cls(
            data=data,
            resolved_model=resolved_model,
            closed_orbit_at_zero=closed_orbit_at_zero,
            orbit_mode=orbit_mode,
            measured=measured,
            optics=optics,
            model_errors=model_errors,
            info=info,
            barrier_s=barrier_s,
        )

    @property
    def model(self) -> ACDipoleMadDriver:
        """The MAD-NG driver used to generate the optics (mutate magnets here)."""
        return self._resolved_model.model

    def update(
        self,
        *,
        magnet_strengths: Mapping[str, float] | None = None,
        pt: float | None = None,
        bpm_names: Collection[str] | None = None,
    ) -> pd.DataFrame:
        """Recompute momentum for an optional BPM subset.

        When *magnet_strengths* is given they are applied to the persisted driver
        and the model optics are regenerated (a new closed orbit and new optics),
        without re-matching the tunes. The momentum remains the explicit value
        in the original :class:`ModelDetails`.
        """
        model = self._resolved_model.model
        if pt is not None:
            model.pt = float(pt)
        if magnet_strengths is not None:
            model.apply_strengths(magnet_strengths)
        if magnet_strengths is not None:
            self._zero_tws = model.run_twiss(observe=1, coupling=True, chrom=True, deltap=0.0)
        if magnet_strengths is not None or pt is not None:
            self._optics_tws = model.run_twiss(observe=1, coupling=True, chrom=True, pt=model.pt)
        resolved_optics = _resolve(
            self._optics_tws,
            self._zero_tws,
            self._data,
            self._optics,
            self._model_errors,
            self._measured,
            dpp=_dpp(model.accelerator, model.pt),
        )
        reference = build_orbit_reference(
            self._closed_orbit_at_zero,
            self._orbit_mode,
            self._zero_tws,
            _reference_names(self._data, self._zero_tws),
        )
        self.latest = reconstruct_momenta(
            self._data,
            resolved_optics,
            reference,
            pt=model.pt,
            info=self._info,
            barrier_s=self._barrier_s,
            bpm_names=bpm_names,
        )
        return self.latest
