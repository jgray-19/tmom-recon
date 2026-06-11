"""Unified momentum-reconstruction entry point.

:func:`calculate_pz` is the single public API: it resolves the optics sources
(model twiss and/or omc3 measurement directory, per category), runs the
neighbour-pair reconstruction, and optionally refines the BPMs around an AC
dipole with the ACD kick reconstruction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from tmom_recon.acd.integration import (
    ACDipoleConfig,
    apply_precomputed_ac_dipole_bpm_overrides_inplace,
    ensure_position_variances,
    run_ac_dipole_reconstruction,
)
from tmom_recon.acd.reconstruction import (
    prepare_ac_dipole_inputs,
    reconstruct_from_prepared,
)
from tmom_recon.data.config import POSITION_STD_DEV
from tmom_recon.lattice.core import get_rng, inject_noise_xy_inplace, validate_input
from tmom_recon.optics import (
    LoadedMeasurement,
    ModelOpticsErrors,
    OpticsCategory,
    load_measurement,
    resolve_optics,
)
from tmom_recon.physics.transverse import reconstruct_momenta

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Collection
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import tfs

    from tmom_recon.acd.madng_driver import ACDipoleMadDriver
    from tmom_recon.acd.reconstruction import PreparedACDInputs

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ACDipoleConfig",
    "ACDipolePzGenerator",
    "ModelOpticsErrors",
    "calculate_pz",
]


def calculate_pz(
    data: pd.DataFrame,
    *,
    model_tws: pd.DataFrame | None = None,
    measurement_dir: str | Path | None = None,
    model_optics: Collection[OpticsCategory] = (),
    use_dispersion: bool = True,
    model_errors: ModelOpticsErrors | None = None,
    reverse_meas_tws: bool = False,
    dpp_override: float | None = None,
    inject_noise: bool | float = False,
    rng: np.random.Generator | None = None,
    info: bool = True,
    acd: ACDipoleConfig | None = None,
    acd_only: bool | Literal["generator"] = False,
) -> pd.DataFrame | tfs.TfsDataFrame | ACDipolePzGenerator:
    """Reconstruct transverse momenta at every BPM from turn-by-turn data.

    Optics are sourced per category (``phase``, ``amplitude`` = beta/alpha,
    ``dispersion``): categories listed in *model_optics* come from
    *model_tws*; everything else comes from *measurement_dir* when given,
    model otherwise. Uncertainties are propagated automatically — measured
    errors where the measurement is used, rough *model_errors* where the
    model is used.

    Args:
        data: Turn-by-turn BPM data with columns ``name, turn, x, y`` and
            position variances ``var_x, var_y``.
        model_tws: Model twiss indexed by element name with tune headers.
        measurement_dir: omc3 optics measurement directory.
        model_optics: Optics categories forced to come from the model.
        use_dispersion: If ``False``, run a pure transverse reconstruction
            (no dispersion terms, dp/p taken as 0 unless overridden).
        model_errors: Rough uncertainties for model-sourced optics.
        reverse_meas_tws: Reverse BPM ordering in the measured phase chain
            (Beam 2 / reverse-direction data).
        dpp_override: Use this dp/p instead of estimating it.
        inject_noise: If truthy, inject Gaussian position noise (``True`` uses
            the standard BPM resolution; a float sets the std dev [m]).
        rng: Optional NumPy random generator for reproducible noise.
        info: Whether to log diagnostics.
        acd: AC-dipole configuration. When given, the BPMs bracketing the AC
            dipole are refined with the ACD kick reconstruction.
        acd_only: Selects the ACD-only behaviour (requires *acd*):

            * ``False`` (default): full all-BPM reconstruction.
            * ``True``: skip the all-BPM reconstruction and return only the ACD
              result — long-form rows for the upstream BPM, ``<acd>_before``,
              ``<acd>_after`` and the downstream BPM, with the per-turn summary
              in ``attrs["summary"]``.
            * ``"generator"``: return an :class:`ACDipolePzGenerator` that
              freezes the (noise-injected) data once and exposes a fast
              ``update(model_tws)`` to recompute the ACD result for new optics.

    Returns:
        Momentum DataFrame (all BPMs) when *acd_only* is ``False``; the small
        ACD ``TfsDataFrame`` when ``True``; an :class:`ACDipolePzGenerator` when
        ``"generator"``. With *acd* and ``acd_only=False`` the full ACD result
        is attached as ``attrs["acd_result"]``.

    Raises:
        ValueError: If no optics source is given, a source request cannot be
            satisfied, or *acd_only* is set without *acd*.
    """
    if acd_only and acd is None:
        raise ValueError("acd_only requires an ACDipoleConfig via acd=")
    if acd is not None and model_tws is None:
        raise ValueError("AC-dipole reconstruction requires model_tws for state transport")

    validate_input(data)
    bpm_names = [str(name) for name in data["name"].unique()]

    # Inject noise once so every downstream reconstruction sees identical data
    rng = get_rng(rng)
    if inject_noise is not False:
        noise_std = POSITION_STD_DEV if inject_noise is True else float(inject_noise)
        data = data.copy(deep=True)
        inject_noise_xy_inplace(data, data, rng, noise_std=noise_std)

    if acd_only == "generator":
        return ACDipolePzGenerator._build(
            data=data,
            model_tws=model_tws,
            acd=acd,
            rng=rng,
            measurement_dir=measurement_dir,
            model_optics=tuple(model_optics),
            use_dispersion=use_dispersion,
            model_errors=model_errors,
            reverse_meas_tws=reverse_meas_tws,
            bpm_names=bpm_names,
        )

    optics = resolve_optics(
        model_tws=model_tws,
        measurement_dir=measurement_dir,
        model_optics=model_optics,
        use_dispersion=use_dispersion,
        model_errors=model_errors,
        reverse_meas_tws=reverse_meas_tws,
        bpm_names=bpm_names,
    )

    if acd_only:
        return run_ac_dipole_reconstruction(data, model_tws, acd, resolved_tws=optics.tws)

    result = reconstruct_momenta(
        data,
        optics,
        inject_noise=False,
        rng=rng,
        dpp_override=dpp_override,
        info=info,
    )

    if acd is not None:
        acd_result = run_ac_dipole_reconstruction(data, model_tws, acd, resolved_tws=optics.tws)
        apply_precomputed_ac_dipole_bpm_overrides_inplace(
            result=result, acd_result=acd_result, config=acd
        )
        result.attrs["acd_result"] = acd_result

    return result


class ACDipolePzGenerator:
    """Fast repeated AC-dipole reconstruction for a fixed dataset, varying optics.

    Built by ``calculate_pz(..., acd_only="generator")``. The measurement data
    (with noise injected once) and the BPM-window selection are frozen at
    construction; each :meth:`update` only re-runs the optics-dependent part of
    the pipeline for a new model twiss, returning the small 4-point ACD frame.

    The MAD-NG driver (``self.model``) may have its magnets changed between
    updates — transport is re-tracked each time, so results stay correct. To
    drive an optics optimiser / live monitor: mutate the magnets on
    ``generator.model``, then call ``generator.update()`` (the new twiss is read
    back from the driver) or ``generator.update(model_tws)`` with an explicit
    twiss.

    Attributes:
        latest: The most recent :meth:`update` result, or ``None`` before the
            first call.
    """

    def __init__(
        self,
        *,
        prepared: PreparedACDInputs,
        acd: ACDipoleConfig,
        measured: LoadedMeasurement | None,
        model_optics: Collection[OpticsCategory],
        use_dispersion: bool,
        model_errors: ModelOpticsErrors | None,
        reverse_meas_tws: bool,
        bpm_names: Collection[str],
    ) -> None:
        self._prepared = prepared
        self._acd = acd
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
        model_tws: pd.DataFrame,
        acd: ACDipoleConfig,
        rng: np.random.Generator,
        measurement_dir: str | Path | None,
        model_optics: Collection[OpticsCategory],
        use_dispersion: bool,
        model_errors: ModelOpticsErrors | None,
        reverse_meas_tws: bool,
        bpm_names: Collection[str],
    ) -> ACDipolePzGenerator:
        """Freeze the data side of the pipeline and return a generator.

        *data* must already have had noise injected (done once by
        :func:`calculate_pz`); the ACD prepare step is run with
        ``inject_noise=False`` so it is not re-applied.
        """
        measured = (
            load_measurement(
                measurement_dir,
                reverse_meas_tws=reverse_meas_tws,
                bpm_names=bpm_names,
            )
            if measurement_dir is not None
            else None
        )
        prepared = prepare_ac_dipole_inputs(
            ensure_position_variances(data),
            model_tws,
            ac_dipole_marker=acd.ac_dipole_marker,
            model=acd.model,
            dpx_tune=acd.dpx_tune,
            dpy_tune=acd.dpy_tune,
            bpm_upstream=acd.bpm_upstream,
            bpm_downstream=acd.bpm_downstream,
            smooth_lambda=acd.smooth_lambda,
            inject_noise=False,
            rng=rng,
        )
        return cls(
            prepared=prepared,
            acd=acd,
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
        return self._acd.model

    def update(self, model_tws: pd.DataFrame | None = None) -> tfs.TfsDataFrame:
        """Recompute the ACD reconstruction for a new model twiss.

        Args:
            model_tws: New model optics (indexed by element name, lowercase
                optics columns and ``q1`` / ``q2`` tune headers). When ``None``,
                a fresh twiss is read back from the driver via
                ``self.model.run_twiss(observe=0)`` (use after mutating magnets).

        Returns:
            The small 4-point ACD ``TfsDataFrame`` (summary in
            ``attrs["summary"]``). Also stored in :attr:`latest`.
        """
        if model_tws is None:
            model_tws = self.model.run_twiss(observe=0)
        optics = resolve_optics(
            model_tws=model_tws,
            measured=self._measured,
            model_optics=self._model_optics,
            use_dispersion=self._use_dispersion,
            model_errors=self._model_errors,
            reverse_meas_tws=self._reverse_meas_tws,
            bpm_names=self._bpm_names,
        )
        self.latest = reconstruct_from_prepared(self._prepared, model_tws, resolved_tws=optics.tws)
        return self.latest
