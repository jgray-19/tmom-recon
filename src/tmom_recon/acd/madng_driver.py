"""Minimal MAD-NG driver for AC-dipole direct state tracking."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
from pymadng_utils.mad import KnobMadInterface

if TYPE_CHECKING:
    from pathlib import Path

    from pymadng_utils.accelerators import Accelerator

N_COORD = 6


class ACDipoleTrackingError(RuntimeError):
    """Raised when MAD-NG fails to track an AC-dipole state batch.

    Attributes:
        source_element: Start of the failed tracking segment.
        target_element: End of the failed tracking segment.
        direction: ``+1`` for forward tracking, ``-1`` for backward.
        range_name: The MAD-NG range string ``"source/target"`` that was used.
        mad_logfile: Path to the MAD-NG log file, if one was configured.
        error: String representation of the underlying exception, if any.
    """

    def __init__(
        self,
        *,
        source_element: str,
        target_element: str,
        direction: int,
        range_name: str,
        mad_logfile: str | None,
        error: str | None = None,
    ) -> None:
        lines = [
            f"MAD-NG failed to track states {source_element} -> {target_element} (dir={direction})",
            f"range={range_name}",
        ]
        if mad_logfile is not None:
            lines.append(f"mad_logfile={mad_logfile}")
        if error is not None:
            lines.append(f"error={error}")
        super().__init__(" | ".join(lines))
        self.source_element = source_element
        self.target_element = target_element
        self.direction = direction
        self.range_name = range_name
        self.mad_logfile = mad_logfile
        self.error = error


class ACDipoleMadDriver(KnobMadInterface):
    """MAD-NG driver dedicated to AC-dipole state tracking.

    Wraps :class:`KnobMadInterface` to provide:

    - sequence loading and beam setup via an accelerator object,
    - BPM observation plus any additional requested elements,
    - direct particle tracking for batches of phase-space states,
    - linearised transport-map (Jacobian) computation via differential algebra.

    Args:
        accelerator: Owns sequence loading, beam parameters, and BPM patterns.
        deltap: Momentum offset dp/p. Converted to canonical ``pt`` at
            construction time.
        observed_elements: Element name(s) to observe in addition to all BPMs.
        tune_knobs_file: Knob file applied for tune corrections.
        corrector_knobs_file: Knob file applied for corrector settings.
        debug: If ``True``, enables MAD-NG debug output.
        mad_logfile: If given, redirect MAD-NG stdout and stderr to this path.
        discard_mad_output: If ``True`` and *mad_logfile* is ``None``, suppress
            all MAD-NG output.
    """

    def __init__(
        self,
        *,
        accelerator: Accelerator,
        deltap: float = 0.0,
        observed_elements: str | list[str] | None = None,
        tune_knobs_file: Path | None = None,
        corrector_knobs_file: Path | None = None,
        debug: bool = False,
        mad_logfile: Path | None = None,
        discard_mad_output: bool = False,
    ) -> None:
        stdout, redirect_stderr = _resolve_mad_output(mad_logfile, discard_mad_output)
        super().__init__(
            accelerator=accelerator,
            stdout=stdout,
            redirect_stderr=redirect_stderr,
            debug=debug,
        )
        self._mad_logfile = str(mad_logfile) if mad_logfile is not None else None
        self.deltap = float(deltap)
        self.pt: float = (
            self.mad.send("py:send(MAD.gphys.dp2pt(py:recv(), loaded_sequence.beam.beta))")
            .send(self.deltap)
            .recv()
        )
        if tune_knobs_file is not None:
            self.set_knobs(tune_knobs_file)
        if corrector_knobs_file is not None:
            self.set_knobs(corrector_knobs_file)
        self.twiss_elements = self.run_twiss(observe=0)
        self.observe(self.accelerator.bpm_pattern)
        for element in _normalise_element_list(observed_elements):
            self.observe(element, unobserve_first=False)

    # ------------------------------------------------------------------
    # Public tracking API
    # ------------------------------------------------------------------

    def track_particles(
        self,
        source_element: str,
        target_element: str,
        states: np.ndarray,
        *,
        direction: int = 1,
    ) -> np.ndarray:
        """Track a batch of phase-space states from one element to another.

        Args:
            source_element: Lattice element name where states originate.
            target_element: Lattice element name where tracking ends.
            states: Shape ``(n, 4)`` with columns ``[x, px, y, py]``, or
                ``(n, 6)`` with ``[x, px, y, py, t, pt]``. For the 6-column
                form, ``t`` must be zero and ``pt`` must equal ``self.pt`` for
                all particles.
            direction: ``+1`` for forward tracking, ``-1`` for backward.

        Returns:
            Shape ``(n, 6)`` array ``[x, px, y, py, t, pt]`` evaluated at
            *target_element*.

        Raises:
            ACDipoleTrackingError: If MAD-NG reports that fewer particles were
                tracked than sent.
        """
        range_name, n_particles = self._setup_tracking(
            source_element, target_element, states, direction
        )
        self.mad.send(
            """
tbl, flw = track {
    sequence=loaded_sequence,
    range=range,
    X0=x0_particles,
    save=true,
    nturn=1,
    dir=direction,
    observe=1,
}
py:send(flw.tpar == flw.npar)
"""
        )
        with self._tracking_errors(source_element, target_element, direction, range_name):
            _assert_all_tracked(self.mad.recv())
            track_df = self.mad.tbl.to_df(force_pandas=True)

        if "id" not in track_df.columns:
            raise ValueError("Track table is missing required particle id column 'id'")

        # Preserve MAD-NG's emitted row order when selecting the endpoint.
        # Sorting by increasing s breaks backward tracks because the first row
        # is then the endpoint while the last is the starting-marker state.
        final_rows = (
            track_df.reset_index(drop=True)
            .groupby("id", sort=False, as_index=False)
            .tail(1)
            .sort_values("id", kind="stable")
        )
        tracked_states = final_rows[["x", "px", "y", "py", "t", "pt"]].to_numpy(dtype=float)
        if tracked_states.shape != (n_particles, N_COORD):
            raise ValueError(
                f"Tracked particle batch must have shape ({n_particles}, {N_COORD}),"
                f" got {tracked_states.shape}"
            )
        return tracked_states

    def compute_jacobian(
        self,
        source_element: str,
        target_element: str,
        base_states: np.ndarray,
        direction: int = 1,
    ) -> np.ndarray:
        """Compute the 6x6 linearised transport map for each base state.

        Uses MAD-NG differential algebra (``damap`` with ``mo=1``) to propagate
        each state and extract its first-order map.

        Args:
            source_element: Lattice element name where states originate.
            target_element: Lattice element name where tracking ends.
            base_states: Shape ``(n, 4)`` or ``(n, 6)`` — same convention as
                :meth:`track_particles`.
            direction: ``+1`` for forward, ``-1`` for backward.

        Returns:
            Shape ``(n, 6, 6)`` where ``maps[i]`` is the first-order transport
            map for the *i*-th base state.

        Raises:
            ACDipoleTrackingError: If MAD-NG reports that fewer particles were
                tracked than sent.
        """
        range_name, n_particles = self._setup_tracking(
            source_element, target_element, base_states, direction
        )
        self.mad.send(
            """
x0_da = MAD.damap{nv=6, mo=1}
list_da = {}
for i, particle in ipairs(x0_particles) do
    list_da[i] = x0_da:copy()
    list_da[i]:set0(particle)
end

tbl, flw = track {
    sequence=loaded_sequence,
    range=range,
    X0=list_da,
    save=true,
    nturn=1,
    dir=direction,
    observe=1,
}
py:send(flw.tpar == flw.npar)
--end
"""
        )
        with self._tracking_errors(source_element, target_element, direction, range_name):
            _assert_all_tracked(self.mad.recv())

        maps = np.zeros((n_particles, N_COORD, N_COORD), dtype=float)
        for i in range(n_particles):
            maps[i] = self.mad.send(f"py:send(list_da[{i + 1}]:get1())").recv()
        return maps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_tracking(
        self,
        source_element: str,
        target_element: str,
        states: np.ndarray,
        direction: int,
    ) -> tuple[str, int]:
        """Validate inputs and send tracking parameters to MAD-NG.

        Args:
            source_element: Lattice element name where states originate.
            target_element: Lattice element name where tracking ends.
            states: Raw state array; passed to :meth:`_validate_states`.
            direction: ``+1`` or ``-1``.

        Returns:
            A ``(range_name, n_particles)`` tuple where *range_name* is the
            MAD-NG range string ``"source/target"`` and *n_particles* is the
            number of particles sent.
        """
        if direction not in (-1, 1):
            raise ValueError(f"direction must be +/- 1, got {direction}")
        state_array = self._validate_states(states)
        range_name = f"{source_element}/{target_element}"
        particles = self._build_particle_dicts(state_array)
        self.mad.send(
            """
range = py:recv()
x0_particles = py:recv()
direction = py:recv()
"""
        ).send(range_name).send(particles).send(direction)
        return range_name, len(particles)

    def _validate_states(self, states: np.ndarray) -> np.ndarray:
        """Validate and normalise a state batch to shape ``(n, 4)``.

        Args:
            states: Shape ``(n, 4)`` or ``(n, 6)``. For the 6-column form the
                longitudinal entries must satisfy ``t=0`` and ``pt=self.pt``.

        Returns:
            Shape ``(n, 4)`` transverse-only state array.

        Raises:
            ValueError: If the shape is wrong, there are no particles, or the
                longitudinal entries in a 6-column input do not match the
                expected values.
        """
        arr = np.asarray(states, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (4, 6):
            raise ValueError(f"states must have shape (n, 4) or (n, 6), got {arr.shape}")
        if len(arr) == 0:
            raise ValueError(f"states must have at least one particle, got shape {arr.shape}")
        if arr.shape[1] == 6:
            if not np.allclose(arr[:, 4], 0.0):
                raise ValueError(f"states[:, 4] (t) must be 0.0, got {arr[:, 4]}")
            if not np.allclose(arr[:, 5], self.pt):
                raise ValueError(f"states[:, 5] (pt) must be {self.pt}, got {arr[:, 5]}")
            arr = arr[:, :4]
        return arr

    def _build_particle_dicts(self, state_array: np.ndarray) -> list[dict]:
        """Convert an ``(n, 4)`` state array to MAD-NG particle-dict format.

        Args:
            state_array: Shape ``(n, 4)`` with columns ``[x, px, y, py]``.

        Returns:
            One dict per particle with keys ``x``, ``px``, ``y``, ``py``,
            ``t`` (always ``0.0``), and ``pt`` (always ``self.pt``).
        """
        # First check for nans just in case.
        if np.isnan(state_array).any():
            index = np.argwhere(np.isnan(state_array))
            raise ValueError(f"State array contains NaNs at indices {index.tolist()}")
        return [
            {
                "x": float(x),
                "px": float(px),
                "y": float(y),
                "py": float(py),
                "t": 0.0,
                "pt": float(self.pt),
            }
            for x, px, y, py in state_array
        ]

    @contextmanager
    def _tracking_errors(
        self,
        source_element: str,
        target_element: str,
        direction: int,
        range_name: str,
    ) -> Generator[None, None, None]:
        """Context manager that re-raises any exception as :class:`ACDipoleTrackingError`.

        Args:
            source_element: Forwarded to :class:`ACDipoleTrackingError`.
            target_element: Forwarded to :class:`ACDipoleTrackingError`.
            direction: Forwarded to :class:`ACDipoleTrackingError`.
            range_name: Forwarded to :class:`ACDipoleTrackingError`.
        """
        try:
            yield
        except ACDipoleTrackingError:
            raise
        except Exception as exc:
            raise ACDipoleTrackingError(
                source_element=source_element,
                target_element=target_element,
                direction=direction,
                range_name=range_name,
                mad_logfile=self._mad_logfile,
                error=f"{type(exc).__name__}: {exc}",
            ) from exc


# ------------------------------------------------------------------
# Module-level helpers (no access to driver state needed)
# ------------------------------------------------------------------


def _resolve_mad_output(
    mad_logfile: Path | None, discard_mad_output: bool
) -> tuple[str | None, bool]:
    """Return the ``(stdout, redirect_stderr)`` pair for ``KnobMadInterface``.

    Args:
        mad_logfile: If not ``None``, redirect output to this file.
        discard_mad_output: If ``True`` and *mad_logfile* is ``None``, redirect
            to ``/dev/null``.

    Returns:
        A ``(stdout, redirect_stderr)`` tuple where *stdout* is a path string
        for stdout redirection (or ``None`` to keep the default) and
        *redirect_stderr* indicates whether stderr should also be redirected.
    """
    if mad_logfile is not None:
        return str(mad_logfile), True
    if discard_mad_output:
        return "/dev/null", True
    return None, False


def _normalise_element_list(elements: str | list[str] | None) -> list[str]:
    """Coerce *elements* to a plain ``list[str]``, returning ``[]`` for ``None``.

    Args:
        elements: A single element name, a list of names, or ``None``.

    Returns:
        Always a plain list; never ``None``.
    """
    if elements is None:
        return []
    if isinstance(elements, str):
        return [elements]
    return list(elements)


def _assert_all_tracked(success_flag: object) -> None:
    """Raise ``RuntimeError`` if MAD-NG did not track all particles.

    Args:
        success_flag: The value returned by ``py:send(flw.tpar == flw.npar)``
            in MAD-NG. Truthy means all particles completed tracking.

    Raises:
        RuntimeError: If *success_flag* is falsy.
    """
    if not bool(success_flag):
        raise RuntimeError("MAD-NG: fewer particles tracked than sent (flw.tpar != flw.npar)")
