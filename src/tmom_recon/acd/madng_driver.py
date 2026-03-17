"""Minimal MAD-NG driver for AC-dipole direct state tracking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from pymadng_utils.mad import CoreMadInterface

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)
N_COORD = 4


class ACDipoleTrackingError(RuntimeError):
    """Raised when MAD-NG fails to track an AC-dipole state batch."""

    def __init__(
        self,
        *,
        source_element: str,
        target_element: str,
        direction: int,
        range_name: str,
        endpoint_a: dict[str, object] | None,
        endpoint_b: dict[str, object] | None,
        mad_logfile: str | None,
        error: str | None = None,
    ) -> None:
        lines = [
            f"MAD-NG failed to track states {source_element} -> {target_element} (dir={direction})",
            f"range={range_name}",
        ]
        if endpoint_a is not None:
            lines.append(f"endpoint_a={endpoint_a}")
        if endpoint_b is not None:
            lines.append(f"endpoint_b={endpoint_b}")
        if mad_logfile is not None:
            lines.append(f"mad_logfile={mad_logfile}")
        if error is not None:
            lines.append(f"error={error}")
        super().__init__(" | ".join(lines))
        self.source_element = source_element
        self.target_element = target_element
        self.direction = direction
        self.range_name = range_name
        self.endpoint_a = endpoint_a
        self.endpoint_b = endpoint_b
        self.mad_logfile = mad_logfile
        self.error = error


class ACDipoleMadDriver(CoreMadInterface):
    """Simplified MAD-NG driver dedicated to AC-dipole state tracking.

    This strips away all Kalman-specific sensitivity and knob machinery and only
    keeps what the AC-dipole reconstruction needs:
    - sequence loading and beam setup,
    - optional cycling to a chosen BPM,
    - observation of BPMs plus the requested AC-dipole element,
    - direct particle tracking for batches of ``[x, px, y, py]`` states.
    """

    def __init__(
        self,
        *,
        sequence_file: Path,
        beam: int,
        beam_energy: float,
        deltap: float = 0.0,
        bpm_pattern: str = "BPM",
        observed_elements: str | list[str] | None = None,
        debug: bool = False,
        mad_logfile: Path | None = None,
        discard_mad_output: bool = False,
    ) -> None:
        stdout = None
        redirect_stderr = False
        if mad_logfile is not None:
            stdout = mad_logfile
            redirect_stderr = True
        elif discard_mad_output:
            stdout = "/dev/null"
            redirect_stderr = True

        super().__init__(
            stdout=stdout,
            redirect_stderr=redirect_stderr,
            debug=debug,
        )
        self._mad_logfile = str(mad_logfile) if mad_logfile is not None else None
        self._element_cache: dict[str, dict[str, object] | None] = {}
        self.load_sequence(sequence_file, f"lhcb{beam}")
        self.setup_beam(beam_energy)
        self.mad["DELTAP"] = deltap
        self.observe_elements(bpm_pattern)
        self.add_observed_elements(observed_elements)

    def add_observed_elements(self, elements: str | list[str] | None) -> None:
        if elements is None:
            return
        if isinstance(elements, str):
            elements = [elements]
        element_lines = "\n".join(
            f'loaded_sequence:select(observed, {{pattern="{element}"}})' for element in elements
        )
        self.mad.send(
            f"""
local observed in MAD.element.flags
{element_lines}
"""
        )

    def _get_range(self, source_element: str, target_element: str) -> str:
        # MAD-NG expects X0 at the first element in the range for both
        # forward and backward tracking. Only ``dir`` changes sign.
        return f"{source_element}/{target_element}"

    def _describe_element(self, element_name: str) -> dict[str, object] | None:
        if element_name in self._element_cache:
            return self._element_cache[element_name]
        self.mad.send(
            """
--begin
local element_name = py:recv()
local elem = loaded_sequence[element_name]
if not elem then
    py:send(nil)
else
    py:send({
        name = elem.name,
        kind = elem.kind,
        at = elem.at or 0,
        l = elem.l or 0,
    }, true)
end
--end
"""
        ).send(element_name)
        description = self.mad.recv()
        self._element_cache[element_name] = description
        return description

    def track_particles(
        self,
        source_element: str,
        target_element: str,
        states: np.ndarray,
        *,
        direction: int = 1,
    ) -> np.ndarray:
        if direction not in (-1, 1):
            raise ValueError(f"direction must be +/- 1, got {direction}")

        range_name = self._get_range(source_element, target_element)
        endpoint_a = self._describe_element(source_element)
        endpoint_b = self._describe_element(target_element)
        state_array = np.asarray(states, dtype=float)
        if state_array.ndim != 2 or state_array.shape[1] != 4:
            raise ValueError(f"states must have shape (n, 4), got {state_array.shape}")
        if len(state_array) == 0:
            return np.empty((0, 4), dtype=float)

        x0_particles = [
            {
                "x": float(x),
                "px": float(px),
                "y": float(y),
                "py": float(py),
                "t": 0.0,
                "pt": 0.0,
            }
            for x, px, y, py in state_array
        ]
        self.mad.send(
            """
--begin
range = py:recv()
x0_particles = py:recv()
direction = py:recv()

tbl, flw = track {
    sequence=loaded_sequence,
    range=range,
    X0=x0_particles,
    save=true,
    nturn=1,
    dir=direction,
    observe=1,
    deltap=DELTAP
}
py:send(true)
--end
"""
        ).send(range_name).send(x0_particles).send(direction)
        try:
            if not bool(self.mad.recv()):
                raise RuntimeError("Unexpected acknowledgement from MAD-NG track call")
            track_df = self.mad.tbl.to_df(force_pandas=True)
        except Exception as exc:
            raise ACDipoleTrackingError(
                source_element=source_element,
                target_element=target_element,
                direction=direction,
                range_name=range_name,
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                mad_logfile=self._mad_logfile,
                error=f"{type(exc).__name__}: {exc}",
            ) from exc

        if "id" not in track_df.columns:
            raise ValueError("Track table is missing required particle id column 'id'")

        final_rows = (
            track_df.sort_values(["id", "turn", "s"], kind="stable")
            .groupby("id", sort=False, as_index=False)
            .tail(1)
            .sort_values("id", kind="stable")
        )
        tracked_states = final_rows[["x", "px", "y", "py"]].to_numpy(dtype=float)
        if tracked_states.shape != (len(x0_particles), N_COORD):
            raise ValueError(
                f"Tracked particle batch must have shape ({len(x0_particles)}, {N_COORD}), got {tracked_states.shape}"
            )
        return tracked_states
