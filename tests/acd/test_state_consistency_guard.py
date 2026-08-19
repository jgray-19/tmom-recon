"""The AC-dipole state-consistency guard and its typed rejection.

The guard is the last thing standing between a badly-reconstructed acquisition
and a quad fit built on it, so its verdict has to be actionable rather than just
fatal: batch callers need to exclude *that acquisition* while still failing hard
if the rejections are systemic. That is what
:class:`~tmom_recon.acd.ACDipoleStateConsistencyError` is for, and these tests
pin down the behaviour a caller relies on -- the type, the attributes, and the
tolerance ladder's shape.

The tolerance itself is deliberately not softened anywhere here. Below the 1 mm
threshold the fixed 1e-4 m floor is a *looser* test than the 10% relative branch
(at |state| = 7e-4 it is 14%), so tripping it means the reconstruction really is
worse than 10% of the driven signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon.acd import ACDipoleStateConsistencyError
from tmom_recon.acd.models import ACDipoleStateSeries
from tmom_recon.acd.reconstruction import _check_bpm_state_consistency

__test__ = False

BPM = "BR3.BPM2L3"
N_TURNS = 512


def _state(amplitude: float, offset: float = 0.0) -> ACDipoleStateSeries:
    turns = np.arange(N_TURNS, dtype=float)
    x = amplitude * np.sin(2.0 * np.pi * 0.1625 * turns) + offset
    zeros = np.zeros_like(x)
    return ACDipoleStateSeries(x=x, px=zeros, y=zeros, py=zeros, t=zeros, pt=zeros)


def _frame(state: ACDipoleStateSeries, *, x_error: float = 0.0) -> pd.DataFrame:
    """Reconstructed frame for *state*, offset in ``x`` by *x_error*."""
    return pd.DataFrame(
        {
            "name": [BPM] * N_TURNS,
            "turn": np.arange(N_TURNS),
            "x": state.x + x_error,
            "px": state.px,
        }
    )


class TestPasses:
    def test_exact_agreement_passes(self) -> None:
        state = _state(7.145e-4)
        _check_bpm_state_consistency(_frame(state), BPM, state)

    def test_just_inside_the_absolute_floor_passes(self) -> None:
        """Sub-mm states get the flat 1e-4 floor, not the relative branch."""
        state = _state(7.145e-4)
        _check_bpm_state_consistency(_frame(state, x_error=0.99e-4), BPM, state)

    def test_large_state_gets_the_relative_branch(self) -> None:
        """At 6 mm the 10% branch allows far more than the 1e-4 floor would."""
        state = _state(6.0e-3)
        _check_bpm_state_consistency(_frame(state, x_error=5.0e-4), BPM, state)


class TestRejects:
    def test_raises_the_typed_error(self) -> None:
        state = _state(7.145e-4)
        with pytest.raises(ACDipoleStateConsistencyError):
            _check_bpm_state_consistency(_frame(state, x_error=1.5e-4), BPM, state)

    def test_is_still_a_value_error(self) -> None:
        """Callers that only know about ValueError must keep working."""
        state = _state(7.145e-4)
        with pytest.raises(ValueError):
            _check_bpm_state_consistency(_frame(state, x_error=1.5e-4), BPM, state)

    def test_carries_the_numbers_a_caller_needs_to_report(self) -> None:
        state = _state(7.145e-4)
        with pytest.raises(ACDipoleStateConsistencyError) as excinfo:
            _check_bpm_state_consistency(_frame(state, x_error=1.5e-4), BPM, state)
        error = excinfo.value
        assert error.bpm_name == BPM
        assert error.coord == "x"
        assert error.tolerance == pytest.approx(1e-4)
        assert error.max_residual == pytest.approx(1.5e-4, rel=1e-6)
        assert error.state_amplitude == pytest.approx(7.145e-4, rel=1e-6)
        # The ratio is what makes the failure readable; it must be recoverable
        # from the attributes without parsing the message.
        assert error.max_residual / error.state_amplitude == pytest.approx(0.21, abs=0.01)

    def test_message_reports_the_ratio_and_the_branch(self) -> None:
        state = _state(7.145e-4)
        with pytest.raises(ACDipoleStateConsistencyError) as excinfo:
            _check_bpm_state_consistency(_frame(state, x_error=1.5e-4), BPM, state)
        message = str(excinfo.value)
        assert "% of |state|" in message
        assert "below the 1 mm relative-branch threshold" in message

    def test_relative_branch_still_rejects_a_gross_error(self) -> None:
        """Widening with the signal must not become a blank cheque."""
        state = _state(6.0e-3)
        with pytest.raises(ACDipoleStateConsistencyError) as excinfo:
            _check_bpm_state_consistency(_frame(state, x_error=1.0e-3), BPM, state)
        assert excinfo.value.tolerance == pytest.approx(6.0e-4)


def test_a_dc_offset_alone_trips_the_guard() -> None:
    """A pure static shift is exactly what a mis-modelled closed orbit looks like."""
    state = _state(7.145e-4)
    with pytest.raises(ACDipoleStateConsistencyError):
        _check_bpm_state_consistency(_frame(state, x_error=3.0e-4), BPM, state)


def test_missing_bpm_is_a_plain_value_error() -> None:
    """Not a rejection of the measurement -- a caller must not skip the file for it."""
    state = _state(7.145e-4)
    frame = _frame(state).assign(name="SOMETHING.ELSE")
    with pytest.raises(ValueError) as excinfo:
        _check_bpm_state_consistency(frame, BPM, state)
    assert not isinstance(excinfo.value, ACDipoleStateConsistencyError)


class TestTheVerdictIsRecorded:
    """The guard's numbers must be available whether or not it rejects.

    A batch caller comparing preprocessing chains needs to say *how far* each
    acquisition was from passing, including the ones that passed comfortably and
    the ones that were thrown out. Only logging the failures makes the accepted
    files a blank in the table and the rejected ones unrecoverable.
    """

    def test_a_pass_returns_one_record_per_coordinate(self) -> None:
        state = _state(7e-4)
        records = _check_bpm_state_consistency(_frame(state), BPM, state)
        assert [record["coord"] for record in records] == ["x", "px"]
        assert all(record["passed"] for record in records)
        assert all(record["bpm"] == BPM for record in records)

    def test_a_record_carries_the_residual_and_the_tolerance_applied(self) -> None:
        state = _state(7e-4)
        records = _check_bpm_state_consistency(_frame(state, x_error=0.5e-4), BPM, state)
        horizontal = next(record for record in records if record["coord"] == "x")
        assert horizontal["max_residual"] == pytest.approx(0.5e-4)
        assert horizontal["tolerance"] == pytest.approx(1e-4)
        # The rms is what tells a DC offset from an oscillating disagreement:
        # for a constant error the two are equal.
        assert horizontal["rms_residual"] == pytest.approx(0.5e-4)

    def test_a_rejection_still_carries_its_records(self) -> None:
        state = _state(7e-4)
        with pytest.raises(ACDipoleStateConsistencyError) as excinfo:
            _check_bpm_state_consistency(_frame(state, x_error=1.5e-4), BPM, state)
        records = excinfo.value.records
        assert [record["coord"] for record in records] == ["x"]
        assert records[0]["passed"] is False
