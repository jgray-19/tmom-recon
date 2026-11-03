"""Global driven-optics ACD marker states on synthetic driven optics (no MAD-NG)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon.acd.global_markers import (
    COORDS,
    cross_mode_bases,
    fit_global_marker_lines,
    fit_mode_amplitude,
    harmonic_lines,
)

TUNES = (0.2515, 0.4435)
TURNS = np.arange(1000)
BPMS = [f"BPM{i}" for i in range(12)]
BEFORE, AFTER = "HACMAP_before", "HACMAP_after"
AMPLITUDE = np.array([2e-3 * np.exp(0.7j), 1.5e-3 * np.exp(-2.1j)])
# The ACD lens changes alpha only: the marker ends share beta and phase.
KICK_ALPHA = (0.08, -0.05)


def _responses(coupling: float = 0.0) -> dict[str, np.ndarray]:
    """Uncoupled Courant-Snyder responses, plus random cross-plane terms of relative size ``coupling``."""
    rng = np.random.default_rng(3)
    out = {}
    names = [*BPMS, BEFORE.upper()]
    for name, mu in zip(
        names, np.sort(rng.uniform(0.0, 4.0, (len(names), 2)), axis=0), strict=True
    ):
        g = np.zeros((4, 2), complex)
        for k in range(2):
            beta, alfa = rng.uniform(3.0, 12.0), rng.uniform(-1.5, 1.5)
            phase = np.exp(2j * np.pi * mu[k])
            own = slice(2 * k, 2 * k + 2)
            g[own, k] = np.array([np.sqrt(beta), -(alfa - 1j) / np.sqrt(beta)]) * phase
            other = slice(2 - 2 * k, 4 - 2 * k)
            cross = rng.normal(size=2) + 1j * rng.normal(size=2)
            g[other, k] = coupling * cross * np.abs(g[own, k])
        out[name] = g
    # After the lens: same positions, momenta jump by -Delta(alpha) x / beta.
    after = out[BEFORE.upper()].copy()
    for k, plane in enumerate((0, 2)):
        beta = abs(after[plane, k]) ** 2
        after[plane + 1, :] -= KICK_ALPHA[k] / beta * after[plane, :]
    out[AFTER.upper()] = after
    return out


def _state(g: np.ndarray) -> dict[str, np.ndarray]:
    """Exact driven state (both modes) of response ``g`` on every turn."""
    rotation = np.exp(2j * np.pi * np.outer(TURNS, TUNES))
    return {col: np.real(rotation @ (AMPLITUDE * g[i])) for i, col in enumerate(COORDS)}


def _data(responses, *, offset: float = 0.0, noise: dict[str, float] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frames = []
    for name in BPMS:
        state = _state(responses[name])
        sigma = (noise or {}).get(name, 0.0)
        frames.append(
            pd.DataFrame(
                {
                    "name": name,
                    "turn": TURNS,
                    "x": state["x"] + offset + sigma * rng.standard_normal(TURNS.size),
                    "y": state["y"] - offset + sigma * rng.standard_normal(TURNS.size),
                    "var_x": max(sigma, 1e-6) ** 2,
                    "var_y": max(sigma, 1e-6) ** 2,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _fit(data, responses, **kwargs):
    return fit_global_marker_lines(
        data, responses, markers=(BEFORE, AFTER), driven_tunes=TUNES, **kwargs
    )


def test_harmonic_lines_recover_both_lines_and_offset() -> None:
    z = np.array([1e-3 * np.exp(1.1j), 2e-5 * np.exp(-0.4j)])
    values = np.real(np.exp(2j * np.pi * np.outer(TURNS, TUNES)) @ z) + 4e-4
    lines, offset = harmonic_lines(TURNS, values, TUNES)
    np.testing.assert_allclose(lines, z, atol=1e-15)
    assert offset == pytest.approx(4e-4, abs=1e-15)


def test_fit_mode_amplitude_is_weighted() -> None:
    model = np.array([1.0, 1.0j, -1.0])
    lines = 2.0 * model
    lines[2] = 5.0  # outlier with zero weight
    assert fit_mode_amplitude(lines, model, np.array([1.0, 1.0, 0.0])) == pytest.approx(2.0)


@pytest.mark.parametrize("coupling", [0.0, 0.05], ids=["uncoupled", "coupled"])
def test_markers_equal_the_exact_driven_states(coupling) -> None:
    responses = _responses(coupling)
    out = _fit(_data(responses, offset=3e-3), responses)
    for name in (BEFORE, AFTER):
        state = out.driven_state(name, TURNS)
        truth = _state(responses[name.upper()])
        for i, col in enumerate(COORDS):
            np.testing.assert_allclose(state[:, i], truth[col], rtol=0, atol=1e-12)
    for k, fit in enumerate(out.fits):
        assert fit.mode == k + 1
        assert fit.amplitude == pytest.approx(AMPLITUDE[k], rel=1e-10)
        assert fit.median_residual < 1e-10
        assert fit.bpms == tuple(BPMS)


def test_coupled_markers_carry_the_other_mode() -> None:
    """With coupling, x at the marker contains the mode-2 line (the per-plane blind spot)."""
    responses = _responses(0.05)
    out = _fit(_data(responses), responses)
    lines = out.lines[BEFORE.upper()]
    np.testing.assert_allclose(lines, AMPLITUDE * responses[BEFORE.upper()], rtol=1e-9)
    assert np.all(np.abs(lines) > 0)


def test_kick_is_the_driven_alpha_jump() -> None:
    """Uncoupled: px_after - px_before = -Delta(alpha) x / beta at the marker."""
    responses = _responses()
    out = _fit(_data(responses), responses)
    before, after = out.driven_state(BEFORE, TURNS), out.driven_state(AFTER, TURNS)
    g = responses[BEFORE.upper()]
    for k in range(2):
        beta = abs(g[2 * k, k]) ** 2
        kick = after[:, 2 * k + 1] - before[:, 2 * k + 1]
        np.testing.assert_allclose(kick, -KICK_ALPHA[k] / beta * before[:, 2 * k], atol=1e-12)
        np.testing.assert_allclose(after[:, 2 * k], before[:, 2 * k], atol=1e-15)


def test_noisy_bpm_is_down_weighted() -> None:
    responses = _responses()
    fit = _fit(_data(responses, noise={BPMS[4]: 5e-4}), responses).fits[0]
    assert abs(fit.amplitude / AMPLITUDE[0] - 1) < 1e-4


def test_unknown_names_and_bad_variances_are_skipped() -> None:
    responses = _responses()
    data = _data(responses)
    extra = data[data["name"] == BPMS[0]].assign(name="NOT_IN_MODEL")
    data.loc[data["name"] == BPMS[1], "var_x"] = np.inf
    bpms = _fit(pd.concat([data, extra]), responses).fits[0].bpms
    assert BPMS[1] not in bpms
    assert "NOT_IN_MODEL" not in bpms


def test_too_few_bpms_raise() -> None:
    responses = _responses()
    data = _data(responses)
    with pytest.raises(ValueError, match="at least 2 BPMs"):
        _fit(data[data["name"] == BPMS[0]], responses)


CROSS = np.array([[3e-4 + 1e-4j, -2e-4j], [1.5e-4, 2e-4 - 1e-4j]])  # d_k, one row per drive


def _cross_bases() -> dict[str, np.ndarray]:
    """Arbitrary other-mode bases, equal at both marker ends (continuous across the AC dipole)."""
    rng = np.random.default_rng(7)
    out = {name: rng.normal(size=(4, 2, 2)) for name in BPMS}
    out[BEFORE.upper()] = out[AFTER.upper()] = rng.normal(size=(4, 2, 2))
    return out


def test_cross_fit_recovers_the_other_mode_missing_from_the_model() -> None:
    """Uncoupled model, coupled data: the fitted cross part at the markers is the data's."""
    responses, bases = _responses(), _cross_bases()
    # The machine: model forced response plus other-mode content B d at every element.
    truth = {
        name: responses[name] + np.einsum("iak,ka->ik", bases[name], CROSS) / AMPLITUDE
        for name in bases
    }
    out = _fit(_data(truth), responses, cross_bases=bases)
    for name in (BEFORE, AFTER):
        state = out.driven_state(name, TURNS)
        expected = _state(truth[name.upper()])
        for i, col in enumerate(COORDS):
            np.testing.assert_allclose(state[:, i], expected[col], rtol=0, atol=1e-12)
    for k, fit in enumerate(out.fits):
        assert fit.amplitude == pytest.approx(AMPLITUDE[k], rel=1e-10)
        np.testing.assert_allclose(fit.cross, CROSS[k], rtol=1e-9)
    # Without the cross fit the model's (absent) coupling is used.
    assert _fit(_data(truth), responses).fits[0].cross is None


def test_cross_mode_bases_carry_the_other_mode_from_the_marker() -> None:
    rng = np.random.default_rng(5)
    names = ["BPM0", BEFORE.upper(), AFTER.upper(), "BPM1"]
    a = {name: rng.normal(size=(4, 4)) for name in names}
    a[AFTER.upper()] = a[BEFORE.upper()]
    vectors = {name: m[:, 0::2] + 1j * m[:, 1::2] for name, m in a.items()}
    bases = cross_mode_bases(vectors, before=BEFORE, after=AFTER)
    a_b = a[BEFORE.upper()]
    other = np.stack([a_b[:, 2:4], a_b[:, 0:2]], axis=-1)  # drive 1 -> mode 2, drive 2 -> mode 1
    for marker in (BEFORE, AFTER):
        np.testing.assert_allclose(bases[marker.upper()], other, atol=1e-12)
    for name in ("BPM0", "BPM1"):
        transfer = a[name] @ np.linalg.inv(a_b)
        np.testing.assert_allclose(
            bases[name], np.einsum("ij,jak->iak", transfer, other), atol=1e-10
        )
