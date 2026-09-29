"""Global driven-optics ACD marker states on synthetic driven optics (no MAD-NG)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon.acd.global_markers import (
    COORDS,
    GLOBAL_SUMMARY_ATTR_NAME,
    apply_global_marker_states,
    cross_mode_bases,
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
STATIC = {"x": 1e-4, "px": -2e-6, "y": -5e-5, "py": 3e-6}


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


def _tracked_result(responses, static: dict[str, float]) -> pd.DataFrame:
    """A tracked result with wrong driven lines, a static offset and a BPM row."""
    rows = []
    wrong = _state(responses[BPMS[3]])  # right tunes, wrong amplitudes and phases
    for row_type, name in (("marker_before", BEFORE), ("marker_after", AFTER)):
        frame = pd.DataFrame({"row_type": row_type, "name": name, "turn": TURNS})
        for col in COORDS:
            frame[col] = 1.3 * wrong[col] + static[col]
            frame["var_" + col] = 4.0e-12
        rows.append(frame)
    bpm = pd.DataFrame({"row_type": "upstream_bpm", "name": BPMS[0], "turn": TURNS})
    for col in (*COORDS, "var_x", "var_px", "var_y", "var_py"):
        bpm[col] = 1.0
    rows.append(bpm)
    return pd.concat(rows, ignore_index=True)


def _markers(out, row_type):
    return out[out["row_type"] == row_type]


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
    out = apply_global_marker_states(
        _tracked_result(responses, STATIC),
        _data(responses, offset=3e-3),
        responses,
        driven_tunes=TUNES,
    )
    for row_type, name in (("marker_before", BEFORE), ("marker_after", AFTER)):
        rows = _markers(out, row_type)
        truth = _state(responses[name.upper()])
        for col in COORDS:
            # The static part is the tracked one; the driven lines are exact.
            np.testing.assert_allclose(
                rows[col].to_numpy() - STATIC[col], truth[col], rtol=0, atol=1e-12
            )
            assert (rows["var_" + col] == 4.0e-12).all()
    fits = out.attrs[GLOBAL_SUMMARY_ATTR_NAME]
    for k, fit in enumerate(fits):
        assert fit.mode == k + 1
        assert fit.amplitude == pytest.approx(AMPLITUDE[k], rel=1e-10)
        assert fit.median_residual < 1e-10
        assert fit.bpms == tuple(BPMS)


def test_coupled_markers_carry_the_other_mode() -> None:
    """With coupling, x at the marker contains the mode-2 line (the per-plane blind spot)."""
    responses = _responses(0.05)
    out = apply_global_marker_states(
        _tracked_result(responses, STATIC), _data(responses), responses, driven_tunes=TUNES
    )
    rows = _markers(out, "marker_before")
    g = responses[BEFORE.upper()]
    for i, col in enumerate(COORDS):
        lines = harmonic_lines(TURNS, rows[col].to_numpy(), TUNES)[0]
        np.testing.assert_allclose(lines, AMPLITUDE * g[i], rtol=1e-9)
        assert np.all(np.abs(lines) > 0)


def test_kick_is_the_driven_alpha_jump() -> None:
    """Uncoupled: px_after - px_before = -Delta(alpha) x / beta at the marker."""
    responses = _responses()
    out = apply_global_marker_states(
        _tracked_result(responses, dict.fromkeys(STATIC, 0.0)),
        _data(responses),
        responses,
        driven_tunes=TUNES,
    )
    before, after = _markers(out, "marker_before"), _markers(out, "marker_after")
    g = responses[BEFORE.upper()]
    for k, (plane, mom) in enumerate((("x", "px"), ("y", "py"))):
        beta = abs(g[2 * k, k]) ** 2
        kick = after[mom].to_numpy() - before[mom].to_numpy()
        np.testing.assert_allclose(
            kick, -KICK_ALPHA[k] / beta * before[plane].to_numpy(), atol=1e-12
        )
        np.testing.assert_allclose(after[plane].to_numpy(), before[plane].to_numpy(), atol=1e-15)


def test_non_marker_rows_are_unchanged() -> None:
    responses = _responses()
    tracked = _tracked_result(responses, STATIC)
    out = apply_global_marker_states(tracked, _data(responses), responses, driven_tunes=TUNES)
    sel = tracked["row_type"] == "upstream_bpm"
    pd.testing.assert_frame_equal(out[sel], tracked[sel])


def test_noisy_bpm_is_down_weighted() -> None:
    responses = _responses()
    noisy = _data(responses, noise={BPMS[4]: 5e-4})
    out = apply_global_marker_states(
        _tracked_result(responses, STATIC), noisy, responses, driven_tunes=TUNES
    )
    fit = out.attrs[GLOBAL_SUMMARY_ATTR_NAME][0]
    assert abs(fit.amplitude / AMPLITUDE[0] - 1) < 1e-4


def test_unknown_names_and_bad_variances_are_skipped() -> None:
    responses = _responses()
    data = _data(responses)
    extra = data[data["name"] == BPMS[0]].assign(name="NOT_IN_MODEL")
    data.loc[data["name"] == BPMS[1], "var_x"] = np.inf
    out = apply_global_marker_states(
        _tracked_result(responses, STATIC), pd.concat([data, extra]), responses, driven_tunes=TUNES
    )
    bpms = out.attrs[GLOBAL_SUMMARY_ATTR_NAME][0].bpms
    assert BPMS[1] not in bpms
    assert "NOT_IN_MODEL" not in bpms


def test_too_few_bpms_raise() -> None:
    responses = _responses()
    data = _data(responses)
    with pytest.raises(ValueError, match="at least 2 BPMs"):
        apply_global_marker_states(
            _tracked_result(responses, STATIC),
            data[data["name"] == BPMS[0]],
            responses,
            driven_tunes=TUNES,
        )


def test_missing_marker_rows_raise() -> None:
    responses = _responses()
    tracked = _tracked_result(responses, STATIC)
    with pytest.raises(ValueError, match="marker_after"):
        apply_global_marker_states(
            tracked[tracked["row_type"] != "marker_after"],
            _data(responses),
            responses,
            driven_tunes=TUNES,
        )


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
    out = apply_global_marker_states(
        _tracked_result(responses, STATIC),
        _data(truth),
        responses,
        driven_tunes=TUNES,
        cross_bases=bases,
    )
    for row_type, name in (("marker_before", BEFORE), ("marker_after", AFTER)):
        rows = _markers(out, row_type)
        expected = _state(truth[name.upper()])
        for col in COORDS:
            np.testing.assert_allclose(
                rows[col].to_numpy() - STATIC[col], expected[col], rtol=0, atol=1e-12
            )
    for k, fit in enumerate(out.attrs[GLOBAL_SUMMARY_ATTR_NAME]):
        assert fit.amplitude == pytest.approx(AMPLITUDE[k], rel=1e-10)
        np.testing.assert_allclose(fit.cross, CROSS[k], rtol=1e-9)
    # Without the cross fit the model's (absent) coupling is used.
    plain = apply_global_marker_states(
        _tracked_result(responses, STATIC), _data(truth), responses, driven_tunes=TUNES
    )
    assert plain.attrs[GLOBAL_SUMMARY_ATTR_NAME][0].cross is None


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
