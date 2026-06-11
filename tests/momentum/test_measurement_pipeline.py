from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tfs

from tmom_recon.optics import resolve_optics


def _write_measurement_file(
    path: Path, data: dict[str, list[object]], headers: dict[str, float]
) -> None:
    frame = tfs.TfsDataFrame(pd.DataFrame(data), headers=headers)
    tfs.write(path, frame, save_index=False)


def _build_measurement_folder(measurement_dir: Path) -> None:
    names = ["BPM1", "BPM2", "BPM3"]
    names2 = ["BPM2", "BPM3", "BPM1"]
    s = [0.0, 10.0, 20.0]
    headers = {"Q1": 0.7, "Q2": 0.8}

    _write_measurement_file(
        measurement_dir / "beta_amplitude_x.tfs",
        {
            "NAME": names,
            "S": s,
            "BETX": [11.0, 12.0, 13.0],
            "ERRBETX": [0.1, 0.1, 0.1],
            "ALFX": [1.1, 1.2, 1.3],
            "ERRALFX": [0.01, 0.01, 0.01],
        },
        headers,
    )
    _write_measurement_file(
        measurement_dir / "beta_amplitude_y.tfs",
        {
            "NAME": names,
            "S": s,
            "BETY": [21.0, 22.0, 23.0],
            "ERRBETY": [0.1, 0.1, 0.1],
            "ALFY": [2.1, 2.2, 2.3],
            "ERRALFY": [0.01, 0.01, 0.01],
        },
        headers,
    )
    _write_measurement_file(
        measurement_dir / "phase_x.tfs",
        {
            "NAME": names,
            "NAME2": names2,
            "PHASEX": [0.2, 0.3, 0.2],
            "ERRPHASEX": [0.01, 0.01, 0.01],
        },
        headers,
    )
    _write_measurement_file(
        measurement_dir / "phase_y.tfs",
        {
            "NAME": names,
            "NAME2": names2,
            "PHASEY": [0.15, 0.25, 0.2],
            "ERRPHASEY": [0.01, 0.01, 0.01],
        },
        headers,
    )
    _write_measurement_file(
        measurement_dir / "orbit_x.tfs",
        {"NAME": names, "S": s, "X": [0.0, 0.0, 0.0], "ERRX": [1e-4, 1e-4, 1e-4]},
        headers,
    )
    _write_measurement_file(
        measurement_dir / "orbit_y.tfs",
        {"NAME": names, "S": s, "Y": [0.0, 0.0, 0.0], "ERRY": [1e-4, 1e-4, 1e-4]},
        headers,
    )
    _write_measurement_file(
        measurement_dir / "dispersion_x.tfs",
        {
            "NAME": names,
            "S": s,
            "DX": [1.0, 2.0, 3.0],
            "DPX": [0.1, 0.2, 0.3],
            "ERRDX": [0.01, 0.01, 0.01],
            "ERRDPX": [0.001, 0.001, 0.001],
        },
        headers,
    )
    _write_measurement_file(
        measurement_dir / "dispersion_y.tfs",
        {
            "NAME": names,
            "S": s,
            "DY": [0.4, 0.5, 0.6],
            "DPY": [0.04, 0.05, 0.06],
            "ERRDY": [0.01, 0.01, 0.01],
            "ERRDPY": [0.001, 0.001, 0.001],
        },
        headers,
    )


def _build_model_twiss() -> tfs.TfsDataFrame:
    return tfs.TfsDataFrame(
        pd.DataFrame(
            {
                "beta11": [101.0, 102.0, 103.0],
                "beta22": [201.0, 202.0, 203.0],
                "alfa11": [11.0, 12.0, 13.0],
                "alfa22": [21.0, 22.0, 23.0],
                "dx": [10.0, 20.0, 30.0],
                "dy": [40.0, 50.0, 60.0],
                "dpx": [1.0, 2.0, 3.0],
                "dpy": [4.0, 5.0, 6.0],
            },
            index=["BPM1", "BPM2", "BPM3"],
        ),
        headers={"q1": 0.7, "q2": 0.8},
    )


def test_resolve_optics_uses_measured_phase_with_model_optics(tmp_path: Path) -> None:
    _build_measurement_folder(tmp_path)
    model_tws = _build_model_twiss()

    # Force amplitude and dispersion from the model; phase (and tunes) stays measured.
    resolved = resolve_optics(
        model_tws=model_tws,
        measurement_dir=tmp_path,
        model_optics=["amplitude", "dispersion"],
        bpm_names=["BPM1", "BPM2", "BPM3"],
    )
    tws = resolved.tws

    assert resolved.sources == {
        "phase": "measurement",
        "amplitude": "model",
        "dispersion": "model",
    }
    assert resolved.use_dispersion
    assert np.allclose(tws["mu1"].to_numpy(), [0.0, 0.2, 0.5])
    assert np.allclose(tws["mu2"].to_numpy(), [0.0, 0.15, 0.4])
    assert np.allclose(tws["beta11"].to_numpy(), [101.0, 102.0, 103.0])
    assert np.allclose(tws["beta22"].to_numpy(), [201.0, 202.0, 203.0])
    assert np.allclose(tws["dx"].to_numpy(), [10.0, 20.0, 30.0])
    assert np.allclose(tws["dpx"].to_numpy(), [1.0, 2.0, 3.0])


def test_resolve_optics_keeps_measured_dispersion_with_model_amplitude(tmp_path: Path) -> None:
    _build_measurement_folder(tmp_path)
    model_tws = _build_model_twiss()

    # Only amplitude from the model; phase and dispersion stay measured.
    resolved = resolve_optics(
        model_tws=model_tws,
        measurement_dir=tmp_path,
        model_optics=["amplitude"],
        bpm_names=["BPM1", "BPM2", "BPM3"],
    )
    tws = resolved.tws

    assert resolved.sources["amplitude"] == "model"
    assert resolved.sources["dispersion"] == "measurement"
    assert resolved.use_dispersion
    assert np.allclose(tws["beta11"].to_numpy(), [101.0, 102.0, 103.0])
    assert np.allclose(tws["dx"].to_numpy(), [1.0, 2.0, 3.0])
    assert np.allclose(tws["dpx"].to_numpy(), [0.1, 0.2, 0.3])


def test_resolve_optics_respects_reverse_phase_accumulation(tmp_path: Path) -> None:
    _build_measurement_folder(tmp_path)

    # No model twiss: every category comes from the measurement, phases reversed.
    resolved = resolve_optics(
        measurement_dir=tmp_path,
        reverse_meas_tws=True,
        bpm_names=["BPM1", "BPM2", "BPM3"],
    )
    tws = resolved.tws

    assert resolved.sources == {
        "phase": "measurement",
        "amplitude": "measurement",
        "dispersion": "measurement",
    }
    assert list(tws.index) == ["BPM3", "BPM2", "BPM1"]
    assert np.allclose(tws["mu1"].to_numpy(), [0.0, 0.3, 0.5])
    assert np.allclose(tws["mu2"].to_numpy(), [0.0, 0.25, 0.4])
