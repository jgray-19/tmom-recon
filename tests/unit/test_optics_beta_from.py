"""OpticsInput.beta_from selects the omc3 beta estimator."""

from __future__ import annotations

import pytest

from tmom_recon import optics as optics_module
from tmom_recon.optics import OpticsInput


def test_beta_from_defaults_to_amplitude() -> None:
    assert OpticsInput().beta_from == "amplitude"


def test_beta_from_rejects_unknown_estimator() -> None:
    with pytest.raises(ValueError, match="beta_from"):
        OpticsInput(beta_from="kmod")


@pytest.mark.parametrize(("beta_from", "use_amplitude"), [("amplitude", True), ("phase", False)])
def test_beta_from_reaches_the_loader(monkeypatch, tmp_path, beta_from, use_amplitude) -> None:
    seen = {}

    def fake_build(measurement_dir, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(optics_module, "build_twiss_from_measurements", fake_build)
    with pytest.raises(RuntimeError, match="stop"):
        optics_module.load_measurement(tmp_path, beta_from=beta_from)
    assert seen["use_amplitude_beta"] is use_amplitude
