"""Fast checks for the cacheable preloaded-measurement path in resolve_optics."""

from __future__ import annotations

from pathlib import Path

from tmom_recon.optics import LoadedMeasurement, load_measurement

MEASUREMENT_DIR = Path(__file__).parent / "data" / "measurements" / "psb_hio_0Hz"


def test_load_measurement_returns_loaded_measurement() -> None:
    loaded = load_measurement(MEASUREMENT_DIR)
    assert isinstance(loaded, LoadedMeasurement)
    assert not loaded.tws.empty
