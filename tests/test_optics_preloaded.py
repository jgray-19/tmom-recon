"""Fast checks for the cacheable preloaded-measurement path in resolve_optics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tmom_recon.optics import LoadedMeasurement, load_measurement, resolve_optics

MEASUREMENT_DIR = Path(__file__).parent / "data" / "measurements" / "psb_hio_0Hz"


def test_load_measurement_returns_loaded_measurement() -> None:
    loaded = load_measurement(MEASUREMENT_DIR)
    assert isinstance(loaded, LoadedMeasurement)
    assert not loaded.tws.empty


def test_resolve_optics_preloaded_matches_disk() -> None:
    """resolve_optics(measured=...) must equal the disk-loading path exactly."""
    loaded = load_measurement(MEASUREMENT_DIR)

    via_disk = resolve_optics(measurement_dir=MEASUREMENT_DIR)
    via_preloaded = resolve_optics(measured=loaded)

    pd.testing.assert_frame_equal(via_disk.tws, via_preloaded.tws)
    assert via_disk.sources == via_preloaded.sources
    assert via_disk.use_dispersion == via_preloaded.use_dispersion
