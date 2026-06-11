from __future__ import annotations

import tmom_recon
import tmom_recon.nbpm as nbpm


def test_top_level_nbpm_api_exports_only_reconstruction_entry_point() -> None:
    assert "calculate_transverse_pz_nbpm" in tmom_recon.__all__
    assert "combine_momentum_blue" not in tmom_recon.__all__
    assert tmom_recon.calculate_transverse_pz_nbpm is nbpm.calculate_transverse_pz_nbpm


def test_nbpm_module_keeps_a_small_public_surface() -> None:
    assert nbpm.__all__ == ["calculate_transverse_pz_nbpm"]
    assert hasattr(nbpm, "calculate_transverse_pz_nbpm")
    assert not hasattr(nbpm, "combine_momentum_blue")
