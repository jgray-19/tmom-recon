Usage
=====

Main entry points for analysis code.

Top-level API
-------------

The package re-exports the main entry points:

.. code-block:: python

   from tmom_recon import (
       ACDipoleConfig,
       build_twiss_from_measurements,
       calculate_ac_dipole_momentum,
       calculate_dispersive_pz,
       calculate_pz_measurement,
       calculate_transverse_pz,
       calculate_transverse_pz_nbpm,
       inject_noise_xy,
   )

Input data
----------

Most reconstruction entry points expect a turn-by-turn BPM frame with columns
such as ``name``, ``turn``, ``x``, ``y``, ``var_x``, and ``var_y`` plus a
Twiss dataframe indexed by BPM element name.

Inputs and outputs:

- the measurement frame provides observed coordinates and their variances
- the Twiss frame provides lattice optics for the same BPM names
- outputs are pandas dataframes

Typical workflow
----------------

Typical flow:

1. prepare a turn-by-turn BPM dataframe
2. prepare or reconstruct a compatible Twiss dataframe
3. run one of the transverse, dispersive, n-BPM, or AC-dipole entry points
4. merge or post-process the returned dataframe inside the analysis code

Two-BPM transverse reconstruction
---------------------------------

.. code-block:: python

   from tmom_recon import calculate_transverse_pz

   result = calculate_transverse_pz(tracking_df, twiss_df)

Reconstructs ``px`` and ``py`` from a BPM pair.

Dispersive momentum reconstruction
----------------------------------

.. code-block:: python

   from tmom_recon import calculate_dispersive_pz

   dp_over_p = calculate_dispersive_pz(tracking_df, twiss_df)

Returns the longitudinal momentum offset ``delta p / p``.

n-BPM combination
-----------------

.. code-block:: python

   from tmom_recon import calculate_transverse_pz_nbpm

   nbpm_result = calculate_transverse_pz_nbpm(tracking_df, twiss_df)

Combines information from multiple BPMs over a reconstruction window.

AC-dipole reconstruction
------------------------

.. code-block:: python

   from tmom_recon import calculate_ac_dipole_momentum

   acd_result = calculate_ac_dipole_momentum(
       tracking_df,
       twiss_df,
       ac_dipole_marker="MKQA.6L4.B1",
       model=acd_model,
       dpx_tune=0.27,
       dpy_tune=0.322,
   )

The ``model`` must provide the MAD-NG tracking interface used by
``tmom_recon.acd.madng_driver.ACDipoleMadDriver``. The result contains raw and
fitted kick estimates, cleaned states, and selected-BPM metadata.

The reconstruction models the AC dipole as a thin kick at the marker:

- same-turn ``x`` and ``y`` at the marker are shared across the kick,
- the kick appears as a jump in ``px`` and ``py``, and
- the cleaned marker-side states are transported back to the adjacent BPMs to
  produce cleaned BPM-local momenta.

Use :class:`tmom_recon.ACDipoleConfig` to apply cleaned momenta in higher-level
reconstruction.

Using ``ACDipoleConfig`` in higher-level reconstruction
-------------------------------------------------------

.. code-block:: python

   from tmom_recon import ACDipoleConfig, calculate_transverse_pz

   result = calculate_transverse_pz(
       tracking_df,
       twiss_df,
       ac_dipole_config=ACDipoleConfig(
       ac_dipole_marker="MKQA.6L4.B1",
       model=acd_model,
       dpx_tune=0.27,
       dpy_tune=0.322,
       ),
   )

This applies AC-dipole correction as an intermediate reconstruction step.

Kicker-based single-turn reconstruction
---------------------------------------

For single-kick datasets:

.. code-block:: python

   from tmom_recon.kicker.core import reconstruct_momentum_kick

   kicker_result = reconstruct_momentum_kick(
       tracking_df,
       twiss_df,
       n_turns_free=1000,
       n_turns_after_kick=3,
   )

Subtracts the closed orbit, detects the kick turn, and solves linear transport
from the kicker to the first downstream BPM.

Measurement-driven Twiss reconstruction
---------------------------------------

The package also exposes utilities for building Twiss-like inputs from
measurement data:

.. code-block:: python

   from tmom_recon import calculate_pz_measurement

   pz_measurement = calculate_pz_measurement(
       tracking_df,
       measurement_folder,
       model_twiss,
       reverse_meas_tws=False,
       use_model_optics=True,
       use_measurement_dispersion=False,
   )

This lets you use measured phase advances together with model beta/alpha, which
is useful for machines where the phase measurement is trusted more than the
full measured optics table. If you want measured phase plus measured
dispersion, keep ``use_measurement_dispersion=True``.

To build a Twiss-like dataframe directly from optics measurement files:

.. code-block:: python

   from pathlib import Path
   from tmom_recon import build_twiss_from_measurements

   measurement_twiss, dispersion_found = build_twiss_from_measurements(
       Path(measurement_folder),
       include_errors=False,
   )

Building the docs
-----------------

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html
