Usage
=====

This page focuses on the package entry points that are intended to be called
directly from analysis code.

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
       inject_noise_xy_inplace,
   )

Input data
----------

Most reconstruction entry points expect a turn-by-turn BPM frame with columns
such as ``name``, ``turn``, ``x``, ``y``, ``var_x``, and ``var_y`` plus a
Twiss dataframe indexed by BPM element name.

As a rule:

- the measurement frame provides observed coordinates and their variances
- the Twiss frame provides lattice optics for the same BPM names
- outputs are returned as pandas dataframes so they can be merged back into
  analysis pipelines easily

Typical workflow
----------------

For most use cases the flow is:

1. prepare a turn-by-turn BPM dataframe
2. prepare or reconstruct a compatible Twiss dataframe
3. run one of the transverse, dispersive, n-BPM, or AC-dipole entry points
4. merge or post-process the returned dataframe inside the analysis code

Two-BPM transverse reconstruction
---------------------------------

.. code-block:: python

   from tmom_recon import calculate_transverse_pz

   result = calculate_transverse_pz(tracking_df, twiss_df)

This is the most direct reconstruction path and is a good default when you
want a lightweight BPM-pair-based estimate of ``px`` and ``py``.

Dispersive momentum reconstruction
----------------------------------

.. code-block:: python

   from tmom_recon import calculate_dispersive_pz

   dp_over_p = calculate_dispersive_pz(tracking_df, twiss_df)

Use this when the quantity of interest is longitudinal momentum offset
``delta p / p`` rather than transverse momentum itself.

n-BPM combination
-----------------

.. code-block:: python

   from tmom_recon import calculate_transverse_pz_nbpm

   nbpm_result = calculate_transverse_pz_nbpm(tracking_df, twiss_df)

The n-BPM workflow combines information from multiple BPMs and is useful when
the local two-BPM estimate is too noisy or when you want a more constrained
reconstruction window.

AC-dipole reconstruction
------------------------

.. code-block:: python

   from tmom_recon import calculate_ac_dipole_momentum

   acd_result = calculate_ac_dipole_momentum(
       tracking_df,
       twiss_df,
       ac_dipole_marker="MKQA.6L4.B1",
       model=acd_model,
   )

The ``model`` object is expected to provide the MAD-NG tracking interface used
by ``tmom_recon.acd.madng_driver.ACDipoleMadDriver``.

The returned dataframe includes both raw kick estimates and cleaned fit-based
quantities such as ``dpx_fit_rad`` and ``dpy_fit_rad``, together with metadata
describing the selected upstream and downstream BPMs.

If you want to apply AC-dipole-cleaned BPM momenta during higher-level
reconstruction, prefer using :class:`tmom_recon.ACDipoleConfig` through the
transverse, dispersive, or n-BPM integration paths.

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
       ),
   )

This pattern is preferable when AC-dipole-corrected BPM momenta are only an
intermediate step inside a larger reconstruction call.

Kicker-based single-turn reconstruction
---------------------------------------

The package also contains a lower-level helper for single-kick datasets:

.. code-block:: python

   from tmom_recon.kicker.core import reconstruct_momentum_kick

   kicker_result = reconstruct_momentum_kick(
       tracking_df,
       twiss_df,
       n_turns_free=1000,
       n_turns_after_kick=3,
   )

This workflow is intended for turn-by-turn data with one clear kicker
excitation. It subtracts the closed orbit, detects the kick turn, then solves
the linear transport equation from the kicker to the first downstream BPM
response using the Twiss-parameterized transfer matrix.

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
   )

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
