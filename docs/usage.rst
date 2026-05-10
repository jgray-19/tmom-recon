Usage
=====

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

Two-BPM transverse reconstruction
---------------------------------

.. code-block:: python

   from tmom_recon import calculate_transverse_pz

   result = calculate_transverse_pz(tracking_df, twiss_df)

Dispersive momentum reconstruction
----------------------------------

.. code-block:: python

   from tmom_recon import calculate_dispersive_pz

   dp_over_p = calculate_dispersive_pz(tracking_df, twiss_df)

n-BPM combination
-----------------

.. code-block:: python

   from tmom_recon import calculate_transverse_pz_nbpm

   nbpm_result = calculate_transverse_pz_nbpm(tracking_df, twiss_df)

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

If you want to apply AC-dipole-cleaned BPM momenta during higher-level
reconstruction, prefer using :class:`tmom_recon.ACDipoleConfig` through the
transverse, dispersive, or n-BPM integration paths.

Building the docs
-----------------

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html
