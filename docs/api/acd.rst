tmom_recon.acd
==============

Overview
--------

Reconstructs a driven transverse kick at a marker from BPM data on both sides
of the element:

- for a given turn, the marker position ``x``/``y`` is shared before and after
  the kick,
- the kick appears as a jump in ``px``/``py``, and
- the fitted marker-side state can be transported back to the adjacent BPMs to
  obtain cleaned local momenta.

Workflow:

1. select one upstream and one downstream BPM around the marker,
2. reconstruct local BPM states from the turn-by-turn orbit data,
3. track those states to the marker with MAD-NG,
4. fit harmonic ``dpx`` and ``dpy`` waveforms at the marker, and
5. transport cleaned pre-/post-kick states back to the selected BPMs.

Quick Start
-----------

Direct reconstruction through the supported ACD API:

.. code-block:: python

   from tmom_recon import ACDipoleConfig, ModelDetails, calculate_acd_pz

   acd_result = calculate_acd_pz(
       tracking_df,
       ModelDetails(accelerator=accelerator, pt=pt_offset),
       ACDipoleConfig(
           ac_dipole_marker="MKQA.6L4.B1",
           driven_tunes=(0.27, 0.322),
       ),
       closed_orbit_at_zero=measured_orbit_zero[["x", "y"]],
       orbit_mode="dynamic",
   )

The measured setting-zero positions are always the subtraction origin. The
mode selects the generated/model state restored after reconstruction.

Key Outputs
-----------

The direct ACD result includes:

- raw marker-side kick estimates ``dpx_rad`` and ``dpy_rad``,
- fitted harmonic waveforms ``dpx_fit_rad`` and ``dpy_fit_rad``,
- cleaned marker-side states such as ``x_acd_upstream_cleaned`` and
  ``px_acd_downstream_cleaned``, and
- cleaned BPM-side momenta such as ``px_bpm_upstream_cleaned`` and
  ``py_bpm_downstream_cleaned``.

Same-turn marker ``x`` and ``y`` match on both sides of the kick; fitted
``dpx``/``dpy`` describe the momentum jump.

Public API
----------

.. automodule:: tmom_recon.acd
   :members:
   :show-inheritance:
   :no-index:

Integration Helpers
-------------------

.. automodule:: tmom_recon.acd.integration
   :members:
   :show-inheritance:
   :no-index:

Core Reconstruction
-------------------

.. automodule:: tmom_recon.acd.reconstruction
   :members:
   :show-inheritance:
   :no-index:

Selection
---------

.. automodule:: tmom_recon.acd.selection
   :members:
   :show-inheritance:
   :no-index:

Cleaning
--------

.. automodule:: tmom_recon.acd.cleaning
   :members:
   :show-inheritance:
   :no-index:

Data Models
-----------

.. automodule:: tmom_recon.acd.models
   :members:
   :show-inheritance:
   :no-index:

MAD-NG Driver
-------------

.. automodule:: tmom_recon.acd.madng_driver
   :members:
   :show-inheritance:
   :no-index:
