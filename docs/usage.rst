Usage
=====

``calculate_pz`` is the public momentum-reconstruction entry point. It accepts
raw turn-by-turn BPM data, generated model details, and an explicit measured
orbit-zero frame.

Input data
----------

The BPM frame contains ``name``, ``turn``, ``x``, ``y`` and normally
``var_x``/``var_y``. Names must be covered by the frame and the resolved optics.

Constructing the frame
----------------------

Dynamic reconstruction removes the same measured orbit zero in both planes::

   from tmom_recon import ReconstructionFrame

   frame = ReconstructionFrame(
       orbit_zero=measured_orbit_zero[["x", "y"]],
       dynamic_planes=("x", "y"),
   )

Absolute reconstruction retains both planes and therefore requires fitted
closed-orbit angles::

   frame = ReconstructionFrame(
       orbit_zero=measured_orbit_zero[["x", "y"]],
       fitted_momenta=fitted_orbit[["px", "py"]],
   )

A horizontal-only retained frame is explicit too::

   frame = ReconstructionFrame(
       orbit_zero=measured_orbit_zero[["x", "y"]],
       dynamic_planes=("y",),
       fitted_momenta=fitted_orbit[["px"]],
   )

Reconstruction
--------------

Pass raw data. Do not subtract the orbit in application code::

   from tmom_recon import ModelDetails, calculate_pz

   result = calculate_pz(
       raw_bpm_data,
       ModelDetails(accelerator=accelerator, pt=pt_offset),
       frame=frame,
       measurement_pt_offset=pt_offset,
       use_dispersion=True,
       barrier_s=None,
   )

Omit ``measurement_pt_offset`` to estimate it after the frame transformation.
The result records the applied/estimated value in ``attrs["PT_EST"]``.

Optics sources
--------------

``measurement_dir`` supplies measured optics. Categories listed in
``model_optics`` are forced to the model; other categories use measurements when
available. Dispersion position and momentum columns are resolved as one category.

AC-dipole reconstruction
------------------------

Supply ``ACDipoleConfig`` to refine the bracketing BPMs, or set
``acd_only=True`` for marker/BPM states only. The same ``frame`` is mandatory on
both paths, so all-BPM and ACD reconstruction cannot use different coordinate
origins.

Generators
----------

``generator=True`` freezes raw data and its frame. Generator updates accept
``measurement_pt_offset`` and updated strengths; they cannot change the orbit
origin underneath an optimization.

Closed-orbit details
--------------------

See :doc:`closed_orbit_handling` for the complete ordering, dispersion, fitted
orbit, and failure-mode contract.

Building the docs
-----------------

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html
