Usage
=====

All reconstruction entry points accept raw BPM data plus two required orbit
inputs: ``closed_orbit_at_zero`` (measured ``x/y`` at ``dp=0``) and
``orbit_mode`` (``"dynamic"`` or ``"absolute"``). For example::

   result = calculate_pz(
       raw_bpm_data,
       ModelDetails(accelerator=accelerator, pt=pt_offset),
       closed_orbit_at_zero=measured_orbit_zero[["x", "y"]],
       orbit_mode="dynamic",
       barrier_s=None,
   )

The package generates all reference Twiss tables. Dynamic mode restores the
generated zero-momentum model ``x/px/y/py``; absolute mode restores measured
``x/y`` and generated model ``px/py``. Callers must not supply or pre-apply a
reference Twiss.

``calculate_acd_pz`` and ``calculate_kicker_pz`` use the same orbit contract.
``estimate_closed_orbit_pt`` takes ``closed_orbit_at_zero`` directly and has no
mode because subtraction is identical in both modes.

Use ``OpticsInput`` to select measured optics categories. Categories default to
the generated model; requested measurement categories never silently fall back.

``PzGenerator.build``, ``ACDipolePzGenerator.build``, and
``KickerPzGenerator.build`` freeze the measured zero orbit and mode. A strength
update refreshes the generated zero-momentum reference as well as active optics.

See :doc:`closed_orbit_handling` for migration guidance, including the required
changes in ``psb_md`` and ``sgd-magnet-tuner``.
