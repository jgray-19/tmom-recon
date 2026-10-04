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

Global Driven-Optics Markers
----------------------------

``calculate_acd_global_pz`` is a second marker method with the same inputs and
output layout as ``calculate_acd_pz``:

.. code-block:: python

   from tmom_recon import calculate_acd_global_pz

   acd_result = calculate_acd_global_pz(
       tracking_df, model_details, acd_config,
       closed_orbit_at_zero=measured_orbit_zero[["x", "y"]],
       orbit_mode="dynamic",
   )
   fits = acd_result.attrs["acd_global_markers"]  # per-plane GlobalPlaneFit

Method. Each AC dipole kicks its own momentum (``px`` at ``Q_1``, ``py`` at
``Q_2``). In steady state the linear ring answers each drive with one forced
response ``g_k(s)`` (4 components ``x, px, y, py``), so the state on turn ``n`` is
``X_n(s) = sum_k Re[C_k g_k(s) exp(2 pi i Q_k n)]`` with ONE complex kick
amplitude ``C_k`` per drive. With ``M_b`` the one-turn matrix at the AC dipole
and ``e_k`` the kicked momentum, the response just before the AC dipole is
``xi_k = (exp(2 pi i Q_k) I - M_b)^-1 M_b e_k``. It is ``xi_k + e_k`` just after,
and it is transported to every BPM from there.

All matrices come from the **undriven** model's linear normalizing map A
(MAD-NG ``twiss{coupling=true, savemap=1}``; every MAD-NG optics column is built
from A, see ``madl_gphys.mad``). The mode vectors are
``v_k = A[:, 2k] + i A[:, 2k+1]``. The tests check them against MAD-NG's ``beta12/21``,
``alfa12/21``, ``ev1/ev2`` and ``kpa`` columns. ``C_1, C_2`` are fitted by
inverse-variance weighted least squares from the ``Q_1`` and ``Q_2`` lines of x
AND y at all BPMs.

Cross-mode content from the data (``fit_cross_modes=True``). The forced
response takes each drive's cross-plane content (drive 1 in y, drive 2 in x)
from the model's coupling. A model with too little coupling (the uncoupled start
of a fit, refreshed only with the coupling found so far) then gives markers with
too little, and the fit is pulled back towards less coupling (psb_sim: fitted
``|f1001|`` about 25% low). With ``fit_cross_modes=True`` each drive's marker
state gains a free component in the other normal mode at the marker,
``P_j d_k`` with ``P_j = [Re v_j, Im v_j]`` and ``d_k`` complex 2-vector,
transported to the BPMs by the same matrices and fitted with ``C_k`` from x and
y at all BPMs (``cross_mode_bases``). ``C_k`` and the kick still follow the forced
response. On the 2 mrad tilted PSB the worst marker cross-line error drops from
100% to 20% with an uncoupled model and from 50% to 9% with half the tilts; the
own lines and the kick are unchanged.

Why not the driven twiss. MAD-NG's driven optics replace each AC dipole by a
lens acting on its own plane at all tunes. Without coupling that equals the
forced response exactly, and the uncoupled results are identical. With coupling,
mode 1 has y motion at the AC dipole and the vertical lens kicks it at ``Q_1``,
which the real vertical AC dipole never does. In the tilted PSB test, the lens
model leaves 1% (own line) and 20% (cross line) errors at the markers; the
forced response leaves 9e-5 and 6e-4.

What is replaced. Only the driven lines (``Q_1`` and ``Q_2``) of the ``marker_before`` /
``marker_after`` rows. The static part (the data-driven off-momentum orbit),
the variances, the BPM rows and ``attrs["summary"]`` stay those of the tracked
reconstruction, which runs first.

When to use it. The tracked kick is a difference of two tracked momenta, so a
local model error between the two BPMs next to the AC dipole enters it at full
size. The global markers take the data only through ``C``, a ring average.
PSB contract (``tests/contracts/test_acd_global_markers.py``), tune-matched
nominal model, kick drive-line error x / y:

================  ===============  ===============
quad rms          global           tracked
================  ===============  ===============
0, dp/p = 1e-3    0.18% / 0.004%   0.015% / 0.003%
1e-3              0.46% / 1.2%     3.2% / 2.6%
3e-3              1.6% / 3.8%      16% / 13%
1e-2              4.8% / 13%       31% / 59%
================  ===============  ===============

Coupled PSB (2 mrad rms quad rolls, 15-20% cross-plane lines), exact model,
worst marker line error:

=============  ===========  ===========
method         own line     cross line
=============  ===========  ===========
global         9.2e-5       6.4e-4
tracked        8.2e-4       6.5e-2
=============  ===========  ===========

Requirements and limits.

- **The model must be on the machine's natural tunes** (match the tune knobs to
  the measured tunes in ``ModelDetails.machine_state``). The driven optics depend on
  ``Q_nat - Q_drive`` (the resonance denominator of ``xi_k``); in the PSB test (offset 0.01) an unmatched 2e-3 tune miss
  alone makes the global kick 20% wrong, worse than tracked.
- ``turn`` must increment at the sequence start, where ``mu_d`` is referenced.
- The cross-plane lines are only as good as the model's coupling: with an
  uncoupled model they are zero.
- Other users of the driven (lens) optics (the tracked ACD path,
  ``calculate_pz(acd=...)``) still carry the lens model's coupling error.
- With an exact model the tracked markers are more accurate; the global method
  pays off once the local model is uncertain.

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

Global Markers
--------------

.. automodule:: tmom_recon.acd.global_markers
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
