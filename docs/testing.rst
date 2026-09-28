Testing
=======

The repository contains both fast unit tests and slower integration tests.

End-to-end matrix
------------------

The supported reconstruction contracts use explicit scenario cells: PSB, LHC
beam 1, and the crossing sequence run negative, zero, and positive momentum
offsets for all-BPM and AC-dipole reconstruction. The PSB kicker contract uses
the same momentum sweep for horizontal, vertical, and diagonal kicks. The
consolidated reconstruction matrix covers clean coordinates and BPM noise. A
setup failure is a failing cell, never a skip or looser numerical threshold.

Fast local checks
-----------------

Run the non-slow tests with:

.. code-block:: bash

   pytest -m "not slow"

This is the quickest way to validate typical refactors and small API changes.

The suite also labels ownership and cost explicitly:

.. code-block:: bash

   pytest -m "unit or (integration and not slow)"
   pytest -m "slow or regression"

``unit`` tests use synthetic data only. ``integration`` tests deliberately
construct accelerator models; ``regression`` tests are small reproductions of
previously observed failures.
PSB and LHC ownership is available through the ``psb`` and ``lhc`` markers.

Optics and reference conventions
--------------------------------

Xsuite provides simulated turn-by-turn coordinates and therefore the momentum
truth in integration tests. It is never an optics truth source. Reconstruction
optics come from MAD-NG, or from an OMC3 measurement generated from long
tracking where a measured-optics scenario is under test.

A reconstruction consumes a measured setting-zero orbit: BPM positions are
``x``/``y`` while optional ``px``/``py`` may come from an externally fitted
magnetic model. ``tmom-recon`` intentionally does not perform that fit.

The reserved ``campaign`` marker is for long tracking plus OMC3 measurement
generation. Run such tests in the active ``accpy`` environment when the project
virtual environment cannot resolve the local accelerator packages.

Diagnostic contracts
--------------------

The ``diagnostic`` tests are the canonical debugging ladder. Each collected
case has one machine condition, one plane where applicable, and one asserted
metric; a failure is therefore a statement about one stage rather than a broad
end-to-end comparison. Run them with ``pytest -m diagnostic``.
See :doc:`test_inventory` for the specialist-test roles.

Investigate failures in this order:

1. ``physics`` and ``lattice`` unit tests for formula and transport invariants;
2. ``contracts/test_model_inputs.py`` for MAD-NG model construction;
3. ``contracts/test_reconstruction.py`` for all-BPM and AC-dipole workflows;
4. ``contracts/test_kicker_reconstruction.py`` for single-kick reconstruction;
5. specialist ACD transport, state-consistency, and generator checks.

The machine-level contracts are parametrised over PSB ring 3, LHC beam 1, and
the 120 cm crossing sequence whenever the underlying physical feature exists.
Xsuite is used only to generate tracked coordinate truth and report the
installed ACD location; the reconstruction model and fake OMC3 measurement are
always MAD-NG derived.

For all-BPM data containing an AC dipole, every ``calculate_pz`` call must
state the marker's MAD-NG longitudinal position through ``barrier_s``. A caller
with no localised kick must state ``barrier_s=None`` explicitly. The location
contract first verifies that Xsuite and MAD-NG place the marker at the same
``s``; the neighbour reconstruction then never transports a BPM pair through
that kick.

Noise budgets and SVD cleaning
------------------------------

The ``bpm_noise`` cells add 1e-5 m Gaussian position noise and reconstruct it
**uncleaned**. Their limits are the propagated noise, not a round number: a
neighbour pair turns a position error into an angle error of order
``sigma_x / beta``, so the same noise costs PSB (beta ~ 5 m) an order of
magnitude more than the LHC (beta ~ 100-8000 m). The measured worst cases are
3.4e-6 / 3.9e-6 rad for PSB and 2.6e-7 / 2.9e-7 rad for the two LHC optics, and
the limits carry 25% headroom on those.

``svd_clean_measurements`` is *optional preprocessing*, not part of the
reconstruction: nothing in ``calculate_pz``, ``calculate_acd_pz`` or
``calculate_kicker_pz`` calls it. On this data a Gavish-Donoho rank truncation
reduces the PSB noisy error 2.6x (3.4e-6 to 1.3e-6 rad) by discarding noise
modes; it removes no bias, and the clean cells are unaffected by it. Budgets are
therefore set on uncleaned data, so a genuine reconstruction error cannot hide
behind the filter.

The AC-dipole cells have their own noisy floor for the kick fit
(``_KICK_R2_MIN``). The kick is a *difference* of two BPM momenta, so its
accuracy is bounded by the propagated noise: over the 100 turns the LHC
scenarios track, a four-parameter harmonic fit leaves ~7e-8 rad on a ~1.5e-6 rad
kick, i.e. R^2 = 0.9979. PSB stays above 0.9995 because its kick is larger and
its fit averages 1000 turns. Reusing the clean floor there asked for an accuracy
the data cannot carry.

``inject_noise_xy`` writes ``var_x``/``var_y`` alongside the noise it adds. A
tracking pipeline leaves a placeholder variance there (1e-4**2 in
``xtrack-tools``), and leaving it in place made the reconstruction weight BPMs
and report an uncertainty that had nothing to do with the noise present.

Test layout
-----------

New tests are organized under ``tests/unit``, ``tests/integration`` and
``tests/regression``. Shared construction and assertions live under
``tests/support``. Specialist ACD checks remain under ``tests/acd``. Excluding
``slow`` tests from local checks does not exclude them from the full test run.

Full test suite
---------------

Run the full suite with:

.. code-block:: bash

   pytest

The slow tests exercise larger tracking-backed workflows and may require a more
complete local environment.

Optional external dependencies
------------------------------

Some tests rely on packages outside the base runtime dependency set, especially
for AC-dipole and MAD-NG-backed workflows. If those dependencies are missing,
those parts of the suite may be skipped or fail during setup depending on the
test path.

Linting and docs
----------------

Run Ruff:

.. code-block:: bash

   ruff check .

Build the docs:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

GitHub Actions
--------------

The repository includes CI workflows for:

- coverage and test execution
- documentation deployment to GitHub Pages

The Pages workflow builds the Sphinx site from ``docs/`` and publishes the
generated HTML artifact.
