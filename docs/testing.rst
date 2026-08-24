Testing
=======

The repository contains both fast unit tests and slower integration tests.

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

An absolute reconstruction consumes a measured reference orbit: BPM positions
are ``x``/``y`` while ``px``/``py`` may come from an externally fitted magnetic
model. ``tmom-recon`` intentionally does not perform that fit. The external-
strength contracts verify that supplied mappings and their derived reference
angles are accepted by fresh reconstruction models. The committed mappings are
deterministic seeded interface fixtures with generator, sequence, and dependency
provenance; they are not presented as optimiser-fit results.

The reserved ``campaign`` marker is for long tracking plus OMC3 measurement
generation. Run such tests in the active ``accpy`` environment when the project
virtual environment cannot resolve the local accelerator packages.

Diagnostic contracts
--------------------

The ``diagnostic`` tests are the canonical debugging ladder. Each collected
case has one machine condition, one plane where applicable, and one asserted
metric; a failure is therefore a statement about one stage rather than a broad
end-to-end comparison. Run them with ``pytest -m diagnostic``.
See :doc:`test_inventory` for the retained specialist and legacy-test roles.

========================  =====================================  ======================================
Order                     Contract family                        A failing case identifies
========================  =====================================  ======================================
1                         ``physics``, ``lattice`` unit tests     Formula, phase/transport, or API invariant
2                         ``contracts/test_model_inputs.py``      MAD-NG model loading, observation, or chromatic Twiss
3                         ``contracts/test_transverse.py``        Reference removal or plain neighbour momentum recovery
4                         ``contracts/test_dispersion.py``        Measured-optics pt estimate or dispersive restoration
5                         ``contracts/test_noise.py``             BPM-noise propagation or SVD cleaning
6                         ``acd/test_*transport*`` and guards     AC-dipole transport inversion or state consistency
7                         ACD generator tests                     Cached-model refresh or generator equivalence
========================  =====================================  ======================================

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

Test layout
-----------

New tests are organized under ``tests/unit``, ``tests/integration`` and
``tests/regression``. Shared construction and assertions live under
``tests/support``. The PSB dynamic-part characterization remains under
``tests/acd`` as a slow, explicitly marked integration suite. Excluding
``slow`` tests from local checks does not exclude it from the full test run.

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
