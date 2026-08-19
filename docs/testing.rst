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
   pytest -m "not crosscode"
   pytest -m crosscode
   pytest -m "slow or regression"

``unit`` tests use synthetic data only. ``integration`` tests deliberately
construct accelerator models; ``crosscode`` tests compare Xsuite and MAD-NG;
``regression`` tests are small reproductions of previously observed failures.
PSB and LHC ownership is available through the ``psb`` and ``lhc`` markers.

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
