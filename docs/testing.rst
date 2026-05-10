Testing
=======

The repository contains both fast unit tests and slower integration tests.

Fast local checks
-----------------

Run the non-slow tests with:

.. code-block:: bash

   pytest -m "not slow"

This is the quickest way to validate typical refactors and small API changes.

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
