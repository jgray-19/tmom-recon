Installation
============

This page covers the Python package itself. Some reconstruction modes, notably
the AC-dipole and MAD-NG-backed integration paths, also depend on external
tooling that is intentionally not installed by the base package extra.

Requirements
------------

- Python 3.11 or newer
- ``numpy``, ``pandas``, ``scipy``, ``tfs-pandas``, ``omc3``

Base install
------------

Use this when you only need the library and its default runtime dependencies.

.. code-block:: bash

   python -m pip install -e .

Test dependencies
-----------------

This installs pytest and coverage-related tooling used by the local test suite.

.. code-block:: bash

   python -m pip install -e '.[test]'

Development dependencies
------------------------

This is the usual working environment for editing code in the repository.

.. code-block:: bash

   python -m pip install -e '.[dev,test]'

Documentation dependencies
--------------------------

Install this extra when building the Sphinx site locally.

.. code-block:: bash

   python -m pip install -e '.[docs]'

Optional AC-dipole stack
------------------------

AC-dipole reconstruction and some slow integration tests also require the
external MAD-NG / tracking toolchain such as ``pymadng_utils`` and
``xtrack-tools``.

In practice this means:

- the base package can be installed and documented without the full tracking stack
- some autodoc pages mock those imports during documentation builds
- slow tests and MAD-NG-backed workflows need those external packages available

Local docs build
----------------

Once the docs extra is installed, build the HTML site with:

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

The generated site is written to ``docs/_build/html``.
