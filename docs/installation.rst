Installation
============

Requirements
------------

- Python 3.11 or newer
- ``numpy``, ``pandas``, ``scipy``, ``tfs-pandas``, ``omc3``

Base install
------------

.. code-block:: bash

   python -m pip install -e .

Test dependencies
-----------------

.. code-block:: bash

   python -m pip install -e '.[test]'

Development dependencies
------------------------

.. code-block:: bash

   python -m pip install -e '.[dev,test]'

Documentation dependencies
--------------------------

.. code-block:: bash

   python -m pip install -e '.[docs]'

Optional AC-dipole stack
------------------------

AC-dipole reconstruction and some slow integration tests also require the
external MAD-NG / tracking toolchain such as ``pymadng_utils`` and
``xtrack-tools``.
