tmom-recon documentation
========================

Momentum reconstruction utilities for turn-by-turn BPM data.

``tmom_recon`` bundles several related workflows around BPM-based beam
reconstruction:

- two-BPM transverse and dispersive reconstruction
- n-BPM combination
- AC-dipole reconstruction and BPM override helpers
- kicker-based single-turn reconstruction
- lattice and measurement utilities
- Kalman-based reconstruction helpers

The documentation is split into:

- :doc:`installation` for environment setup and optional dependencies
- :doc:`usage` for the main reconstruction entry points and expected inputs
- :doc:`closed_orbit_handling` for the orbit-frame and momentum convention
- :doc:`testing` for local validation and CI-oriented commands
- :doc:`test_inventory` for the diagnostic test ladder and legacy-test roles
- :doc:`api/index` for the generated module reference

Most users only need the top-level functions re-exported from
``tmom_recon``. The lower-level modules are documented as well, but many of
them are implementation-oriented building blocks rather than stable public API.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   closed_orbit_handling
   testing
   test_inventory
   api/index
