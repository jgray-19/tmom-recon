Test inventory and debugging order
==================================

The suite is organised around the three supported reconstruction workflows:
all-BPM, AC-dipole, and kicker. The ``diagnostic`` contracts are the first
place to investigate a machine-level failure; focused unit and specialist
tests retain formula and regression detail.

Canonical contracts
-------------------

``tests/unit/test_public_workflow_contracts.py``
  Fast checks for the focused top-level API, explicit barrier decision, optics
  provenance, and negative/zero/positive matrix construction.

``tests/contracts/test_model_inputs.py``
  Required MAD-NG model columns and observation contract.

``tests/contracts/test_acd_location.py``
  Xtrack installation and MAD-NG marker-location agreement. Xsuite Twiss is not
  used as reconstruction optics.

``tests/contracts/test_reconstruction.py``
  Consolidated all-BPM and AC-dipole reconstruction matrix. PSB, LHC beam 1,
  and the 120 cm crossing sequence each cover negative, zero, and positive
  momentum offsets in clean and BPM-noise conditions.

``tests/contracts/test_kicker_reconstruction.py``
  PSB horizontal, vertical, and diagonal single-kick recovery at negative,
  zero, and positive momentum offsets.

Supporting tests
----------------

``tests/physics`` and ``tests/lattice``
  Synthetic formula, phase, uncertainty, neighbour, and transport contracts.

``tests/acd``
  Specialist marker transport, state-consistency, and generator lifecycle
  checks not duplicated by the consolidated matrix.

``tests/kicker``
  Kick detection and the least-squares kick solve, against an analytically
  generated single kick in a flat ring.

``tests/measurements``, ``tests/momentum``, and ``tests/filtering``
  Measurement loading, optics resolution, generator, and SVD primitives.

The removed n-BPM and Kalman workflows are not part of the supported public
surface. Their test directories are excluded from collection while historical
files are removed from the migration.

Truth and optics boundaries
---------------------------

Xtrack supplies particle coordinates and marker locations for integration
truth. MAD-NG, or an OMC3 measurement generated for an explicit measured-optics
scenario, supplies reconstruction optics. Xsuite Twiss must not become an
optics oracle.

Truth comparisons require one-to-one ``(name, turn)`` coverage. Missing or
non-finite reconstructed rows are failures and must not be hidden by an inner
merge or pre-RMSE filtering.

Rules for changing tests
------------------------

* Keep explicit machine, workflow, momentum, and condition identifiers.
* Do not replace an established numerical limit with a looser inferred limit.
* Do not turn model or setup failures into skips.
* Keep raw tracked coordinates separate from reconstruction optics.
* Add supported end-to-end behaviour to the consolidated workflow matrix;
  retain specialist files only for genuinely distinct hypotheses.
