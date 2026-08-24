Test inventory and debugging order
==================================

The test suite accumulated several overlapping end-to-end tests. They remain
valuable as specialist regressions, but the ``diagnostic`` marker is the first
place to debug a failure. Each diagnostic case has one machine condition, one
plane where applicable, and one asserted outcome.

``physics`` / ``lattice``
  Formulae, phase convention, transport, and API: canonical fast foundations.

``contracts/model_inputs``
  MAD-NG model columns and observation: canonical model-input gate.

``contracts/acd_location``
  Xtrack installation/MAD-NG marker ``s`` agreement: canonical ACD-anchor gate;
  no Xsuite Twiss values participate.

``contracts/transverse``, ``dispersion``, ``noise``, and ``external_strengths``
  All-BPM reconstruction, dispersive recovery, noise/SVD behaviour, and the
  optimiser hand-off: canonical machine-level contracts.

``acd/test_*`` and ``momentum/test_*``
  Specialist ACD scenarios and historical full-pipeline regressions: retain
  while they exercise unique behaviour.

``nbpm``, ``kicker``, ``measurements``, and ``filtering``
  Independent reconstruction APIs and their supporting primitives.

Machine coverage
----------------

The canonical model, location, transverse, dispersion, noise, and
external-strength contracts are parametrised over PSB ring 3, LHC beam 1, and
the 120 cm crossing sequence whenever their physical inputs are available.
The ACD specialist files retain deeper marker-state assertions that are not
duplicated by an all-BPM contract.

The canonical ladder currently collects 72 diagnostic cases (the full suite
collects 227 nodes). The complete pre-contract inventory is stored as 180 JSON
Lines rows in
``legacy_test_audit.jsonl``. Each row is keyed by the exact node collected at
commit ``a2ee8ed`` and records its inputs, truth source, assertions, proposed
replacement, evidence state, and retain/delete decision. Unit tests enforce
the schema and deletion rules.

Canonical diagnostic map
------------------------

Run one row at a time while investigating a failure. The parametrised id gives
the exact machine and, where relevant, the plane. These tests deliberately do
not combine unrelated physical hypotheses in one assertion.

``pytest tests/physics tests/lattice -m unit``
  Synthetic only. A failure identifies a formula, phase, matrix, barrier, or
  API invariant.

``pytest tests/contracts/test_model_inputs.py``
  PSB, LHCb1, and crossing. A required MAD-NG reconstruction input is absent.

``pytest tests/contracts/test_acd_location.py``
  PSB, LHCb1, and crossing. The installed Xtrack ACD is not at the MAD-NG
  barrier.

``pytest tests/contracts/test_transverse.py``
  Three machines and ``px``/``py``. Plain all-BPM/reference/neighbour momentum
  recovery is incorrect.

``pytest tests/contracts/test_dispersion.py``
  Three off-momentum machines. ``pt`` estimation or known-``pt`` dispersive
  recovery is incorrect.

``pytest tests/contracts/test_noise.py``
  Three machines and ``px``/``py``. Noise injection or SVD-cleaning behaviour
  is incorrect.

``pytest tests/contracts/test_external_strengths.py``
  Three machines and ``px``/``py``. The external strength/reference-angle
  hand-off is broken.

``pytest tests/contracts/test_acd_reconstruction.py``
  Three machines and ``px``/``py``. Local ACD state transport on an adjacent
  BPM is incorrect.

The default command for the whole ladder is ``pytest -m diagnostic``. Use the
single-file commands above after it identifies a family. The unit row is kept
outside ``diagnostic`` because it is the general, fast foundation for every
feature rather than an accelerator-model contract.

Legacy and specialist coverage
------------------------------

The following groups are intentionally retained, but are not the first
debugging signal. They either cover a condition not yet represented in the
canonical ladder or preserve a historical regression while its replacement is
being compared.

The standalone LHC and PSB ACD momentum modules, standalone LHC and PSB
dispersion modules, and standalone transverse-momentum module have been
removed. Their 35 baseline nodes are represented by the shared contracts;
the audit records exact same-input, two-plane, finite-row, missing-row, and
equal-limit evidence for each deleted node.

``tests/acd/test_psb_transport_backtracking.py``
  Ring-leg inverse transport and Jacobian closure. Keep; it has no all-BPM
  replacement.

``tests/acd/test_psb_dynamic_part_acd.py`` and ``tests/acd/test_acd_generator.py``
  Dynamic/static-part physics and cached-generator lifecycle. Keep as
  specialist ACD coverage.

``tests/integration/psb/test_external_fitted_strengths.py``
  PSB external-strength hand-off with its original fixture. Keep until its
  comparison to the canonical hand-off test is recorded.

``tests/regression/psb/test_offmomentum_reference.py``
  Physical-BPM versus BPMT pseudo-monitor classification. Keep because the
  diagnostic dispersion contract deliberately consumes only physical BPMs.

``tests/nbpm`` and ``tests/kicker``
  Alternative multi-BPM and kicker reconstruction APIs. Keep as independent
  features.

No legacy test is removed merely because it resembles a contract. Removal
requires the same machine condition, truth source, both momentum planes where
applicable, and an equal-or-stronger existing numerical check.

AC-dipole Twiss roles
---------------------

The ordinary reconstruction path resolves one MAD-NG model Twiss. The ACD path
has three deliberately different inputs: driven off-momentum optics for BPM
pair reconstruction, undriven off-momentum Twiss for state restoration, and
undriven nominal Twiss for the dispersion/reference expansion. The driven
model cannot be used as the transport model because installing the ACD changes
the local element used as the transport anchor. Conversely, the nominal Twiss
is not a replacement for the off-momentum state. This separation is internal
to ``ResolvedACDipoleConfig``; callers provide model details and an
``ACDipoleConfig``, not individual Twiss tables.

``simulated_reference_from_tracking_positions_and_model_angles`` is a
test-only fixture for the optimiser boundary. It uses tracked nominal-orbit
positions as the synthetic measured reference and a fresh fitted-strength
MAD-NG model only for reference angles. It is intentionally not a production
closed-orbit source.

Rules for changing tests
------------------------

* Do not replace an existing limit with a looser one. A new canonical test
  reuses the established limit for that physical scenario.
* Do not filter non-finite or missing reconstructed BPM/turn rows before a
  truth comparison. Such rows are a reconstruction failure.
* Xtrack supplies tracked coordinates and marker positions only. MAD-NG (or a
  generated measurement from MAD-NG) supplies reconstruction optics. Xsuite
  Twiss must not be used as an optics oracle, even in a compatibility test; the
  former Xsuite-versus-MAD-NG agreement nodes are recorded as deleted in the
  audit for this reason.
* Keep a legacy test until the canonical replacement has identical or stronger
  coverage and its failure signature has been compared on the same inputs.
