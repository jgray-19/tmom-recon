# AC Dipole Reconstruction Process

This note documents the AC-dipole reconstruction flow used in `tmom_recon`,
with special attention to the assumption that, for a given turn, the marker
coordinates `x` and `y` at the AC dipole should be the same whether they are
estimated from the upstream BPM side or the downstream BPM side.

It is written as a detailed process note for future maintenance and auditing.

## Purpose

The AC-dipole workflow exists to do two related things:

1. Reconstruct the effective AC-dipole kick waveforms `dpx(turn)` and
   `dpy(turn)` at a chosen lattice marker.
2. Use that cleaned marker-side state to improve the nearby BPM momentum
   estimates.

The key physical model is that the AC dipole acts like an instantaneous
transverse kick at the marker:

- `x` does not jump across the marker
- `y` does not jump across the marker
- `px` may jump
- `py` may jump

Equivalently, for the same turn:

- `x_pre(turn) = x_post(turn)`
- `y_pre(turn) = y_post(turn)`
- `px_post(turn) = px_pre(turn) + dpx(turn)`
- `py_post(turn) = py_pre(turn) + dpy(turn)`

That is the assumption you pointed out: if we track from either BPM to the ACD,
the same-turn marker position should agree in `x` and `y`, up to measurement
and reconstruction noise. Because of that, averaging the upstream- and
downstream-derived marker positions is useful and physically well motivated.

## High-Level Pipeline

The implementation lives primarily in:

- `src/tmom_recon/acd/reconstruction.py`
- `src/tmom_recon/acd/cleaning.py`
- `src/tmom_recon/acd/madng_driver.py`

The full flow is:

1. Select BPMs on both sides of the AC dipole.
2. Build raw local BPM state estimates `(x, px, y, py)` using the two-BPM
   formulas.
3. Track those BPM states to the AC-dipole marker with MAD-NG.
4. Combine the tracked marker `x` and `y` values into one common same-turn
   ACD position.
5. Fit harmonic kick waveforms for `dpx` and `dpy`.
6. Solve for smoothed pre-kick momenta `px_pre` and `py_pre`.
7. Reconstruct post-kick momenta from the fitted kick:
   `px_post = px_pre + dpx_fit`, `py_post = py_pre + dpy_fit`.
8. Transport the cleaned marker states back to the selected upstream and
   downstream BPMs.
9. Publish both marker-side and BPM-side cleaned quantities.

## Step 1: BPM Selection

The reconstruction first finds BPMs surrounding the chosen AC-dipole marker.
This is handled by the selection helpers in `src/tmom_recon/acd/selection.py`.

Conceptually:

- one primary upstream BPM is chosen
- one primary downstream BPM is chosen
- optionally a wider window of upstream/downstream BPMs can also contribute to
  the marker-side cleaning

The selected names are stored in the result metadata so downstream code knows
which BPMs were used.

## Step 2: Raw Local BPM State Reconstruction

For each selected BPM, the code reconstructs a local state estimate:

- `x`
- `px`
- `y`
- `py`

This is done in `src/tmom_recon/acd/reconstruction.py` via:

- `_prepare_prev_reconstruction(...)` for upstream BPMs
- `_prepare_next_reconstruction(...)` for downstream BPMs

These functions use the standard two-BPM momentum formulas. At this stage the
estimates are still local BPM-side quantities and still noisy.

Important detail:

- the raw BPM estimates are reconstructed after closed-orbit subtraction
- if the model carries a nonzero `deltap`, that estimate is propagated into the
  local reconstruction formulas

## Step 3: Track BPM States to the AC Dipole

Each local BPM state is then transported to the AC-dipole marker with the
MAD-NG tracking driver:

- `_transport_to_marker(...)`
- `_build_tracked_state_table(...)`
- `ACDipoleMadDriver.track_particles(...)`

This produces marker-side state tables of the form:

- `turn`
- `x`
- `px`
- `y`
- `py`
- `var_x`
- `var_px`
- `var_y`
- `var_py`

There are separate tracked tables for the upstream and downstream sides.

At this point we generally have, for the same turn:

- one marker estimate inferred from upstream data
- one marker estimate inferred from downstream data

Because the local BPM reconstructions are noisy, those two marker estimates do
not agree perfectly.

## Step 4: Enforce One Common Same-Turn Marker Position in `x` and `y`

This is the key part relevant to your question.

The cleaner now explicitly uses the physical statement:

- the AC dipole gives a momentum kick, not a position jump

Therefore the same-turn marker coordinates should be shared between the
upstream-derived and downstream-derived states.

In code this is now made explicit by:

- `_combine_marker_transverse_positions(...)` in
  `src/tmom_recon/acd/cleaning.py`

That helper takes all upstream and downstream tracked marker estimates and
computes inverse-variance-weighted averages for:

- `x_common(turn)`
- `y_common(turn)`

with corresponding variances:

- `var_x_common(turn)`
- `var_y_common(turn)`

So if the tracked marker estimates are:

- upstream: `x_u(turn), y_u(turn)`
- downstream: `x_d(turn), y_d(turn)`

then the cleaner produces one shared marker trajectory:

- `x_common(turn) = weighted_average(x_u, x_d, ...)`
- `y_common(turn) = weighted_average(y_u, y_d, ...)`

This shared trajectory is then reused for both the pre-kick and post-kick ACD
states.

That means the cleaner is not merely hoping the two sides agree; it is
explicitly imposing the model that they refer to the same marker position and
using averaging to reduce noise.

## Step 5: Fit the Harmonic Kick Waveforms

Once the marker positions are handled, the code fits the kick waveforms for the
momentum jump:

- `dpx_fit(turn)`
- `dpy_fit(turn)`

This happens in `src/tmom_recon/acd/cleaning.py` through:

- `_refine_known_kick_fit(...)`

The workflow is:

1. Start from user-supplied tune hints `dpx_tune` and `dpy_tune`.
2. Build a harmonic model with sine, cosine, and offset terms.
3. Refine the tune locally around the hint.
4. Return the best-fit harmonic waveform.

The fit is done on the difference between downstream and upstream marker-side
momenta, because that difference is the kick itself:

- `dpx_raw(turn) = px_downstream_marker(turn) - px_upstream_marker(turn)`
- `dpy_raw(turn) = py_downstream_marker(turn) - py_upstream_marker(turn)`

These raw differences are noisy, so the harmonic fit provides a cleaned version
that is more physically constrained.

## Step 6: Solve for the Pre-Kick Marker Momenta

After fitting the kick waveform, the code solves for the pre-kick marker
momenta:

- `px_pre(turn)`
- `py_pre(turn)`

using:

- `_solve_smoothed_pre_momentum_with_known_kick(...)`

The solver combines:

- the upstream marker momentum estimate directly
- the downstream marker momentum estimate mapped back to pre-kick space by
  subtracting the fitted kick
- an optional smoothness regularization across turns

Conceptually, for `px`, the downstream estimate is first converted into a
pre-kick estimate:

- `px_down_as_pre(turn) = px_downstream_marker(turn) - dpx_fit(turn)`

Then the solver finds a smooth `px_pre(turn)` that best agrees with both the
upstream pre-kick estimate and the mapped downstream estimate.

The same is done for `py`.

## Step 7: Build the Cleaned Pre- and Post-Kick ACD States

The final cleaned marker states are:

- pre-kick:
  - `x = x_common`
  - `y = y_common`
  - `px = px_pre`
  - `py = py_pre`

- post-kick:
  - `x = x_common`
  - `y = y_common`
  - `px = px_post`
  - `py = py_post`

with:

- `px_post = px_pre + dpx_fit`
- `py_post = py_pre + dpy_fit`

This is implemented in `_clean_ac_dipole_states(...)`.

The important structural point is:

- `x_common` is used in both the pre-kick and post-kick states
- `y_common` is used in both the pre-kick and post-kick states

So the no-position-jump assumption is imposed by construction.

## Step 8: Transport the Cleaned Marker State Back to the BPMs

This is the place where the previous audit found a weakness and where the code
has now been made more consistent.

### Previous behavior

Previously the cleaned BPM momenta were not reconstructed by transporting the
cleaned marker state back to the BPMs. Instead, the code applied a heuristic
"shrinkage" correction to the raw BPM momentum estimates based on the
marker-side correction.

That approach had two problems:

1. It was not the exact inverse of the marker transport model.
2. It did not guarantee that the cleaned BPM momenta were dynamically
   consistent with the cleaned marker state.

### Current behavior

The current code now explicitly transports the cleaned marker states back to the
selected BPMs:

- `_transport_marker_state_to_bpm(...)`
- `_reconstruct_bpm_momentum_from_cleaned_acd(...)`

The directions are:

- cleaned upstream marker state back to upstream BPM with `direction=-1`
- cleaned downstream marker state back to downstream BPM with `direction=1`

This means the cleaned BPM `px` and `py` are now derived from the same cleaned
marker states that satisfied:

- shared `x(turn)`
- shared `y(turn)`
- fitted `dpx(turn)`
- fitted `dpy(turn)`

So the assumption is now enforced all the way through the BPM override path,
not only inside the marker-side cleaner.

## Why `observe=0` Is Important for the MAD-NG Tracking Step

The MAD-NG driver now uses `observe=0` in the low-level `track { ... }` call
inside `ACDipoleMadDriver.track_particles(...)`.

This is intentional.

If `observe=1` is used, the returned table contains intermediate observations at
all observed elements. For backward tracking or wrapped ranges, recovering the
true endpoint by sorting on `s` can be ambiguous and may select the wrong row.

Using `observe=0` avoids that ambiguity:

- the track result corresponds to the endpoint of the requested range
- the wrapper no longer needs to infer the endpoint from a set of intermediate
  observations
- the shortest-path direction check becomes much more robust

This is especially relevant for the case you highlighted:

- when the ACD is behind the source BPM in lattice order, the correct path is
  to track backward with `dir=-1`

The regression coverage for this now lives in:

- `tests/acd/test_ac_dipole_momentum.py`

and there is also a standalone verification script:

- `experimental/verify_acd_shortest_path.py`

## Step 9: Published Outputs

The result dataframe includes:

- raw BPM momenta around the ACD
- raw marker-side states
- raw kick estimates
- fitted kick waveforms
- cleaned marker-side states
- cleaned BPM momenta

Relevant columns include:

- `px_bpm_upstream`
- `py_bpm_upstream`
- `px_bpm_downstream`
- `py_bpm_downstream`
- `x_acd_upstream`
- `y_acd_upstream`
- `x_acd_downstream`
- `y_acd_downstream`
- `dpx_rad`
- `dpy_rad`
- `dpx_fit_rad`
- `dpy_fit_rad`
- `x_acd_upstream_cleaned`
- `y_acd_upstream_cleaned`
- `x_acd_downstream_cleaned`
- `y_acd_downstream_cleaned`
- `px_acd_upstream_cleaned`
- `py_acd_upstream_cleaned`
- `px_acd_downstream_cleaned`
- `py_acd_downstream_cleaned`
- `px_bpm_upstream_cleaned`
- `py_bpm_upstream_cleaned`
- `px_bpm_downstream_cleaned`
- `py_bpm_downstream_cleaned`

Because the cleaner uses one common same-turn marker position, the cleaned
marker columns should satisfy:

- `x_acd_upstream_cleaned == x_acd_downstream_cleaned`
- `y_acd_upstream_cleaned == y_acd_downstream_cleaned`

up to numerical precision.

## Why Averaging `x` and `y` at the Marker Is Useful

This averaging is useful for both physical and statistical reasons.

### Physical reason

The AC dipole is modeled as a thin kick:

- it changes momentum
- it does not instantaneously displace the particle in position

So there should only be one position at the marker per turn.

### Statistical reason

The upstream-derived and downstream-derived marker positions contain independent
or partially independent reconstruction noise. Averaging them reduces that
noise and gives a better estimate of the true marker position.

### Modeling reason

If we did not enforce a common marker `x` and `y`, then the reconstructed
pre-kick and post-kick states would imply an unphysical position discontinuity
at the marker. That would be inconsistent with the thin-kick picture.

## Current Tests That Cover This

The test coverage now includes:

1. A direct unit test that `_clean_ac_dipole_states(...)` averages conflicting
   upstream/downstream marker `x` and `y` estimates into one common same-turn
   marker trajectory.
2. A unit test that the cleaned marker state is back-transported to the BPMs,
   rather than only being applied as a heuristic correction.
3. Integration checks that the cleaned marker `x` and `y` from the upstream and
   downstream sides are equal in the final reconstruction output.
4. Integration checks that cleaned BPM momenta improve over the raw BPM
   momenta on noisy data.

## Practical Interpretation

If you are thinking about the algorithm physically, the cleanest mental model
is:

1. Each side gives a noisy view of the same ACD marker state.
2. The position part of that state should be shared:
   `x_same_turn`, `y_same_turn`.
3. The momentum part differs across the kick by `dpx(turn)` and `dpy(turn)`.
4. Once the marker state is cleaned, transport it back out to the BPMs.

That is now the intended and implemented model.

## Limitations and Future Refinements

There are still some limitations worth remembering:

1. The transport is delegated to MAD-NG tracking, so the quality of the BPM
   reconstruction depends on the quality of the machine model used by the
   driver.
2. The `x`/`y` common-state combination is currently an inverse-variance
   weighted average; it does not yet include a more structured dynamical model
   for marker positions across turns.
3. The post-kick variances are currently copied from the pre-kick variances
   because the kick fit is treated as known at that stage.
4. The method assumes the thin-kick picture is appropriate for the AC dipole
   marker and surrounding transport.

Potential future improvements could include:

- propagating uncertainty from the kick-fit parameters into the post-kick
  covariance
- using a joint state-space model for marker `x`, `px`, `y`, `py`
- validating whether additional BPMs on each side improve the marker position
  average in difficult optics conditions

## Summary

Yes: the reconstruction should use the assumption that, for the same turn,
tracking from either side to the AC-dipole marker should give the same `x` and
`y`, because the AC dipole contributes a momentum kick rather than a position
jump.

The current implementation does this by:

1. transporting upstream and downstream BPM state estimates to the marker
2. averaging the marker `x` and `y` estimates into one common same-turn ACD
   position
3. fitting `dpx` and `dpy`
4. building cleaned pre- and post-kick marker states with shared `x` and `y`
5. transporting those cleaned marker states back to the BPMs

That makes the BPM-side cleaned momenta consistent with the marker-side
thin-kick model end-to-end.
