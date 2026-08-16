# Off-momentum momentum reconstruction in a realistic PSB measurement

PSB ring 3 · 2026-08-11 · branch `investigate/offmom-co-2am`

This note supersedes the decision sections of
`NOTES_offmomentum_closed_orbit.md`. It records the evidence behind the adopted
PSB reference frame and the resulting implementation across `tmom-recon`,
`sgd-magnet-tuner`, and `psb_md`.

## Decision

PSB reconstruction now uses a mandatory mixed momentum reference:

| reconstructed quantity | reference source |
|---|---|
| `x`, `y` | measured closed orbit from AC-dipole-off blank data |
| `px`, `py` | closed orbit of the fitted lattice |

The lattice is fitted jointly to closed orbit and BPM-to-BPM phase at multiple
momenta. The fitted knobs are main-bend strength, quadrupole gradient, and
vertical quadrupole offset. The regularisation prior defaults to `1e-2`.

The production requirements are deliberate:

1. At least two distinct momenta are required.
2. Every fitted momentum must have matching blank and driven acquisitions.
3. Closed-orbit positions come only from blank acquisitions. A driven turn mean
   is not a closed-orbit measurement.
4. Phase is equation-compensated at every momentum, **including 0dpp**. The
   nominal-momentum data do not take an uncompensated special path.
5. The same nominal-RF blank reference is used for all on- and off-momentum
   reconstructions and for every live or per-epoch generator.
6. Bend and quadrupole-`dy` values are frozen after the reference fit. Later
   dynamic-optics stages may change only their intended quadrupole and
   sextupole knobs.

The fitted lattice does not replace the measured position reference. It supplies
the angles that BPMs cannot measure.

## Why the reference must be split

A BPM reports

```text
x_measured,i(turn) = x_true,i(turn) + offset_i
```

Subtracting a blank measurement from driven data at the same BPM cancels
`offset_i` exactly. Subtracting a fitted model orbit removes the physical orbit
but leaves the offset as a false betatron displacement. PSB BPM offsets are of
the same order as the driven displacement, so this is a leading effect.

Fitting the offsets as though they were orbit errors is worse: the fit can
absorb a BPM offset into an upstream dipole error and convert an observable
position bias into an unobservable angle bias. The only robust combination is
therefore measured `x/y` with fitted-model `px/py`.

This mixed pair is a reference frame, not a claim that its four coordinates form
one exact model orbit. Reconstruction and tracking optimisation use deviations
around it. Absolute reconstructed states retain measured positions and fitted
angles; optimisation compares their dynamic deviations.

## Evidence

### 1. Position-only orbit fitting does not determine the angle

The tracking machine carried 0.08% RMS bend-field errors (seed 7) and 0.1% RMS
quadrupole-gradient errors (seed 11). The nominal model knew neither. Closed
orbits were evaluated at `dp/p = 0, ±1e-3, ±3e-3, ±8e-3`.

The corrected PSB ring-3 BPM selection contains 16 BPMs. An early study used a
bare `BR3.BPM` prefix and accidentally included `BR3.BPMT3L1`; all final
conclusions use `PSB.BPM_PATTERN_TEMPLATE`, which excludes it.

With 16 horizontal observations and 32 individual bend unknowns, the orbit
response has rank 16. A position-only fit can thread the measured BPM positions
while leaving bend combinations in a null space. Those combinations are
invisible in position at the BPMs but not in angle.

At `dp/p = 8e-3`:

| model | max position error | max angle error |
|---|---:|---:|
| nominal | `6.141e-03 m` | `2.566e-03 rad` |
| alternating bend+quad SVD, 16 BPMs | `5.87e-07 m` | `1.337e-04 rad` |

The position improves by four orders of magnitude while the angle stops near
`1e-4 rad`. Better agreement in BPM position is therefore not evidence of a
usable momentum reference.

Multiple momenta do add information, but mainly by exposing the dispersion
error from unknown quadrupole gradients. A polynomial decomposition of the
orbit residual showed:

| component | maximum contribution at `dp/p = 8e-3` |
|---|---:|
| constant bend-error orbit | `5.6e-03 m` |
| linear dispersion error | `6.31e-04 m` |
| quadratic dispersion error | `8.17e-05 m` |

After the bend fit, measured linear and quadratic terms reduce the BPM position
error to about `2e-6 m`, but this empirical table still contains no `px` or
`py`. A fitted lattice is required for the angle.

### 2. The reconstruction failure is an angle offset

The full AC-dipole pipeline was run at `dp/p = 8e-3`. Absolute BPM momenta were
compared with tracked truth; kick-fit quality alone was not used as the metric.

| model | CO position error | `px` upstream R2 | `px` downstream R2 |
|---|---:|---:|---:|
| nominal | `6.14e-03 m` | `0.9695` | `-1.6482` |
| bend+quad SVD | `1.71e-06 m` | `0.8225` | `0.8427` |
| true errors, first-order dispersion | `0` | `0.9997` | `0.9907` |
| true errors, exact CO | `0` | `1.0000` | `1.0000` |

For the SVD-fitted models, an affine decomposition gives unit gain and a nearly
constant `px` offset of `-3.8e-05 rad`, against a driven amplitude of
`9.15e-05 rad`. Removing the affine offset restores `R2 = 1.0`. Supplying the
true bends while retaining the fitted quadrupoles gives `R2 = 0.9998`.

The plateau is thus a residual closed-orbit angle error, not a failure of the
AC-dipole kick fit or the linear transport shape. A high kick-fit R2 can coexist
with incorrect absolute momenta.

### 3. Joint orbit-plus-phase fitting resolves the useful lattice

`sgd-magnet-tuner`'s `ClosedTwissFitter` was used instead of the hand-written
alternating SVD. It fits several momenta simultaneously with exact TPSA knob
derivatives and a joint regularised Gauss-Newton solve.

The important compatibility rule is:

| observables | permitted useful knob families |
|---|---|
| orbit only | bends |
| orbit plus phase | bends and quadrupole gradients |

An orbit-only fit must not expose quadrupole-gradient knobs: a centred orbit is
nearly blind to them, so they add unconstrained noise-absorbing freedom. The
production API rejects this configuration.

Regularisation is decisive. In the synthetic full-observable study:

| prior strength | fitted/true knob error | `px` error at `dp/p = 0` |
|---:|---:|---:|
| `0` | `5.75` | `1.75e-03 rad` |
| `1e-3` | `1.015` | `2.54e-04 rad` |
| `1e-2` | `0.607` | `6.32e-05 rad` |
| `1e-1` | `0.738` | `5.94e-05 rad` |

The useful region is broad, so `1e-2` is a stable default rather than a sharply
tuned value. The value is recorded with the fit diagnostics and may be
overridden explicitly.

The production observables are `x`, `y`, `mu1`, and `mu2`. Phase is taken from
the driven measurements after equation compensation. This applies equally to
negative dpp, positive dpp, and **0dpp**; using uncompensated nominal phase would
place one momentum in a different measurement frame and bias the joint fit.

### 4. First-, second-, and exact closed-orbit treatments

MAD-NG's chromatic columns use `pt`, not `delta`, and already include the
second-order Taylor factor:

```text
x(pt)  = x(0)  + pt*dx  + pt^2*ddx
px(pt) = px(0) + pt*dpx + pt^2*ddpx
```

There is no additional factor of `1/2`, and no conversion by `1/beta`.

Against an exact MAD-NG orbit at `dp/p = 8e-3`:

| lattice | quantity | first-order residual | second-order residual |
|---|---|---:|---:|
| nominal | `x` | `3.16e-05 m` | `3.84e-07 m` |
| nominal | `px` | `9.14e-06 rad` | `8.63e-08 rad` |
| with known bend+quad errors | `x` | `1.23e-04 m` | `1.32e-05 m` |
| with known bend+quad errors | `px` | `4.25e-05 rad` | `4.99e-06 rad` |

Second order is now automatic when `chrom=True` columns are available and is
backward compatible when they are absent. In the non-ACD `calculate_pz` sweep at
`dp/p = 1e-2`, it reduced relative `pt` bias from `2.04e-3` to `1.15e-5` and
`px` RMSE from `5.90e-6` to `1.67e-6 rad`.

Exact MAD-NG closed orbit remains opt-in. It matters once the model is already
accurate: in the matched-error AC-dipole case it improves downstream `px` R2
from `0.9907` to `0.999999`. It does not repair a fitted-angle error of
`1e-4 rad`; model error dominates the dispersion truncation by roughly two
orders of magnitude there.

The two direct off-momentum PSB ACD tests now request exact closed orbit
explicitly. This preserves the distinction: second-order dispersion is
automatic when available, while exact closed orbit is a deliberate choice.

## Production workflow

The reference stage runs before reconstruction:

```text
selected campaign momenta
        |
        +-- blank AC-dipole-off frames --> measured x/y closed orbits
        |
        +-- driven frames
                --> equation-compensated BPM phase, including 0dpp
        |
        v
joint regularised fit: bend dk0l + quad dk1l + quad dy
        |
        +-- fitted model orbit --> px/py reference
        +-- measured nominal blank --> x/y reference
        v
canonical mixed reference --> reconstruction --> dynamic deviations for tuning
```

Coverage is validated before expensive processing. Missing blanks, driven
phase, measured positions, fitted angles, or BPM rows are actionable errors.
There is no production fallback to a driven-data mean, a zero orbit, or model
positions.

The canonical artifacts are:

- `momentum_reference.parquet`: BPM-indexed mixed position/angle reference;
- `momentum_reference.json`: fitted strengths, uncertainties, model orbit,
  reference `pt`, observables, prior, momentum points, BPM coverage, source
  files, convergence diagnostics, and residual summaries.

Both the reference content and fitted-strength map enter cache identity. A
reconstruction cannot be reused under a different reference frame or lattice.

The fitted strength map crosses repository boundaries as plain data through
`ModelDetails.magnet_strengths`. The round-trip test verifies that `.dk0l` and
`.dk1l` values produce the same closed orbit through
`aba_optimiser.momentum_reference.closed_orbit_at` and
`tmom_recon.resolve_model_details`.

## Implementation map

### `sgd-magnet-tuner`

- `aba_optimiser.momentum_reference` is the production fitting API.
- Momentum keys are explicitly MAD-NG `pt`.
- At least two distinct momenta are mandatory.
- Defaults are `("x", "y", "mu1", "mu2")` and `prior_strength=1e-2`.
- Tune knobs, correctors, and baseline strengths are applied consistently during
  fitting and reference evaluation.
- MAD interfaces are closed deterministically.
- `MomentumReference` contains only serialisable fit and reference data.

### `psb_md`

- The main ACD workflow builds blank-only closed orbits at every fitted
  momentum and compensated phase at every driven momentum.
- `_compensated_phase_dir` intentionally has no 0dpp exception;
  `test_zero_dpp_phase_is_equation_compensated` pins this behaviour.
- The fitted reference is mandatory and is passed to every `calculate_pz` and
  `estimate_pt_from_model` call, including live, plotting, and diagnostic paths.
- The standalone bend/orbit-correction script remains diagnostic only.
- Fitted bend and quad-`dy` values become fixed baselines for later tuning.

### `tmom-recon`

- `reference_co` is mandatory on production reconstruction paths and must cover
  every selected BPM.
- `physics.transverse` removes measured positions and restores fitted model
  momenta.
- `chrom=True` optics carry `ddx`, `ddpx`, `ddy`, and `ddpy` through the
  momentum calculation when available.
- `ModelDetails.magnet_strengths` remains the plain-data fitted-lattice seam.
- Exact closed orbit remains opt-in.

## Verification status

Focused verification completed during the migration:

| repository | result |
|---|---|
| `sgd-magnet-tuner` momentum-reference tests | 11 passed |
| `psb_md` reference/workflow tests | 123 passed |
| `tmom-recon` selected migration tests | 37 passed, 1 expected xfail |
| Ruff on touched files | clean in all three repositories |
| `git diff --check` | clean in all three repositories |

The full `psb_md` suite stopped on four failures in the pre-existing dirty
rejection-threshold tests: the working tree defines
`MAX_REJECTED_FRACTION = 4/5`, while those tests require at most `1/3`. This is
unrelated to the momentum-reference migration and was left untouched.

The full `tmom-recon` and `sgd-magnet-tuner` suites were started but not allowed
to finish because their physics-heavy cases are long-running. No failure was
observed before interruption. The production acceptance run on the
`no-qde-no-qstrips` campaign remains to be performed; do not treat the synthetic
results as production validation.

## Remaining work

- Run the `no-qde-no-qstrips` `-2/0/+2 mm` campaign end to end and record fit
  residuals, fitted strengths, reference consistency, and absolute-momentum
  improvement.
- Complete the long full suites in `tmom-recon` and `sgd-magnet-tuner`.
- Resolve the unrelated `psb_md` rejection-threshold conflict, then rerun its
  full suite.
- Evaluate the prior on real data with an L-curve or equivalent diagnostic;
  synthetic knob truth was available for the study but will not be available in
  production.

## Experimental scripts and regression tests

The exploratory scripts remain under `experimental/offmom/`:

- `part_a_bend_fit.py`, `part_a2_dp_decomposition.py`, and
  `part_a3_quad_fit.py` reproduce the position-only response studies;
- `part_b_crossover.py`, `part_b2_optics_error.py`, and
  `part_b3_diagnose_plateau.py` reproduce the AC-dipole ladder and affine-offset
  diagnosis;
- `part_c_second_order_dispersion.py` establishes the MAD-NG chromatic-column
  convention;
- `dpp_sweep_pz.py` compares first- and second-order `calculate_pz` behaviour;
- `co_angle_from_orbit.py`, `co_angle_realistic.py`, and
  `co_angle_error_sweep.py` retain the superseded orbit-only angle studies for
  provenance.

The principal regression tests are:

- `tests/acd/test_madng_second_order_dispersion.py`;
- `tests/physics/test_second_order_dispersion_momenta.py`;
- `tests/momentum/test_second_order_dispersion_pipeline.py`;
- `tests/model/test_fitted_strength_roundtrip.py`;
- `psb_md/tests/test_acd_workflow.py`, including the compensated-0dpp test;
- `sgd-magnet-tuner/tests/training/test_momentum_reference.py`.

Run the exploratory scripts with the CERN `accpy` interpreter and the repository
on `PYTHONPATH`. The Part A studies take seconds to about a minute. The Part B
tracking studies build one PSB setup per momentum and take roughly one to two
minutes per point.
