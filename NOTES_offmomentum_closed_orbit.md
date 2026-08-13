# Off-momentum ACD closed-orbit handling

PSB ring 3 · 2026-08-10

## Summary

The off-momentum AC-dipole tests fail because the momentum offset was raised
(`test_psb_acd_momentum.py` 1e-3 -> 1e-2, `test_psb_closed_orbit_acd.py` 8e-3),
not because of a code change. At those offsets the pipeline's first-order
treatment of the dispersive closed orbit stops being adequate.

Two separate problems came out of the investigation:

1. **Truncation error** (fixed, opt-in). The dispersive closed orbit is modelled
   as `pt * D`, first order only. Fixed by referencing the exact closed orbit
   MAD-NG solves at `pt`.
2. **Model error** (open, and the one that matters operationally). When the model
   does not know the machine's magnet errors, the reconstruction collapses, and
   the designated remedy — the data-mean closed orbit — is provably a no-op for
   the BPM momenta.

Problem 1 is what the failing tests were tripping on. Problem 2 is the larger
issue and is untouched by the fix.

---

## 1. Diagnosis of the test failures

Three failures, all the same signature — R² just under threshold on a quantity
whose *shape* is perfect:

```
FAILED test_psb_ac_dipole_momentum_reconstruction[off_momentum-clean]       px_bpm_downstream R²=0.9836
FAILED test_psb_ac_dipole_momentum_reconstruction[off_momentum-noise_1e-5]  BR3.DES3L1_BEFORE x pos R²=0.9900
FAILED test_psb_acd_reconstruction_with_dipole_closed_orbit[off_momentum]   px_bpm_downstream R²=0.9906
```

Fitting reconstructed against true as `a·t + b` gives R² = 0.99999 once the
affine part is removed. The error is a constant offset plus a small gain term.

### The offset scales as pt²

| delta_p | px bias (downstream) | 1 − slope |
|---------|---------------------|-----------|
| 1e-3    | −1.05e-7            | 6.2e-4    |
| 3e-3    | −9.53e-7            | 1.8e-3    |
| 1e-2    | −1.06e-5            | 5.8e-3    |

Quadratic in the offset; the gain error is linear.

### Root cause

The model twiss is on-momentum (`x = px = 0`) and the dispersive closed orbit is
reintroduced only to first order as `pt * D`, in `physics/momenta.py:187-203`:

```python
x_current_norm = (x_current - pt_est * dx_current) / sqrt_beta_x
...
px = sign_x * (...betatron terms...) / sqrt_beta_x + dpx_current * pt_est
```

The neglected `pt**2 * D2` term is a constant per BPM. Measured against the
tracked turn-mean at `BR3.BPM3L3`, δ = 1e-2:

```
x  residual = −3.07e-5 m       px residual = +1.44e-5 rad
```

growing 1.37e-7 -> 1.28e-6 -> 1.44e-5 for pt ×1, ×3, ×10. Against a driven px
amplitude of ~8e-5 rad that constant is ~17% of the signal. The vertical plane is
unaffected (no vertical dispersion), matching the failure pattern.

The linear gain error has the same origin one order down: beta/alpha/phase are
also on-momentum, so chromatic optics are missing. Negligible for R² here.

### Why this hid: the kick fit cannot see it

`dpx`/`dpy` fit R² stays at 0.999999 under *both* methods, because the harmonic
fit absorbs a constant into its own offset parameter. Only the absolute BPM
momenta expose the bias.

**Kick-fit quality is not a valid check on off-momentum closed-orbit handling.**

---

## 2. The fix for the truncation error

MAD-NG already solves the exact off-momentum closed orbit, and the pipeline
already computes it — `reconstruction.py:353` builds
`tracking_tws = run_twiss(pt=model.pt)` but uses it only as a source of
dispersion columns, never as the orbit reference. Used as the reference it agrees
with the tracked turn-mean to ~1e-8.

New opt-in flag `ACDipoleConfig.dispersive_closed_orbit` (default `False`),
plumbed through `calculate_pz` -> `ACDipolePzGenerator` ->
`calculate_ac_dipole_momentum` -> `reconstruct_from_prepared`.

`pt` had two conflicting jobs — seeding MAD-NG transport, and being the
coefficient of the first-order dispersive orbit. Now separated:

```python
betatron_pt = 0.0 if dispersive_closed_orbit else model.pt
```

`model.pt` still drives transport; only the betatron stage sees zero. This also
resolves the overloading of `pt_est`, which previously doubled as the on/off
switch for dispersion (`bpm_reconstruction.py:189`).

### Results, errors matched between line and model

0.08% RMS bend field error **and** 0.1% RMS quad gradient error, both mirrored
onto the MAD-NG model:

| Case | Method | kick dpx | kick dpy | px up | px down |
|------|--------|----------|----------|-------|---------|
| on-momentum | linear pt·D | 1.000000 | 0.999999 | 1.000000 | 1.000000 |
| on-momentum | exact CO | 1.000000 | 0.999999 | 1.000000 | 1.000000 |
| off-mom δ=8e-3 | linear pt·D | 0.999999 | 1.000000 | 0.999683 | **0.990700** |
| off-mom δ=8e-3 | exact CO | 0.999999 | 1.000000 | 1.000000 | **0.999999** |

### Errors and dispersion do not superpose

The distorted orbit samples the perturbed quadrupoles off-axis and feeds down
into the dispersion, so the two contributions cannot be added:

| at BR3.BPM3L3 | CO(err,0) + CO(0,δ) | CO(err,δ) | difference |
|---------------|---------------------|-----------|------------|
| x  | −1.524234e-02 | −1.581081e-02 | +5.7e-4 |
| px | +5.100095e-03 | +5.404148e-03 | −3.0e-4 |

Both far above the 1e-4 state-consistency tolerance. Errors also make the
truncation worse rather than neutral: the pure-dispersion second-order residual
scaled to δ=8e-3 would be ~2.0e-5 in x, but with errors it is 1.25e-4 — ~6×
amplified.

This matters because `_data_mean_closed_orbit` (`reconstruction.py:749-751`)
subtracts `pt*dx` from the measured turn-mean to reduce it to a "δ=0" quantity so
`pt*D` can be re-added downstream. That round trip assumes exactly the
superposition these numbers refute.

---

## 3. The realistic case: the model does not know the errors

**This is the case that matters, and the fix does not address it.**

Bend + quad errors applied to the tracking line only; MAD-NG model nominal; no
error strengths passed to `ModelDetails`.

```
model CO error vs machine:  6.141e-03 m      (6 mm — a different lattice)

CO source             pt method       kick dpx     px up    px down
model twiss           linear pt·D     0.999963  0.969525  -1.648187
model twiss           exact CO        0.999963  0.966933  -1.930319
data mean (xy)        linear pt·D     0.999963  0.969525  -1.648187
data mean (xy)        exact CO        ACDipoleStateConsistencyError
```

`px_down` R² is *negative* — worse than predicting the mean. Which `pt` method is
used is irrelevant at this level of model error.

Intermediate case, for scale (quad errors withheld from the model, bend errors
still matched, δ=8e-3):

| Quads in model | Method | \|CO error\| | kick dpx | px up | px down |
|----------------|--------|------------|----------|-------|---------|
| yes | linear pt·D | 9.38e-07 | 0.999999 | 0.999683 | 0.990700 |
| yes | exact CO | 9.38e-07 | 0.999999 | 1.000000 | 0.999999 |
| no | linear pt·D | 2.30e-04 | 0.999964 | 0.999957 | 0.976770 |
| no | exact CO | 2.30e-04 | 0.999964 | 0.999301 | 0.997152 |

The exact CO still removes the `pt²·D2` truncation regardless of what else is
wrong (0.9768 -> 0.9972), but plateaus below threshold. It is exact for *the
model's* lattice.

### The data-mean closed orbit is a no-op for the BPM momenta

The designated remedy for an unknown-error model does nothing to the quantity
that is failing. The override *does* engage (log: `Using per-BPM data mean as
closed-orbit reference for plane(s) x, y`) — it is algebraically self-cancelling.

`_closed_orbit_momenta` infers the CO angle by feeding the CO positions through
`prepare_direct_bpm_reconstruction` — the **same** linear neighbour-pair operator
used to reconstruct the momenta. With `f` linear:

```
px_out = f_pt(x_data − x_co) + f_0(x_co)
       = f_lin(x_data) − f_lin(x_co) − pt·f_lin(D) + pt·D' + f_lin(x_co)
       = f_pt(x_data)
```

The `x_co` terms cancel identically, for any `pt` and any closed orbit.
Subtracting an orbit and restoring the angle derived from that same orbit with
that same operator cannot change the answer. Confirmed numerically: the
`data mean (xy)` row above is bit-identical to the `model twiss` row.

The override can only bite where something non-linear intervenes — the MAD-NG
transport to the marker — which is consistent with the state-consistency check
behaving differently while `px_up`/`px_down` match to all printed digits.

This is a genuine gap, independent of everything changed today.

---

## 4. Changes made

### Library

- `ACDipoleConfig.dispersive_closed_orbit: bool = False` — new opt-in flag.
- `reconstruct_from_prepared` / `calculate_ac_dipole_momentum` take the flag;
  `betatron_pt` separates the transport role of `pt` from its
  dispersion-coefficient role.
- `calculate_pz` and `ACDipolePzGenerator` swap `closed_orbit_tws` for
  `tracking_tws` when the flag is set.

Default `False`, so the existing path is unchanged.

### Test infrastructure

- `tests/psb_tracking.py`: `quad_error_rms` / `quad_error_seed` /
  `apply_quad_errors_to_model`, with `_apply_relative_quad_gradient_errors`
  scaling `k1` by `(1+N(0,rms))` and the same absolute `k1` written to MAD-NG,
  mirroring the existing bend-error pattern. Returns `quad_k1`.
- `tests/acd/test_psb_closed_orbit_acd.py`: parametrised over
  `delta_p × dispersive_closed_orbit`, quad errors enabled.
  `off_momentum-linear` is `xfail(strict=True)` with the physics in the reason.
  The other three hold the original strict limits (0.999).

```
tests/acd/test_psb_closed_orbit_acd.py:  3 passed, 1 xfailed
full suite:                              2 failed, 157 passed, 1 xfailed
```

**Caveat on this test:** it asserts the *matched-error* configuration, which is a
debugging control rather than a real scenario. It is a valid regression guard on
the `pt²·D2` truncation, but it is not evidence that the pipeline handles magnet
errors. See §3.

---

## 5. Open items

- **Data-mean angle recovery is self-cancelling** (§3). Needs a different way to
  get the closed-orbit angle — one not derived from the same operator. This is
  the blocker for measurements where the model lacks the errors.
- **No realistic-error test.** The current tests only cover matched errors. A
  test pinning the §3 failure would stop it being rediscovered.
- **Two remaining failures.** `test_psb_acd_momentum.py::[off_momentum-clean|noise_1e-5]`
  at δ=1e-2 — same truncation cause, but it calls `calculate_ac_dipole_momentum`
  directly and never sees the flag. Needs
  `closed_orbit_tws=model.run_twiss(observe=1, coupling=True, pt=model.pt)` plus
  `dispersive_closed_orbit=True`.
- **Flag default.** Whether the exact orbit should become the default is not yet
  decided.
- **A June fix appears to have been lost.** Notes record this same root cause
  fixed on `fix/acd-off-momentum-dispersive-px` in June 2026 with the same
  remedy. That code is not in the tree — `remove_closed_orbit_inplace`,
  `src/tmom_recon/acd/transport.py` and `tests/acd/test_psb_off_momentum_acd.py`
  no longer exist. Worth `git log --all --oneline -- src/tmom_recon/acd/transport.py`.
  That version also handled the transport side (lifting betatron states to the
  full-orbit frame), which this change does not.

---

## Environment note

The project `.venv` was rebuilt during this work after a `uv run` recreated it
(the `[dev]` extra requires `xtrack-tools` from the registry and fails
resolution). Reinstalled from `.[test]` plus editable local `xtrack-tools` and
`pymadng-utils`, and `xsuite` / `pymadng`.
