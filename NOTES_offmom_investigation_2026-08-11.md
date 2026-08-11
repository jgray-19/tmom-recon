# Off-momentum closed orbit in a realistic PSB measurement

PSB ring 3 · 2026-08-11 · branch `investigate/offmom-co-2am`

Extends `NOTES_offmomentum_closed_orbit.md` (2026-08-10). The baseline numbers
there are taken as given and are not re-derived.

Scenario throughout: the **tracking line** carries 0.08% RMS bend field errors
(seed 7) *and* 0.1% RMS quad gradient errors (seed 11); the **model** knows
neither (`apply_bend_errors_to_model=False`, `apply_quad_errors_to_model=False`).
The operational handle is that the closed orbit can be measured at several
`delta_p`.

---

## Answers

**Q1. Can bend errors be usefully estimated from closed orbits at multiple
`delta_p` when quad errors are also present and unknown?**

Yes for the closed-orbit *position*, no for the individual bend errors, and — the
result that decides everything downstream — **only partly for the closed-orbit
angle**, which is the quantity the reconstruction actually consumes.

| quantity | nominal model | after fit | improvement |
|---|---|---|---|
| individual `dk0` recovered | — | 40% (corr 0.80) | — |
| max \|CO **position** error\| at BPMs, δ=8e-3 | 6.14e-03 m | 1.71e-06 m | **3600x** |
| max \|CO **angle** error\| at BPMs, δ=8e-3 | 2.57e-03 rad | 1.04e-04 rad | 25x, then plateaus |

The response matrix has **rank 17** (17 horizontal BPMs) against **32 bend
unknowns**. Everything in the 15-dimensional null space is invisible to the
measurement in position but *not* in angle, so `px_co` saturates at ~1e-4 rad no
matter how good the position fit gets. Measured, not inferred: see §A.3.

Multiple `delta_p` do help, but not in the way the framing assumed — see §A.2.

**Q2. Is the existing first-order `pt*D` treatment sufficient, or is the exact-CO
(or second-order) route needed?**

In this scenario the pt method is **irrelevant**: the residual model error swamps
it at every rung of the ladder. The crossover (§B.1) is that the choice of pt
method only starts to matter once the residual model CO *angle* error at the BPMs
falls below roughly **1e-6 rad** at δ=8e-3 — and the best orbit-only fit
achievable here stops at 1e-4 rad, two orders of magnitude short.

So: the existing first-order treatment is *not* the limitation, and neither is
the exact-CO fix a solution. The exact-CO flag remains correct and worth keeping
(it is the difference between R²=0.9907 and 0.999999 when the model *does* know
the errors), but it buys nothing in a real measurement until the model error is
fixed first.

**Q3. Is second-order dispersion (`ddx`/`ddpx`) usable, and what does it buy?**

Usable, convention established (§C.1): MAD-NG's `ddx`/`ddpx` are **per unit `pt`**
(not per delta — the 1.92x `1/beta` trap does not apply) and the Taylor `1/2` is
**already folded in**:

```
x(pt) = x(0) + pt*dx + pt**2 * ddx        (no 1/2)
```

It buys a 40-80x reduction of the truncation residual on a clean lattice and ~9x
once magnet errors are present. That makes it a genuine third method between
"linear `pt*D`" and "exact CO", and the only one available if the dispersion is
measured rather than modelled. It does not help the scenario above, because
truncation is not what limits it.

---

## A. Estimating the bend errors from off-momentum closed orbits

Scripts: `experimental/offmom/part_a_bend_fit.py`, `part_a2_dp_decomposition.py`,
`part_a3_quad_fit.py`.

**Closed-orbit measurement.** Taken as `line.twiss(method="4d", delta0=dp)` on the
xsuite tracking line, not as the turn-mean of tracked AC-dipole data. Justified
by §2 of the previous notes, where the exact MAD-NG orbit was shown to agree with
the tracked turn-mean to ~1e-8; the twiss route costs 0.24 s against 1-2 min for a
tracking run, which is what makes a 32-column response matrix and a 4-iteration
alternating fit affordable at all. The AC-dipole pipeline is still used unchanged
for everything in §B.

**Estimator.** Orbit response matrix `R = d(x,y at BPMs)/d(k0)` per powered
main-bend, by central difference on the *model*, inverted by truncated SVD
against `measured − model` orbit residuals. `delta_p ∈ {0, ±1e-3, ±3e-3, ±8e-3}`.

### A.1 Rank, not dispersion confusion, is what limits the bend fit

Singular values of `R` (32 columns): 17 non-zero, then a clean drop to 1e-18.
Vertical BPM rows contribute nothing — bend field errors are horizontal only.

| n_sv | `dk0` rms error | corr(fit, true) | \|CO err\| δ=0 | \|CO err\| δ=8e-3 |
|---|---|---|---|---|
| 6 | 8.9e-05 | 0.28 | 1.5e-03 | 1.5e-03 |
| 10 | 8.1e-05 | 0.56 | 1.21e-03 | 1.16e-03 |
| 12 | 6.3e-05 | 0.61 | 5.30e-04 | 6.79e-04 |
| 14 | 5.6e-05 | 0.71 | 2.97e-04 | 4.25e-04 |
| 16 | 5.0e-05 | 0.78 | 9.49e-05 | 2.33e-04 |
| 17 | 4.8e-05 | 0.80 | 2.50e-06 | 2.05e-04 |

(true `dk0` rms 7.96e-05; uncorrected model CO error 5.56e-03 at δ=0, 6.14e-03 at
δ=8e-3.)

Note the `dk0` rms error stops improving at 4.75e-05 — 60% of the true error is
never recovered — while the CO error at δ=0 falls to 2.5e-06. The unrecovered
part lives entirely in the null space of `R` and produces no orbit *at the BPMs*.
This is measured, not argued: the fit from the δ=0 orbit alone and the fit from
the stacked multi-δ orbits give `dk0` errors agreeing to 4 significant figures.

### A.2 What multiple `delta_p` actually buys

**Not** separation of the error orbit from the dispersive orbit — that separation
is already free, because the model's own `CO(dp)` is subtracted before fitting.
The `dk0` estimate from `{0, ±1e-3, ±3e-3, ±8e-3}` is numerically the same as the
estimate from δ=0 alone.

What it buys is a **measurement of the dispersion error** caused by the unknown
quad gradients. Fitting the per-BPM residual as a polynomial in `dp`:

| poly order | residual-of-fit rms |
|---|---|
| 0 (constant) | 1.835e-04 |
| 1 (+ linear) | 1.971e-05 |
| 2 (+ quadratic) | 1.192e-06 |

| term | rms | max | orbit contribution at δ=8e-3 |
|---|---|---|---|
| constant (bend-error orbit) | 2.68e-03 m | 5.56e-03 m | 5.6e-03 m |
| linear (dispersion error) | 3.97e-02 | 7.89e-02 | 6.31e-04 m |
| quadratic (2nd-order disp error) | 7.20e-01 | 1.28 | 8.17e-05 m |

Applying the re-measured linear+quadratic terms as an empirical per-BPM CO table
on top of the bend fit flattens the position error completely:

| model | δ=0 | δ=1e-3 | δ=3e-3 | δ=8e-3 |
|---|---|---|---|---|
| nominal | 5.56e-03 | 5.62e-03 | 5.75e-03 | 6.14e-03 |
| bend fit only | 2.44e-06 | 2.22e-05 | 6.99e-05 | 2.05e-04 |
| bend fit + measured linear disp | 2.50e-06 | 2.52e-06 | 1.88e-06 | 2.71e-05 |
| bend fit + measured linear+quad disp | 2.50e-06 | 2.86e-06 | 3.46e-06 | 2.14e-06 |

The quadratic term is required: linear-only leaves 2.7e-05 at δ=8e-3, 10x worse.

**Trap avoided.** The dispersion correction must be re-measured *against the
bend-corrected model*, not against the nominal one. Correcting the bends also
changes the model's dispersion, so applying the nominal-referenced linear term
double-counts and made things ~4x worse (8.5e-04 at δ=8e-3). Leftover terms after
the bend fit are 68x smaller in the linear coefficient (1.18e-02 vs 3.97e-02).

**Limitation.** This empirical table gives `x`, `y` at BPMs only. It carries no
`px`, `py` — the same structural problem as the data-mean closed orbit in the
previous notes. To get momenta you need a *lattice*, which is §A.3.

### A.3 Fitting the quads too, and the angle plateau

Alternating fit: bends against the δ=0 orbit (SVD, 17 sv), quads against the
δ=±8e-3 orbit residual (relative `k1` response, 48 unknowns, SVD truncated),
4 iterations. This produces a real lattice, so `px_co` follows.

| quad n_sv | \|CO position err\| δ=0 | δ=3e-3 | δ=8e-3 |
|---|---|---|---|
| 8 | 6.3e-11 | 1.65e-05 | 4.43e-05 |
| 12 | 8.5e-10 | 7.67e-06 | 2.08e-05 |
| 16 | 2.4e-09 | 5.13e-07 | 1.71e-06 |

Now the number that matters. Same models, comparing the closed-orbit **angle** at
the BPMs against the machine, at δ=8e-3:

| model | \|CO pos err\| | \|CO **angle** err\| | dqx | dqy | max\|Δβx/βx\| |
|---|---|---|---|---|---|
| nominal | 6.14e-03 | 2.566e-03 | +0.00162 | −0.00202 | 1.26e-02 |
| bend-only fit | 2.26e-04 | 1.740e-04 | +0.00169 | −0.00219 | 1.27e-02 |
| bend+quad q8 | 4.43e-05 | 1.112e-04 | +0.00161 | −0.00205 | 1.30e-02 |
| bend+quad q12 | 2.08e-05 | 1.074e-04 | +0.00159 | −0.00201 | 1.26e-02 |
| bend+quad q16 | 1.71e-06 | 1.044e-04 | −0.00001 | +0.00100 | 4.24e-03 |
| TRUE errors | 0 | 0 | 0 | 0 | 0 |

The position error improves 3600x; the angle error improves 25x and then sits at
1.04e-04 rad regardless. **The plateau is the null space of a position-only
response matrix.** 17 horizontal BPMs cannot pin 32 bend errors, and the
undetermined combinations are precisely those that close in position but not in
angle at the BPM locations.

**Contamination by the unknown quads** (control run with quad errors switched off
in the machine): the bend-fit residual is then *flat* in `dp` (1.0e-04 at every δ
from 0 to ±8e-3), against 2.5e-06 → 2.05e-04 growing with δ when quads are
present. So the quad contamination is entirely a *dispersion* error, ~2.5e-05 m
per 1e-3 of `dp`, and it is separable — that is what §A.2 does.

---

## B. Does the pt method still matter? (the crossover)

Script: `experimental/offmom/part_b_crossover.py`. Full AC-dipole pipeline,
tracked data with errors, model fed the Part A estimates through
`ModelDetails.magnet_strengths`. R² on absolute BPM momenta against tracked truth,
never on the kick fit.

### B.1 The ladder

δ = 8e-3:

| model | \|CO pos err\| | pt method | kick dpx | px up | px down |
|---|---|---|---|---|---|
| nominal | 6.14e-03 | linear | 0.999963 | 0.969525 | **−1.648187** |
| nominal | 6.14e-03 | exact | 0.999963 | 0.966933 | **−1.930319** |
| bend-only fit | 2.26e-04 | linear | 0.999964 | 0.824770 | 0.865663 |
| bend-only fit | 2.26e-04 | exact | 0.999964 | 0.841242 | 0.783185 |
| bend+quad q8 | 4.43e-05 | linear | 0.999963 | 0.828415 | 0.865726 |
| bend+quad q8 | 4.43e-05 | exact | 0.999963 | 0.842939 | 0.785819 |
| bend+quad q12 | 2.08e-05 | linear | 0.999949 | 0.832664 | 0.863132 |
| bend+quad q12 | 2.08e-05 | exact | 0.999949 | 0.846972 | 0.782486 |
| bend+quad q16 | 1.71e-06 | linear | 0.999997 | 0.822520 | 0.842713 |
| bend+quad q16 | 1.71e-06 | exact | 0.999997 | 0.837033 | 0.756522 |
| TRUE errors | 0 | linear | 0.999999 | 0.999683 | **0.990700** |
| TRUE errors | 0 | exact | 0.999999 | 1.000000 | **0.999999** |

δ = 3e-3 and δ = 1e-3 show the same structure with the same plateau
(px down 0.78-0.80 for every fitted model, both methods; TRUE errors 0.999839 /
0.999998 linear and 1.000000 exact). Full log: `experimental/offmom/part_b_results.txt`.

Reading the crossover off this table:

| residual model CO **angle** error | does the pt method matter? |
|---|---|
| 2.6e-03 rad (nominal) | no — reconstruction is broken (R² negative) |
| 1.7e-04 … 1.0e-04 rad (any orbit-only fit) | no — plateau at R² ≈ 0.78-0.87, ±0.08 between methods and *not* systematically in favour of exact |
| 0 rad (matched errors) | **yes** — 0.9907 (linear) vs 0.999999 (exact) |

The δ=8e-3 rows where `linear` beats `exact` (0.866 vs 0.783) are **not** evidence
that linear is better. Both are dominated by the same constant offset (§B.2); the
neglected `pt²D2` term happens to partially cancel it. Reporting that as a win
would be exactly the self-cancelling-operator trap the previous notes warn about.

Because the plateau is flat from 1.7e-04 down to 1.7e-06 in position error, the
crossover cannot be placed from the position axis at all — it is controlled by the
angle error, which never drops below 1.0e-04 in this ladder. Extrapolating from
the matched-error rung, the pt method starts to matter when the angle error is
below the size of the truncation term itself, ~1e-6 rad at δ=8e-3.

### B.2 What the plateau actually is

Script: `experimental/offmom/part_b3_diagnose_plateau.py`. Fitting
`reconstructed = a·true + b` on `px` at the upstream/downstream BPMs, δ=1e-3:

| model | px up R² | gain `a` | offset `b` | R² after affine removal |
|---|---|---|---|---|
| nominal | 0.967849 | 1.00013 | +1.62e-05 | 1.000000 |
| bend-only fit | 0.831781 | 1.00021 | −3.79e-05 | 1.000000 |
| bend+quad q16 fit | 0.824010 | 0.99984 | −3.82e-05 | 1.000000 |
| **true bends + fitted quads** | **0.999833** | 1.00021 | +8.8e-07 | 1.000000 |
| TRUE errors | 1.000000 | 0.99998 | +1.4e-09 | 1.000000 |

Three things follow, all measured:

1. The plateau is a **pure constant offset** of −3.8e-05 rad against a driven `px`
   amplitude of 9.15e-05 rad (42% of the signal). Shape and gain are perfect
   (R² = 1.000000 after removing the affine part).
2. The offset is the residual **closed-orbit angle** error of §A.3 (1.0e-04 rad
   at the worst BPM, −3.8e-05 at these two), fed straight through.
3. **The bend fit is the limitation, not the quad fit.** Giving the model the true
   bends and keeping the *fitted* quads restores R² = 0.9998. The quad fit is good
   enough; the rank-17 bend fit is not.

This also explains why the nominal model has a *better* `px up` (0.968) than the
fitted models (0.82): with a 6 mm CO error the pipeline's failure is concentrated
downstream (R² = −1.65), while the fitted models spread a smaller, more uniform
angle error across both sides. Improving the model CO position by 3600x improved
`px down` from −1.65 to 0.79 and made `px up` worse. Both are the same offset
mechanism.

---

## C. Second-order dispersion

Script: `experimental/offmom/part_c_second_order_dispersion.py`.
Test: `tests/acd/test_madng_second_order_dispersion.py`.

### C.1 Convention (established, not assumed)

Fitting `x(pt) − x(0) = c1·pt·dx + c2·pt²·ddx` over the 17 BPMs against the exact
MAD-NG orbit, nominal PSB ring 3 lattice (β = 0.5197529):

| δ | pt | c1(x) | c2(x) | c1(px) | c2(px) |
|---|---|---|---|---|---|
| 1e-3 | 5.199e-04 | 1.00000 | 0.99800 | 1.00000 | 0.99241 |
| 3e-3 | 1.561e-03 | 1.00000 | 0.99399 | 1.00001 | 0.99132 |
| 8e-3 | 4.170e-03 | 1.00001 | 0.98408 | 1.00000 | 0.98957 |
| 1e-2 | 5.216e-03 | 1.00002 | 0.98015 | 1.00000 | 0.98932 |

`c1 = 1.00000` — the columns are **per unit `pt`**, not per unit delta (per delta
would give 1/β = 1.924). `c2 → 1` as δ → 0 — the Taylor **1/2 is already folded
into `ddx`/`ddpx`**. The drift of `c2` to 0.98 at δ=1e-2 is the neglected third
order, consistent with `c2` moving linearly with δ.

```
CO(pt) = CO(0) + pt * [dx, dpx] + pt**2 * [ddx, ddpx]
```

### C.2 Residual left by a second-order closed-orbit model

Max over BPMs against the exact MAD-NG orbit.

Nominal lattice (no magnet errors):

| δ | x, 1st order | x, 2nd order | px, 1st order | px, 2nd order |
|---|---|---|---|---|
| 1e-3 | 4.96e-07 | 7.52e-10 | 1.43e-07 | 1.69e-10 |
| 3e-3 | 4.46e-06 | 2.03e-08 | 1.29e-06 | 4.56e-09 |
| 8e-3 | 3.16e-05 | 3.84e-07 | 9.14e-06 | 8.63e-08 |
| 1e-2 | 4.92e-05 | 7.50e-07 | 1.43e-05 | 1.68e-07 |

With the machine's bend + quad errors applied to the model (the operationally
relevant case):

| δ | x, 1st | x, 2nd | px, 1st | px, 2nd |
|---|---|---|---|---|
| 1e-3 | 1.74e-06 | 2.21e-08 | 5.94e-07 | 8.42e-09 |
| 3e-3 | 1.61e-05 | 6.22e-07 | 5.51e-06 | 2.37e-07 |
| 8e-3 | 1.23e-04 | 1.32e-05 | 4.25e-05 | 4.99e-06 |
| 1e-2 | 1.99e-04 | 2.70e-05 | 6.88e-05 | 1.02e-05 |

Gain: **40-80x on a clean lattice, ~9x with errors** (and `c2` drifts to 1.17-1.23
with errors, i.e. the error orbit feeding down through the perturbed quads
generates third-order terms that `ddx` cannot represent). This reproduces the
previous notes' finding that errors amplify the truncation ~6x, from the other
direction.

Verdict on a third method: worth having, but it is *not* a substitute for the
exact CO when the model is trustworthy (5.0e-06 rad residual at δ=8e-3 vs ~1e-8
for exact). Its real niche is the one named in the brief — good dispersion
knowledge, untrustworthy model closed orbit, e.g. a **measured** dispersion. Note
that §A.2 already measures exactly that (the linear and quadratic `dp`
coefficients of the orbit), so the two combine naturally; the blocker is that a
per-BPM measured table has no `px` (see open items).

### C.3 Chromatic phase: `dmu1` explains the linear gain error

The gain error documented in the previous notes (1−slope = 6.2e-4 / 1.8e-3 /
5.8e-3 at δ = 1e-3 / 3e-3 / 1e-2) comes from on-momentum β/μ. Measured phase
movement and how well `dmu1` predicts it (nominal lattice):

| δ | max\|Δμx\| [2π] | max\|Δβx/βx\| | fitted `c1` on `pt·dmu1` | residual after `dmu1` |
|---|---|---|---|---|
| 1e-3 | 3.250e-03 | 1.71e-04 | 0.99848 | 4.94e-06 |
| 3e-3 | 9.729e-03 | 5.12e-04 | 0.99545 | 4.44e-05 |
| 8e-3 | 2.579e-02 | 1.36e-03 | 0.98795 | 3.15e-04 |
| 1e-2 | 3.217e-02 | 1.70e-03 | 0.98498 | 4.91e-04 |

`dmu1` is also **per unit `pt`** (`c1 → 1`) and removes 98-99% of the chromatic
phase shift: 2.58e-02 → 3.15e-04 at δ=8e-3, a 78x reduction. So yes, the
chromatic optics terms do explain and can correct the linear-in-pt gain error.

Caveat, stated explicitly: this is measured on the *twiss columns*, not through
the reconstruction. In §B.2 the observed gains were 1.0002 (2e-4) — well below
the offset error — so correcting them changes nothing until the offset is fixed.
`wx`/`phix` are present but were not needed; `dmu1` alone suffices.
`dmu2` was **not** present in this MAD-NG version's output (columns found:
`dx dpx ddx ddpx dmu1 wx`) — `dmu2`/`phix`/`ddy`/`ddpy` are absent for this
lattice, presumably because the PSB model here is uncoupled with no vertical
dispersion. Not chased further.

---

## D. What this means for the pipeline

1. **The exact-CO flag is correct and should stay**, but it is not the fix for a
   real measurement. Keep it opt-in; the case for making it default is the
   matched-model case only.
2. **Orbit-based error fitting fixes the wrong quantity.** Fitting BPM *positions*
   drives the position error to 1e-6 while leaving the angle error at 1e-4. Since
   the reconstruction consumes closed-orbit *momenta*, a 3600x position
   improvement bought a plateau at R² ≈ 0.8, not a solution.
3. **The constraint is BPM count.** 17 horizontal BPMs, 32 bends. Any fix has to
   either reduce the unknowns (fit per-sector or per-magnet-family rather than
   per-magnet, or regularise towards a plausible error spectrum) or add
   independent information that constrains the angle.
4. **Validate on absolute BPM momenta, and decompose.** `kick dpx` stayed at
   0.99996-0.999999 across *every* row of §B.1, including the R² = −1.65 rows.
   The affine decomposition in §B.2 is what turned "R² = 0.82, unclear why" into
   "a −3.8e-05 rad constant, and it is the bend fit's fault".

---

## Open items

- **Constrain the closed-orbit angle.** The whole result hangs on this. Options
  not tried: fitting per-family rather than per-magnet to get the unknowns below
  17; Tikhonov regularisation towards the known error spectrum instead of a hard
  SVD cut; using the *driven* data (turn-by-turn phase) rather than only the
  orbit; using dispersion measurements at the BPMs as extra rows.
- **The measured-dispersion table has no `px`.** §A.2 gives a per-BPM CO table
  accurate to 2e-6 m at every δ, but only in position. Turning it into momenta
  runs straight back into the self-cancelling operator documented in §3 of the
  previous notes. A second-order dispersion route (§C) applied to a *measured*
  dispersion would inherit the same problem unless `dpx` is modelled.
- **No test pins the §B plateau.** As in the previous notes' open items, the
  realistic-error failure is still untested. The scripts in
  `experimental/offmom/` reproduce it but take ~10 min.
- **Second-order CO is not implemented in the pipeline.** §C establishes the
  convention and the gain; nothing was wired into `momenta.py`. Doing so would
  need `chrom=true` on the model twiss and a third value of the pt-method
  selector, currently a bool.
- **`dmu2` / `ddy` / `ddpy` absent from the twiss output.** Assumed to be the
  uncoupled/no-vertical-dispersion case; not verified.
- **Bend fit uses xsuite for the response matrix and MAD-NG for the
  reconstruction.** The two agree to ~1e-9 (established), so this is believed
  harmless, but the fit was never repeated with a MAD-NG response matrix.

---

## Files added (branch `investigate/offmom-co-2am`)

| file | what |
|---|---|
| `experimental/offmom/co_common.py` | line/BPM/bend helpers, closed orbit at δ |
| `experimental/offmom/part_a_bend_fit.py` | response matrix + SVD bend fit, n_sv sweep, quad-contamination control |
| `experimental/offmom/part_a2_dp_decomposition.py` | polynomial-in-δ decomposition, measured dispersion correction |
| `experimental/offmom/part_a3_quad_fit.py` | alternating bend+quad fit, writes `fitted_errors_q*.npz` |
| `experimental/offmom/part_b_crossover.py` | full pipeline over the model ladder × pt method × δ |
| `experimental/offmom/part_b2_optics_error.py` | β-beat, tune and CO-angle error of each rung |
| `experimental/offmom/part_b3_diagnose_plateau.py` | affine decomposition of the plateau |
| `experimental/offmom/part_c_second_order_dispersion.py` | `ddx`/`ddpx`/`dmu1` convention and residuals |
| `tests/acd/test_madng_second_order_dispersion.py` | pins the §C.1 convention (`@pytest.mark.slow`) |

Run as `PYTHONPATH=. .venv/bin/python experimental/offmom/<script>.py`. Part A
scripts are seconds to ~1 min; Part B builds one PSB tracking setup per `delta_p`
(~1-2 min each) and is cached within the run.

---

## Test status on this branch

```
.venv/bin/python -m pytest tests/acd/test_psb_closed_orbit_acd.py -q
    3 passed, 1 xfailed              (unchanged from the 2026-08-10 notes)

.venv/bin/python -m pytest tests/acd -q
    2 failed, 54 passed, 1 xfailed
    the 2 failures are the known pre-existing
    test_psb_acd_momentum.py::...[off_momentum-clean|noise_1e-5]

.venv/bin/python -m pytest tests/acd/test_madng_second_order_dispersion.py -q
    1 passed                         (new)
```

The `tests/acd` run above predates the new test file; the new test was run
separately and passes.
