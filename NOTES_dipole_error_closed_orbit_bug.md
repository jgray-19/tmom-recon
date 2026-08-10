# ACD reconstruction fails with a dipole-error closed orbit — findings

**Date:** 2026-07-08
**Test:** `tests/acd/test_psb_closed_orbit_acd.py` (currently RED, intentionally)
**Status:** OPEN BUG confirmed in the reconstruction closed-orbit handling.

## The real failure this reproduces

In `../psb_md/scripts/optimise_quads.py` a real PSB measurement reconstruction raises:

```
ValueError: Reconstructed x at BPM BR3.BPM2L3 does not match the predicted value
within absolute tolerance 1.0e-04 (max|residual|=7.698e-04).
```

from `_check_bpm_state_consistency` in
`src/tmom_recon/acd/reconstruction.py:643`.

## What I built

A controlled reproduction that adds a **0.08 % RMS relative dipole field error** to
every powered PSB main bend, applied in a *matched* way to **both**:

- the **xsuite tracking line** (generates the "measurement" — an AC-dipole
  excitation seeded on the distorted closed orbit), and
- the **MAD-NG reconstruction model** (so its twiss carries the same closed orbit
  and matches the model → `_check_has_zero_closed_orbit` takes the
  closed-orbit-removal branch, exactly like the real measurement).

Helpers live in `tests/psb_tracking.py`:
`build_psb_tracking_setup(..., bend_error_rms=, bend_error_seed=)`,
`_apply_bend_errors_to_line`, `_apply_bend_errors_to_model`.

## Result: it is a code bug, not a data/model mismatch

- The two codes' closed orbits agree to **~1.5e-9 at the BPMs** (the xsuite/MAD-NG
  numerical floor) — four orders of magnitude below the failing residual.
- The reconstruction still fails at `_check_bpm_state_consistency` with
  **max|residual| ≈ 1.05e-4 in x** at `BR3.BPM2L3`.
- The **dpx/dpy momentum harmonic fits are perfect (R² = 1.0)**. The failure is in
  the **x position** round-trip only, **horizontal plane** only.

Since data and model share the orbit to 1e-9, the ~1e-4 residual cannot be a
mismatch → **bug in the closed-orbit handling** of `reconstruct_from_prepared`
(the `non_zero_tws` branch: `remove_closed_orbit` → add CO back to momenta → MAD-NG
transport → `transport_marker_state_to_bpm`). This matches the latent warning in the
off-momentum fix: that fix handled the *dispersive* CO (`run_twiss(deltap)`), but a
*dipole-error* CO at pt=0 is not covered by the `co_frame`/`closed_orbit` machinery.

## Key gotcha (for the matched perturbation)

Perturb the bend's **native dipole field `k0`** (`k0 = h·(1+δ)`, geometry
`h = angle/L` held fixed), **not** a separate `knl[0]` multipole. A stand-alone
dipole multipole inside a *curved* element is transported differently by xsuite vs
MAD-NG (~0.1 % / ~1e-5 orbit disagreement on a 3e-3 orbit); the native `k0` field is
handled identically by both codes → ~1e-9 agreement. PSB main bends are 32 sector
bends `br.bhz{N}{1,2}` (upper-cased in MAD-NG) with `k0 = from_h` (nominal
`k0 == h == angle/L`).

## Next step to debug

Trace the horizontal x round-trip in `reconstruct_from_prepared` for the
`non_zero_tws` branch: confirm whether the dipole-error CO in `x` is removed from
positions but not consistently restored/transported through
`transport_marker_state_to_bpm`, the way the dispersive CO is. Make the test green.
