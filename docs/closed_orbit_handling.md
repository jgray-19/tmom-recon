# Closed-orbit handling in the AC-dipole reconstruction

How the closed orbit (CO) is chosen, removed and restored, and every branch the
code can take. Written against `src/tmom_recon/acd/reconstruction.py` and
`src/tmom_recon/physics/closed_orbit.py`.

## 1. Why there is a choice at all

The AC-dipole reconstruction works in **betatron coordinates**. The measured
turn-by-turn data is an oscillation *about* the machine closed orbit, so a
reference orbit must be subtracted before the betatron reconstruction and added
back afterwards. The question is only: *where does that reference come from?*

Two sources exist, and they are not always the same object:

| source | what it is | when it is right |
| --- | --- | --- |
| **model twiss** (`closed_orbit_tws`) | the CO of the MAD-NG model, positions *and* angles, exact and self-consistent | the model reproduces the machine orbit (errors are in the model) |
| **data mean** | per-BPM turn-mean of the measurement; positions only | the model twiss is on ~zero but the machine has a real mm-scale orbit |

The data mean is a valid CO estimate because the driven oscillation has ~zero
mean over the flat-top turns, so the per-BPM mean of `x`/`y` *is* the orbit at
that BPM.

## 2. The default branch: twiss closed orbit

With `data_mean_closed_orbit_planes=None` (the default), **both planes take the
CO straight from `closed_orbit_tws`** — position and angle. This is the
preferred path: the twiss orbit is a genuine closed solution of the lattice, so
`x, px, y, py` are mutually consistent and MAD-NG can transport the restored
state without complaint.

No data-derived quantity enters the reference at all. The only thing computed
from the data is the mismatch warning (§4).

## 3. The opt-in branch: data-mean closed orbit

```python
ACDipoleConfig(..., data_mean_closed_orbit_planes="x")   # "x" | "y" | "xy" | "yx" | None
```

Selected **per plane**, so `"x"` means *x from the data, y from the twiss*. The
string is case-insensitive and order-insensitive; it is normalised by
`parse_plane_spec` to a canonical `("x", "y")`-ordered tuple, and validated in
`ACDipoleConfig.__post_init__` so a typo raises at config construction rather
than deep inside a MAD-NG run.

The motivating case is the real PSB measurement: bend field errors or
quadrupole misalignments give the machine a distorted orbit, but the
reconstruction model is nominal, so its twiss orbit is ~0. Subtracting zero
would leave the whole mm orbit in the data, biasing the betatron reconstruction
and tripping `_check_bpm_state_consistency`.

### 3.1 Building the data-mean orbit

`_data_mean_closed_orbit` groups the data by BPM and takes the mean of `x`/`y`,
then **subtracts the dispersive orbit `pt · D`**. This matters: the twiss
reference is a `dp/p = 0` orbit, while the data mean on an off-momentum run
contains `pt·Dx`. Without this correction the two are different physical
quantities and the comparison in §4 is meaningless.

### 3.2 Recovering the angle

The data mean gives **positions only**. Restoring a position with a zero angle
would produce a state that is not a closed orbit; MAD-NG would transport it into
something wrong and the consistency check would fail.

Instead `_closed_orbit_momenta` infers the angle from the positions:

1. the static mean orbit is replayed as **two identical synthetic "turns"**,
   with `var_x = var_y = 0`;
2. it is fed through `prepare_direct_bpm_reconstruction` — the *same*
   π/2-phase-advance neighbour-pair machinery used for the turn-by-turn momenta
   — with `pt_est=0.0`;
3. the resulting `px`/`py` are written back into the reference.

Two details in that function are deliberate:

- Only the momenta of the **overridden planes** are zeroed and refilled. A plane
  still taking the twiss CO keeps its exact twiss angle. (An earlier version
  zeroed both and silently destroyed the non-overridden plane's angle.)
- A neighbour look-up that **wraps around the ring** shifts the partner turn by
  ±1, leaving one of the two synthetic turns without a partner and therefore
  `NaN`. The code takes the first *non-NaN* row rather than row 0.

The two synthetic turns exist purely so that those wrap-around look-ups find a
partner at all.

**Known limitation.** The angle is inferred with the *nominal model* optics, for
an orbit the model does not contain, so it is only **first-order correct**. In
the zero-twiss PSB test it recovers roughly 90–95 % of the true CO angle. That
is far better than restoring nothing (a 100 % miss, the old behaviour), but it
is not exact, and the residual grows where the local optics error is largest.
`tests/acd/test_psb_zero_twiss_acd.py` asserts on that residual rather than on
R², precisely because the quantity being characterised is the leftover angle.

## 4. The warning branch

`warn_on_closed_orbit_mismatch` compares the twiss CO against the
(dispersion-corrected) data mean and logs a warning when
`max |twiss − data| > CLOSED_ORBIT_WARN_TOLERANCE`.

- The tolerance is **1 mm, absolute**, a module constant in
  `physics/closed_orbit.py`. It is deliberately *not* a configurable argument.
- It is checked **only for planes not overridden**. In an overridden plane the
  twiss orbit is known to be wrong — that is why the user opted in — so warning
  about it would be noise.
- The message names the worst BPM, its offset, the RMS, and suggests the
  `data_mean_closed_orbit_planes` value that would address it.

So the warning is the mechanism that turns the silent failure mode ("model
twiss is on zero, results are quietly biased") into something visible.

## 5. Control flow, end to end

Inside `reconstruct_from_prepared`:

```
co_bpm   <- closed_orbit_tws           (twiss CO: x, px, y, py)
disp_bpm <- dispersion_tws or closed_orbit_tws
mean_co  <- _data_mean_closed_orbit(data, ..., disp_bpm, model.pt)

warn_on_closed_orbit_mismatch(co_bpm, mean_co, planes = not-overridden)

if override_planes:                    # branch B
    for plane in override_planes:
        co_bpm[plane] = mean_co[plane]           # positions from data
    co_bpm = _closed_orbit_momenta(co_bpm, ...)  # angles inferred
                                       # branch A (default): co_bpm untouched

data = remove_closed_orbit(data, co_bpm)
   ... betatron reconstruction, MAD-NG transport, harmonic fit ...
frame[plane] += co_bpm.loc[bpm_name, plane]      # restore
```

Both branches converge on the *same* remove/restore code. The only difference is
what `co_bpm` contains by the time `remove_closed_orbit` is called. That is the
key structural property: there is no separate "data-mean code path" downstream,
so the two modes cannot drift apart.

## 6. Interaction with harmonic cleaning

The restored CO angle is a **DC (constant) term** in the reconstructed momentum.
The harmonic cleaning in `acd/cleaning.py` fits and keeps the driven harmonics,
and in doing so **attenuates that DC term**.

Consequence: for *absolute* momenta the `*_cleaned` columns can be worse than
the raw `*_bpm_*` ones. In the zero-twiss `y` test the raw momenta recover the
CO angle to ~5 % while the cleaned ones only reach ~17 %. This is pre-existing
cleaner behaviour that the CO restoration merely made visible — before, the
angle was absent from both. The test therefore asserts only on the raw momenta
and logs the cleaned ones.

If you need absolute momenta, prefer the raw columns, or treat the DC term of
the cleaner as a separate item to fix.

## 7. Where each piece lives

| symbol | file |
| --- | --- |
| `CLOSED_ORBIT_WARN_TOLERANCE`, `parse_plane_spec`, `warn_on_closed_orbit_mismatch` | `physics/closed_orbit.py` |
| `_data_mean_closed_orbit`, `_closed_orbit_momenta`, branch in `reconstruct_from_prepared` | `acd/reconstruction.py` |
| `data_mean_closed_orbit_planes` field + `__post_init__` validation | `acd/integration.py` |
| call sites threading the option through | `reconstruction.py` |
| characterisation test | `tests/acd/test_psb_zero_twiss_acd.py` |

`estimate_closed_orbit`, also in `physics/closed_orbit.py`, is **unrelated** to
this machinery: it belongs to the `pt` estimation path used by
`physics/pt_calculation.py`.
