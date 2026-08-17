# Closed-orbit handling in the AC-dipole reconstruction

How the closed orbit (CO) is chosen, removed and restored. Written against
`src/tmom_recon/acd/reconstruction.py`.

## 1. Why there is a reference at all

The AC-dipole reconstruction works in **betatron coordinates**. The measured
turn-by-turn data is an oscillation *about* the machine closed orbit, so a
reference orbit must be subtracted before the betatron reconstruction and added
back afterwards.

**There is exactly one source: the model twiss** (`closed_orbit_tws`), taken in
both planes, positions *and* angles. The twiss orbit is a genuine closed solution
of the lattice, so `x, px, y, py` are mutually consistent and MAD-NG can
transport the restored state without complaint. No data-derived quantity enters
the reference.

This is a deliberate narrowing. The library used to offer a per-plane opt-in
(`ACDipoleConfig.data_mean_closed_orbit_planes`) that replaced the twiss orbit
with the per-BPM turn-mean of the data and inferred the matching angle from those
positions. It was removed in August 2026 — see §3.

## 2. Control flow, end to end

Inside `reconstruct_from_prepared`:

```
co_bpm   <- closed_orbit_tws           (twiss CO: x, px, y, py)
disp_bpm <- dispersion_tws or closed_orbit_tws

data = remove_closed_orbit(data, co_bpm)
   ... betatron reconstruction, MAD-NG transport, harmonic fit ...
frame[plane] += co_bpm.loc[bpm_name, plane]      # restore
```

The removal and the restoration are the same object, so whatever `co_bpm`
contains cancels identically on any turn-varying quantity.

## 3. Why the data-mean branch was removed

The motivating case was the real PSB measurement: bend errors or quadrupole
misalignments give the machine a distorted orbit while the reconstruction model
is nominal, so its twiss orbit is ~0 and subtracting it leaves the whole mm orbit
in the data.

Two things made the branch the wrong place to solve that.

**It cancelled.** The angle was inferred from the mean positions with the *same*
π/2-neighbour-pair operator the orbit was then subtracted with, so the mechanism
removed and restored the same quantity. It is exactly neutral on any DC
observable — the AC-dipole kick DC offset was measured to be unchanged by it —
and on the dynamic part it cancels by linearity regardless.

**It was only first-order correct.** The angle came from nominal model optics for
an orbit the model does not contain, recovering ~90–95 % of the true CO angle. On
a model whose twiss orbit was already right, overriding it with the data mean
made the DC offset *worse* by two orders of magnitude.

The real fix lives one level up, in `psb_md`, and is a choice of *frame* rather
than a source of orbit: `--orbit-mode absolute` fits the machine's bend and
quadrupole-`dy` errors so the model twiss orbit genuinely is the machine orbit,
and `--orbit-mode dynamic` subtracts the orbit from the input data and never
restores it. See `psb_md/orbit_frame.py`.

## 4. Interaction with harmonic cleaning

The restored CO angle is a **DC (constant) term** in the reconstructed momentum.
The harmonic cleaning in `acd/cleaning.py` fits and keeps the driven harmonics,
and in doing so **attenuates that DC term**.

Consequence: for *absolute* momenta the `*_cleaned` columns can be worse than the
raw `*_bpm_*` ones. If you need absolute momenta, prefer the raw columns, or
treat the DC term of the cleaner as a separate item to fix. For dynamic-part work
the attenuation is irrelevant.

## 5. Where each piece lives

| symbol | file |
| --- | --- |
| `remove_closed_orbit`, restoration, `reconstruct_from_prepared` | `acd/reconstruction.py` |
| `estimate_closed_orbit` | `physics/closed_orbit.py` |
| dynamic-part characterisation | `tests/acd/test_psb_dynamic_part_acd.py` |

`estimate_closed_orbit` is **unrelated** to the remove/restore machinery above:
it belongs to the `pt` estimation path used by `physics/pt_calculation.py`, and
it is what `psb_md` wraps as `ClosedOrbitFrame.turn_mean`.
