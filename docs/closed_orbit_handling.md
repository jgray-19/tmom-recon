# Closed-orbit handling in the AC-dipole reconstruction

Closed-orbit handling in `src/tmom_recon/acd/reconstruction.py`.

## 1. Why there is a reference at all

The AC-dipole reconstruction uses betatron coordinates. It subtracts a
reference orbit before reconstruction and restores it afterwards.

**There is exactly one source: the model twiss** (`closed_orbit_tws`), taken in
both planes, positions *and* angles. The twiss orbit is a genuine closed solution
of the lattice, so `x, px, y, py` are mutually consistent and MAD-NG can
transport the restored state without complaint. No data-derived quantity enters
the reference.

The data-mean closed-orbit option is not used.

## 2. Control flow, end to end

Inside `reconstruct_from_prepared`:

```
co_bpm   <- closed_orbit_tws           (twiss CO: x, px, y, py)
disp_bpm <- dispersion_tws or closed_orbit_tws

data = remove_closed_orbit(data, co_bpm)
   ... betatron reconstruction, MAD-NG transport, harmonic fit ...
frame[plane] += co_bpm.loc[bpm_name, plane]      # restore
```

The same `co_bpm` values are used for removal and restoration.

## 3. Why the data-mean branch was removed

Machine-orbit mismatches are handled upstream by the model/frame selection:
`--orbit-mode absolute` includes fitted orbit errors, while
`--orbit-mode dynamic` keeps only the dynamic part.

## 4. Interaction with harmonic cleaning

The restored CO angle is a DC term. Harmonic cleaning can attenuate it.

For absolute momenta, prefer the raw `*_bpm_*` columns when the DC term matters.

## 5. Where each piece lives

| symbol | file |
| --- | --- |
| `remove_closed_orbit`, restoration, `reconstruct_from_prepared` | `acd/reconstruction.py` |
| `estimate_closed_orbit` | `physics/closed_orbit.py` |
| dynamic-part characterisation | `tests/acd/test_psb_dynamic_part_acd.py` |

`estimate_closed_orbit` is **unrelated** to the remove/restore machinery above:
it belongs to the `pt` estimation path used by `physics/pt_calculation.py`, and
it is what `psb_md` wraps as `ClosedOrbitFrame.turn_mean`.
