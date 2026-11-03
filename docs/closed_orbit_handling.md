# Closed-orbit handling

Every reconstruction API requires `closed_orbit_at_zero`, containing measured
BPM `x/y` at `dp=0`, and `orbit_mode`, exactly `"dynamic"` or `"absolute"`.
Names are matched case-insensitively and must be unique, finite, and cover every
reconstructed BPM. Extra `px/py` columns are ignored.

The package subtracts measured zero-momentum positions before reconstruction and
generates every reference Twiss internally from `ModelDetails` and the active
strengths. The mode controls the state restored afterwards:

- `dynamic`: generated zero-momentum model `x/px/y/py`;
- `absolute`: measured zero-momentum `x/y` and generated model `px/py`.

Off-momentum displacement remains in the subtracted data, preserving dispersive
motion relative to the zero baseline. The same state is restored before ACD
fitting and transport and in kicker reconstruction. Strength updates regenerate
the model zero reference; momentum-only updates may reuse it.

`estimate_closed_orbit_pt` takes `closed_orbit_at_zero` directly because the
subtraction is identical in both modes.

## Migration

`Frame`, `DynamicFrame`, and `AbsoluteFrame` were removed without compatibility
aliases. Replace, for example:

```python
calculate_pz(data, details, frame=DynamicFrame(closed_orbit, reference), barrier_s=None)
```

with:

```python
calculate_pz(
    data,
    details,
    closed_orbit_at_zero=closed_orbit,
    orbit_mode="dynamic",
    barrier_s=None,
)
```

Use `orbit_mode="absolute"` for the former absolute workflow and remove the old
reference/estimated Twiss entirely. Both `psb_md` and `sgd-magnet-tuner` must
migrate reconstruction, estimator, ACD, kicker, and generator calls; those
repositories are intentionally not changed here.
