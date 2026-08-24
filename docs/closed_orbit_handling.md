# Orbit frames, momentum, and dispersion

This page is the canonical contract for closed-orbit handling in tmom-recon.

## The coordinate origin

The measured orbit at the campaign's setting **0** is coordinate zero. Every
measurement, supplied momentum, and reconstructed state is expressed relative to
that one orbit. It is not legitimate to replace it with a model orbit or with the
turn mean of the file currently being reconstructed.

There are three distinct physical quantities:

1. the non-dispersive machine orbit at setting 0;
2. the dispersive displacement and momentum at the acquisition's momentum;
3. the driven/betatron motion.

Changing the coordinate frame may remove (1). It must never remove or disable
(2), and it must preserve (3).

## Required ordering

For every acquisition tmom-recon performs the following sequence:

1. Start from raw BPM positions.
2. Subtract the same measured orbit-0 positions in dynamic planes.
3. Estimate `pt`, when it was not supplied, from these transformed coordinates.
4. Reconstruct transverse momenta with first- and available second-order
   dispersion.
5. Restore the state selected by the frame.

The ordering is load-bearing. Estimating momentum from raw coordinates leaks an
unknown dipole-error orbit into `pt`. Subtracting each file's own turn mean gives
each momentum setting its own zero and erases the dispersive signal.

## Frame behavior

`ReconstructionFrame` owns the transformation:

| frame | input transformation | restored state |
| --- | --- | --- |
| absolute | none | measured orbit-0 `x/y` and fitted `px/py` |
| dynamic | subtract orbit-0 in x and y | zero non-dispersive state |
| horizontal retained | subtract orbit-0 in y | measured x and fitted px; zero y/py |

The generic API represents these choices through `dynamic_planes`; mode names
belong to applications. A retained plane requires an explicit fitted angle.
There is deliberately no implicit zero-angle or model-angle fallback.

When dipolar components or quadrupole `dy` are fitted, their strengths belong in
`ModelDetails` and their BPM angles belong in `fitted_momenta`. Fitted strengths
must not be applied in a dynamic plane, because that would put a non-dispersive
orbit back into a frame from which it was removed.

## Momentum convention

The orbit-zero frame has `pt = 0` by definition. `measurement_pt_offset` is the
MAD-NG `pt` offset from it, never a machine-absolute value. If it is omitted,
tmom-recon estimates the same offset after applying the frame transformation.
There is no second reference-momentum subtraction.

`estimate_pt_from_model` follows the same contract when called directly: it
accepts raw data and a `ReconstructionFrame`, applies the frame, and projects the
remaining orbit onto `D`. If `DD` is available it solves the second-order
quadratic as well.

## Dispersion is independent of the frame

Dynamic does **not** mean non-dispersive. The reconstructed physical state uses

```
x_beta  = x - pt*D_x  - pt**2*DD_x
px      = px_beta + pt*D_px + pt**2*DD_px
```

and the analogous vertical expressions. Model or measured dispersion may be
selected, but `D` and `D'` must come from the same source. Second-order
dispersion comes from a chromatic model when available.

`use_dispersion=False` is only valid for an explicitly pure transverse,
zero-offset calculation. A nonzero `measurement_pt_offset` with dispersion
disabled is rejected.

## AC-dipole reconstruction

The coordinate-frame subtraction happens before both the all-BPM and AC-dipole
paths. The ACD BPM reference is composed explicitly as

```
frame.closed_orbit + (tracking_orbit_model - orbit_zero_model)
```

The two model tables are mandatory. Their difference contains the physical
dispersive change at the acquisition momentum without replacing the measured
origin by a model orbit. This composed state is removed for betatron
reconstruction and added back only after the dynamic kick is calculated.

Thus a dynamic frame suppresses the non-dispersive orbit-0 state while retaining
the dispersive position and momentum. In retained planes, fitted model strengths
provide the tracking orbit and the frame restores measured BPM positions with
fitted BPM angles.

## Failure modes that must remain impossible

- deriving dispersion enablement from dynamic/absolute mode;
- estimating `pt` before orbit-0 subtraction in a dynamic plane;
- subtracting a per-file mean or an independently estimated orbit per momentum;
- passing machine-absolute `pt` as an offset, or subtracting the origin twice;
- using a model orbit in place of measured orbit-zero positions;
- restoring fitted dipole or quad-`dy` state in a dynamic plane;
- silently supplying zero/model angles for a retained plane;
- mixing measured `D` with model `D'`.
