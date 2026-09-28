# tmom-recon
[![codecov](https://codecov.io/gh/jgray-19/tmom-recon/graph/badge.svg?token=1R2UUJGSP3)](https://codecov.io/gh/jgray-19/tmom-recon)
[![Coverage](https://github.com/jgray-19/tmom-recon/actions/workflows/coverage.yml/badge.svg)](https://github.com/jgray-19/tmom-recon/actions/workflows/coverage.yml)

Momentum reconstruction utilities for turn-by-turn BPM data.

The package bundles the reconstruction formulae behind focused all-BPM,
AC-dipole, and kicker workflows, together with lattice helpers and accelerator
descriptors used by the MAD-NG drivers.

## Requirements

- Python 3.11 or newer
- `numpy`, `pandas`, `scipy`, `tfs-pandas`, `omc3`

Optional workflows need extra packages:

- AC-dipole reconstruction and MAD-NG-driven tests also rely on `pymadng_utils`
  and `xtrack-tools`
- local development uses `pytest`, `pytest-cov`, `ruff`, and `pre-commit`

## Install

Base install:

```bash
python -m pip install -e .
```

With test dependencies:

```bash
python -m pip install -e '.[test]'
```

With development dependencies:

```bash
python -m pip install -e '.[dev,test]'
```

If you plan to use the AC-dipole reconstruction helpers, install the external
tracking stack in the same environment as well.

## Public API

The top-level package re-exports the main entry points:

```python
from tmom_recon import (
    ACDipoleConfig,
    OpticsInput,
    ModelDetails,
    build_twiss_from_measurements,
    calculate_acd_pz,
    calculate_kicker_pz,
    calculate_pz,
    inject_noise_xy,
)
```

Main modules:

- `tmom_recon.physics`: two-BPM transverse and dispersive momentum formulae.
- `tmom_recon.measurements`: measured `delta p / p` and Twiss reconstruction helpers.
- `tmom_recon.acd`: AC-dipole reconstruction, BPM override, and MAD-NG integration helpers.
- `tmom_recon.kicker`: single-kick reconstruction helpers based on kicker-to-BPM transport.
- `tmom_recon.lattice`: neighbor, lattice, and transport-matrix helper functions.

## Usage

All-BPM reconstruction starts from raw BPM positions and an explicit measured
setting-zero orbit:

```python
from tmom_recon import ModelDetails, calculate_pz

result = calculate_pz(
    tracking_df,
    ModelDetails(accelerator=accelerator, pt=pt_offset),
    closed_orbit_at_zero=measured_orbit_zero[["x", "y"]],
    orbit_mode="dynamic",
    barrier_s=None,
)
```

`ModelDetails.pt` must contain the caller's momentum estimate. `tracking_df` is expected to contain turn-by-turn BPM rows with at least
`name`, `turn`, `x`, `y`, `var_x`, and `var_y`. `calculate_pz` generates the
model optics.
```

AC-dipole workflow is separate from all-BPM reconstruction:

```python
from tmom_recon import ACDipoleConfig, ModelDetails, calculate_acd_pz

acd_result = calculate_acd_pz(
    tracking_df,
    ModelDetails(accelerator=accelerator, pt=pt_offset),
    ACDipoleConfig(
        ac_dipole_marker="MKQA.6L4.B1",
        driven_tunes=(0.27, 0.322),
    ),
    closed_orbit_at_zero=measured_orbit_zero[["x", "y"]],
    orbit_mode="dynamic",
)
```

Both required orbit inputs are explicit. ``dynamic`` restores the complete
generated model state at zero momentum; ``absolute`` restores measured ``x/y``
and generated zero-momentum model ``px/py``. Reference Twiss tables are never
accepted from callers.

That `orbit_zero` must be the on-momentum **closed orbit**. A turn mean of
driven data is a biased estimate of it: over 100 driven turns of LHC B1 the mean
sits 5.3e-5 m rms from the closed orbit, which reaches the reconstructed angles
as a static 1.5e-6 rad per-BPM bias.

The ACD workflow fits `dpx` and `dpy` at the marker itself, treats the marker
position `x/y` as shared across the kick for the same turn, and then
transports the cleaned pre-/post-kick marker states back to the selected
adjacent BPMs.

Measured optics are selected explicitly. Requested measurement categories must
exist; they never silently fall back to model values:

```python
optics = OpticsInput(
    measurement_dir="path/to/omc3",
    sources={"phase": "measurement", "beta": "measurement"},
)
```

Kicker-based single-kick reconstruction:

```python
from tmom_recon import KickerConfig, calculate_kicker_pz

kick = calculate_kicker_pz(
    tracking_df,
    ModelDetails(accelerator=accelerator, pt=pt_offset),
    KickerConfig(kicker="KICKER", n_turns_free=1000),
    closed_orbit_at_zero=measured_orbit_zero[["x", "y"]],
    orbit_mode="dynamic",
)
```

This workflow is for datasets with a single clear kicker excitation. It removes
the frame's measured orbit zero *and* the closed orbit the beam rides at its own
momentum, detects the turn the kicker fired, and solves

```text
x_i = R12_i * dpx,   y_i = R34_i * dpy
```

in the inverse-variance-weighted least-squares sense over *every* BPM of the
first beam pass after the kick — those downstream of the kicker on the kick turn
plus those upstream of it on the next turn, reached by wrapping the phase advance
by the tune.

The result is a one-row frame holding the kicker state on the kick turn, in
frame coordinates: the closed orbit at the kicker plus the kick. On momentum
that is a zero position and the bare kick; at `dp/p = 1e-3` in the PSB the
dispersive angle at the kicker is 43× the kick itself, so it is not optional.
That orbit is taken as the exact difference of two MAD-NG twisses, at `pt` and
at zero — **not** as `pt*D + pt²*D''`, because the reconstruction twiss is
generated *at* `pt`, so its dispersion columns are derivatives about `pt` and
re-expanding from them double-counts. `attrs["kick"]` carries the fit
diagnostics, including the position residual.

The optics must be those *at the kick point*. MAD-NG puts a Twiss row at each
element's exit, while a thick element deflects the beam at its centre, so a
thick kicker needs a marker at its centre in the model. Against PSB tracking a
thin corrector reconstructs the kick to 6.5e-12 rad; a 1.23 m quadrupole, whose
row sits 0.017 turns of phase past the kick point, degrades that to 1.8e-9 and
shows a 6% residual.

Unlike the other two workflows this one takes no `OpticsInput`: the transport it
needs starts *at the kicker*, and an omc3 measurement only provides BPM-to-BPM
phases with an arbitrary origin. Supply a better lattice through
`ModelDetails.magnet_strengths` instead.

Accelerator descriptors for driver setup:

```python
from pymadng_utils.accelerators import LHC

accelerator = LHC(
    beam=1,
    sequence_file="lhcb1.seq",
    kinetic_energy=6800,
)
```

## Testing

Fast unit tests:

```bash
pytest -m "not slow"
```

Full suite:

```bash
pytest
```

Some slow integration tests require the external MAD-NG / xtrack toolchain and
machine sequence data to be available in the active environment.

## Momentum Reconstruction Formulae

For each BPM and one of its neighbors, the code reconstructs the transverse
momenta from the measured positions, Twiss parameters, phase advance, and
optional dispersion terms.

Define the phase advances

```text
\phi_x = 2\pi \Delta_x
\phi_y = 2\pi \Delta_y
```

the normalized coordinates

\[
\tilde x = \frac{x - p_t D_x + p_t^2 D_x^{(2)}}{\sqrt{\beta_x}},
\qquad
\tilde x_n = \frac{x_n - p_t D_{x,n} + p_t^2 D_{x,n}^{(2)}}{\sqrt{\beta_{x,n}}},
\]
\[
\tilde y = \frac{y - p_t D_y + p_t^2 D_y^{(2)}}{\sqrt{\beta_y}},
\qquad
\tilde y_n = \frac{y_n - p_t D_{y,n} + p_t^2 D_{y,n}^{(2)}}{\sqrt{\beta_{y,n}}},
\]

with `p_t` the MAD-NG longitudinal energy coordinate — the dispersion columns
are derivatives with respect to `pt`, never `dp/p`. The second-order sign is
subtracted from the orbit, not added: the reconstruction twiss is generated *at*
`pt`, so `x(pt) - x(0) = pt D - pt^2 D^{(2)}`, and the forward sign is worse
than dropping the term (6.0e-6 m against 3.0e-6 m and 5.5e-8 m for LHC B1 at
`dp/p = 4e-4`). Then the sign convention

```text
s = -1 for previous neighbor
s = +1 for next neighbor

a = +1 for previous neighbor
a = -1 for next neighbor
```

The nominal reconstructed momenta are

```text
p_x =
s * (x~_n sec(\phi_x) + x~ (tan(\phi_x) + a \alpha_x)) / sqrt(\beta_x)
+ D_x' p_t - D_x^{(2)}' p_t^2

p_y =
s * (y~_n sec(\phi_y) + y~ (tan(\phi_y) + a \alpha_y)) / sqrt(\beta_y)
+ D_y' p_t - D_y^{(2)}' p_t^2
```

The measurement-only variances are

```text
var_meas(p_x) =
\sigma^2_{x_n} * (s sec(\phi_x) / (sqrt(\beta_x) sqrt(\beta_{x,n})))^2
+ \sigma^2_x * (s (tan(\phi_x) + a \alpha_x) / \beta_x)^2

var_meas(p_y) =
\sigma^2_{y_n} * (s sec(\phi_y) / (sqrt(\beta_y) sqrt(\beta_{y,n})))^2
+ \sigma^2_y * (s (tan(\phi_y) + a \alpha_y) / \beta_y)^2
```

When optics uncertainties are enabled, the code adds the usual linear
propagation terms

```text
var_opt(p_x) = \sum_i \sigma_i^2 ( \partial p_x / \partial q_i )^2
q_i in {D_{x,n}, D_x, D_x', \alpha_x, sqrt(\beta_x), sqrt(\beta_{x,n}), \Delta_x}

var_opt(p_y) = \sum_i \sigma_i^2 ( \partial p_y / \partial q_i )^2
q_i in {D_{y,n}, D_y, D_y', \alpha_y, sqrt(\beta_y), sqrt(\beta_{y,n}), \Delta_y}
```

The total variances reported by the reconstruction are

```text
var(p_x) = var_meas(p_x) + var_opt(p_x)
var(p_y) = var_meas(p_y) + var_opt(p_y)
```

Finally, the previous- and next-neighbor estimates are combined by
inverse-variance weighting:

```text
p^ = (p_a / \sigma_a^2 + p_b / \sigma_b^2) / (1 / \sigma_a^2 + 1 / \sigma_b^2)
var(p^) = 1 / (1 / \sigma_a^2 + 1 / \sigma_b^2)
```

## Development

Install the editable development environment first:

```bash
python -m pip install -e '.[dev,test,docs]'
```

Set up hooks:

```bash
pre-commit install
```

Run the test suite:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

Build the documentation:

```bash
python -m pip install -e '.[docs]'
sphinx-build -b html docs docs/_build/html
```

Published documentation:

- https://jgray-19.github.io/tmom-recon/

Deploy the documentation with GitHub Pages:

- the repository includes [`.github/workflows/docs-pages.yml`](/afs/cern.ch/work/j/jmgray/private/tmom-recon/.github/workflows/docs-pages.yml)
- GitHub Pages should be configured to deploy from `GitHub Actions`
- pushes to `main` trigger a fresh docs build and publish
