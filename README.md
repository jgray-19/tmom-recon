# tmom-recon
[![codecov](https://codecov.io/gh/jgray-19/tmom-recon/graph/badge.svg?token=1R2UUJGSP3)](https://codecov.io/gh/jgray-19/tmom-recon)
[![Coverage](https://github.com/jgray-19/tmom-recon/actions/workflows/coverage.yml/badge.svg)](https://github.com/jgray-19/tmom-recon/actions/workflows/coverage.yml)

Momentum reconstruction utilities for turn-by-turn BPM data.

The package now bundles the core two-BPM reconstruction formulae together with
higher-level workflows for dispersive momentum estimation, n-BPM BLUE
combination, AC-dipole reconstruction, lattice helpers, and accelerator
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
    build_twiss_from_measurements,
    calculate_ac_dipole_momentum,
    calculate_dispersive_pz,
    calculate_pz_measurement,
    calculate_transverse_pz,
    calculate_transverse_pz_nbpm,
    inject_noise_xy,
)
```

Main modules:

- `tmom_recon.physics`: two-BPM transverse and dispersive momentum formulae.
- `tmom_recon.measurements`: measured `delta p / p` and Twiss reconstruction helpers.
- `tmom_recon.nbpm`: n-BPM transverse reconstruction.
- `tmom_recon.acd`: AC-dipole reconstruction, BPM override, and MAD-NG integration helpers.
- `tmom_recon.kicker`: single-kick reconstruction helpers based on kicker-to-BPM transport.
- `tmom_recon.kalman`: Kalman-based reconstruction utilities.
- `tmom_recon.lattice`: neighbor, lattice, and transport-matrix helper functions.

## Usage

Two-BPM transverse reconstruction:

```python
from tmom_recon import calculate_transverse_pz

result = calculate_transverse_pz(
    tracking_df,
    twiss_df,
)
```

`tracking_df` is expected to contain turn-by-turn BPM rows with at least
`name`, `turn`, `x`, `y`, `var_x`, and `var_y`. The Twiss frame is expected to
be indexed by BPM element name.

Dispersive momentum reconstruction:

```python
from tmom_recon import calculate_dispersive_pz

dp_over_p = calculate_dispersive_pz(
    tracking_df,
    twiss_df,
)
```

n-BPM combination:

```python
from tmom_recon import calculate_transverse_pz_nbpm

nbpm_result = calculate_transverse_pz_nbpm(
    tracking_df,
    twiss_df,
)
```

AC-dipole workflow:

```python
from tmom_recon import ACDipoleConfig, calculate_ac_dipole_momentum

acd_result = calculate_ac_dipole_momentum(
    tracking_df,
    twiss_df,
    ac_dipole_marker="MKQA.6L4.B1",
    model=acd_model,
    dpx_tune=0.27,
    dpy_tune=0.322,
)
```

The `model` object must provide the MAD-NG tracking interface used by
`tmom_recon.acd.madng_driver.ACDipoleMadDriver`. If you are integrating the
result back into transverse or n-BPM reconstruction, use `ACDipoleConfig`
through the higher-level APIs rather than wiring the BPM overrides yourself.

The ACD workflow fits `dpx` and `dpy` at the marker itself, treats the marker
position `x/y` as shared across the kick for the same turn, and then
transports the cleaned pre-/post-kick marker states back to the selected
adjacent BPMs.

Kicker-based single-turn reconstruction:

```python
from tmom_recon.kicker.core import reconstruct_momentum_kick

kicker_result = reconstruct_momentum_kick(
    tracking_df,
    twiss_df,
    n_turns_free=1000,
    n_turns_after_kick=3,
)
```

This lower-level helper is intended for datasets with a single clear kicker
excitation. It removes the closed orbit, identifies the kick turn, and solves
the kicker-to-BPM transport equation using the Twiss-parameterized transfer
matrix between the kicker and the first downstream BPM response.

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
\tilde x = \frac{x - \delta D_x}{\sqrt{\beta_x}},
\qquad
\tilde x_n = \frac{x_n - \delta D_{x,n}}{\sqrt{\beta_{x,n}}},
\]
\[
\tilde y = \frac{y - \delta D_y}{\sqrt{\beta_y}},
\qquad
\tilde y_n = \frac{y_n - \delta D_{y,n}}{\sqrt{\beta_{y,n}}},
\]

with `\delta = \Delta p / p`, and the sign convention

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
+ D_x' \delta

p_y =
s * (y~_n sec(\phi_y) + y~ (tan(\phi_y) + a \alpha_y)) / sqrt(\beta_y)
+ D_y' \delta
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
