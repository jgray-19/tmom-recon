# tmom-recon
[![codecov](https://codecov.io/gh/jgray-19/tmom-recon/graph/badge.svg?token=1R2UUJGSP3)](https://codecov.io/gh/jgray-19/tmom-recon)

Momentum reconstruction utilities for turn-by-turn BPM data.

The package now bundles the core two-BPM reconstruction formulae together with
higher-level workflows for dispersive momentum estimation, n-BPM BLUE
combination, AC-dipole reconstruction, lattice helpers, and accelerator
descriptors used by the MAD-NG drivers.

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
    inject_noise_xy_inplace,
)
```

Main modules:

- `tmom_recon.physics`: two-BPM transverse and dispersive momentum formulae.
- `tmom_recon.measurements`: measured `delta p / p` and Twiss reconstruction helpers.
- `tmom_recon.nbpm`: n-BPM transverse reconstruction.
- `tmom_recon.acd`: AC-dipole reconstruction and MAD-NG integration helpers.
- `tmom_recon.kalman`: Kalman-based reconstruction utilities.
- `tmom_recon.lattice`: neighbor and lattice helper functions.

## Usage

Two-BPM transverse reconstruction:

```python
from tmom_recon import calculate_transverse_pz

result = calculate_transverse_pz(
    tracking_df,
    twiss_df,
)
```

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
from tmom_recon import calculate_ac_dipole_momentum

acd_result = calculate_ac_dipole_momentum(
    tracking_df,
    twiss_df,
    ac_dipole_marker="MKQA.6L4.B1",
    model=acd_model,
)
```

Accelerator descriptors for driver setup:

```python
from pymadng_utils.accelerators import LHC

accelerator = LHC(
    beam=1,
    sequence_file="lhcb1.seq",
    kinetic_energy=6800,
)
```

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
python -m pip install -e '.[dev,test]'
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
