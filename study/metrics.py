"""Reconstruction + scoring helpers for the limitations study.

Thin wrappers around :func:`tmom_recon.calculate_pz` that take MAD-NG tracking
truth + a model twiss, optionally add BPM noise / SVD cleaning / a distorted
model, and return the per-plane RMSE and R^2 of the reconstructed momenta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tmom_recon import calculate_pz, inject_noise_xy
from tmom_recon.svd import svd_clean_measurements, weighted_svd_clean_measurements

# A small nominal BPM variance floor so the inverse-variance weighting is well
# defined even for noise-free runs (value in m^2).
VAR_FLOOR = (1e-5) ** 2


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(predicted) - np.asarray(actual)) ** 2)))


def r_squared(true: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _prepare_data(
    tracking_df: pd.DataFrame,
    *,
    noise_std: float,
    rng: np.random.Generator | None,
    svd: bool,
) -> pd.DataFrame:
    """Build the ``name, turn, x, y, var_x, var_y`` input, optionally noisy/SVD-cleaned."""
    data = tracking_df[["name", "turn", "x", "y"]].copy()
    if noise_std > 0:
        rng = np.random.default_rng() if rng is None else rng
        data = inject_noise_xy(data, rng, noise_std=noise_std)
        if svd:
            data = svd_clean_measurements(data)
    var = max(noise_std, np.sqrt(VAR_FLOOR)) ** 2
    data["var_x"] = var
    data["var_y"] = var
    return data


def reconstruct(
    tracking_df: pd.DataFrame,
    model_tws: pd.DataFrame,
    *,
    use_dispersion: bool = False,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    svd: bool = False,
    pt_override: float | None = None,
) -> pd.DataFrame:
    """Reconstruct momenta and merge with the tracked truth.

    Returns the truth merged with reconstruction, columns
    ``name, turn, px_true, py_true, px, py``.
    """
    data = _prepare_data(tracking_df, noise_std=noise_std, rng=rng, svd=svd)
    result = calculate_pz(
        data,
        model_tws=model_tws,
        use_dispersion=use_dispersion,
        pt_override=pt_override,
        info=False,
    )
    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )
    return truth.merge(result[["name", "turn", "px", "py"]], on=["name", "turn"])


def score(merged: pd.DataFrame) -> dict[str, float]:
    """Per-plane RMSE and R^2 from a merged truth/reconstruction frame."""
    return {
        "px_rmse": rmse(merged["px_true"], merged["px"]),
        "py_rmse": rmse(merged["py_true"], merged["py"]),
        "px_r2": r_squared(merged["px_true"], merged["px"]),
        "py_r2": r_squared(merged["py_true"], merged["py"]),
        "px_scale": float(merged["px_true"].std()),
        "py_scale": float(merged["py_true"].std()),
    }


def per_bpm_rmse(merged: pd.DataFrame, tws: pd.DataFrame) -> pd.DataFrame:
    """Per-BPM RMSE vs longitudinal position ``s`` (for around-the-ring plots)."""
    rows = (
        merged.groupby("name")
        .apply(
            lambda g: pd.Series(
                {"px_rmse": rmse(g["px_true"], g["px"]), "py_rmse": rmse(g["py_true"], g["py"])}
            ),
            include_groups=False,
        )
        .reset_index()
    )
    rows["s"] = rows["name"].map(tws["s"])
    return rows.sort_values("s").reset_index(drop=True)


def draw_per_bpm_sigma(
    names: np.ndarray,
    rng: np.random.Generator,
    *,
    median: float,
    spread: float,
    bad_fraction: float = 0.0,
    bad_factor: float = 5.0,
) -> dict[str, float]:
    """Draw a per-BPM noise sigma [m], one value per unique BPM name.

    Models the heteroscedastic reality measured on the LHC (see
    ``noise_investigation/bpm_std.txt``): most BPMs cluster near a ``median``
    resolution, drawn log-normally with multiplicative ``spread`` (the geometric
    sigma, so ``spread=1`` is uniform), plus a tail of ``bad_fraction`` noisy BPMs
    scaled up by ``bad_factor``. This spread is exactly what a *weighted* SVD can
    exploit and a plain SVD cannot.
    """
    unique = np.unique(names)
    sigma = median * np.exp(rng.normal(0.0, np.log(spread) if spread > 1 else 0.0, len(unique)))
    if bad_fraction > 0:
        bad = rng.random(len(unique)) < bad_fraction
        sigma[bad] *= bad_factor
    return dict(zip(unique, sigma, strict=True))


def inject_per_bpm_noise(
    tracking_df: pd.DataFrame,
    rng: np.random.Generator,
    per_bpm_sigma: dict[str, float],
) -> pd.DataFrame:
    """Add zero-mean Gaussian noise with a per-BPM sigma; attach known variances.

    Returns ``name, turn, x, y, var_x, var_y`` where ``var_x = var_y`` are the
    (known) per-BPM variances the weighted SVD uses to whiten each column.
    """
    data = tracking_df[["name", "turn", "x", "y"]].copy()
    sig = data["name"].map(per_bpm_sigma).to_numpy(dtype=float)
    data["x"] = data["x"].to_numpy() + rng.normal(0.0, sig)
    data["y"] = data["y"].to_numpy() + rng.normal(0.0, sig)
    data["var_x"] = sig**2
    data["var_y"] = sig**2
    return data


def _clean(data: pd.DataFrame, method: str) -> pd.DataFrame:
    """Dispatch BPM cleaning: ``"none"``, ``"svd"`` (plain) or ``"wsvd"`` (weighted)."""
    if method == "none":
        return data
    if method == "svd":
        return svd_clean_measurements(data)
    if method == "wsvd":
        return weighted_svd_clean_measurements(data)
    raise ValueError(f"unknown cleaning method {method!r}")


def reconstruct_cleaned(
    tracking_df: pd.DataFrame,
    model_tws: pd.DataFrame,
    data: pd.DataFrame,
    *,
    method: str,
    use_dispersion: bool = False,
    pt_override: float | None = None,
) -> pd.DataFrame:
    """Clean a prepared noisy ``data`` frame, reconstruct, merge with truth.

    ``data`` must carry ``name, turn, x, y, var_x, var_y`` (e.g. from
    :func:`inject_per_bpm_noise`). The variance columns are preserved through
    cleaning so the reconstruction's inverse-variance weighting stays consistent.
    """
    cleaned = _clean(data, method)
    if "var_x" not in cleaned.columns:
        cleaned = cleaned.merge(data[["name", "turn", "var_x", "var_y"]], on=["name", "turn"])
    result = calculate_pz(
        cleaned,
        model_tws=model_tws,
        use_dispersion=use_dispersion,
        pt_override=pt_override,
        info=False,
    )
    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )
    return truth.merge(result[["name", "turn", "px", "py"]], on=["name", "turn"])


def per_bpm_method_rmse(
    tracking_df: pd.DataFrame,
    model_tws: pd.DataFrame,
    methods: list[str],
    *,
    median: float,
    spread: float,
    bad_fraction: float,
    bad_factor: float,
    seeds,
    use_dispersion: bool = False,
) -> pd.DataFrame:
    """Per-BPM reconstruction RMSE around the full ring, per cleaning method.

    For each seed a per-BPM heteroscedastic noise is injected and each method in
    ``methods`` (``"none"``/``"svd"``/``"wsvd"``) is reconstructed; the per-BPM
    squared errors are pooled over all seeds and turns, so the returned RMSE is a
    seed-and-turn average. Also returns the mean injected sigma per BPM.

    Returns a frame indexed by BPM with columns ``s``, ``sigma`` and, per method,
    ``<method>_px_rmse`` / ``<method>_py_rmse``.
    """
    sq = {m: {"px": {}, "py": {}} for m in methods}  # name -> list of squared resid
    sig_acc: dict[str, list[float]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sigma = draw_per_bpm_sigma(
            tracking_df["name"].to_numpy(),
            rng,
            median=median,
            spread=spread,
            bad_fraction=bad_fraction,
            bad_factor=bad_factor,
        )
        for name, s in sigma.items():
            sig_acc.setdefault(name, []).append(s)
        data = inject_per_bpm_noise(tracking_df, rng, sigma)
        for method in methods:
            merged = reconstruct_cleaned(
                tracking_df, model_tws, data, method=method, use_dispersion=use_dispersion
            )
            for plane in ("px", "py"):
                resid2 = (merged[plane].to_numpy() - merged[f"{plane}_true"].to_numpy()) ** 2
                for name, val in zip(merged["name"].to_numpy(), resid2, strict=True):
                    sq[method][plane].setdefault(name, []).append(val)

    names = sorted(sig_acc, key=lambda n: float(model_tws.loc[n, "s"]))
    rows = []
    for name in names:
        row = {
            "name": name,
            "s": float(model_tws.loc[name, "s"]),
            "sigma": float(np.mean(sig_acc[name])),
        }
        for method in methods:
            for plane in ("px", "py"):
                vals = sq[method][plane].get(name, [])
                row[f"{method}_{plane}_rmse"] = float(np.sqrt(np.mean(vals))) if vals else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("name")


def distort_optics(
    tws: pd.DataFrame,
    rng: np.random.Generator,
    *,
    beta_rel: float = 0.0,
    alfa_rel: float = 0.0,
    phase_abs: float = 0.0,
    disp_rel: float = 0.0,
) -> pd.DataFrame:
    """Return a copy of ``tws`` with random per-BPM optics distortions applied.

    Models the "we never know the optics perfectly" limitation: the lattice that
    is *tracked* is untouched, but the *model* handed to the reconstruction has
    beta-beating (``beta_rel``), alpha errors (``alfa_rel``), random phase errors
    (``phase_abs`` in units of 2*pi*tune) and dispersion errors (``disp_rel``).
    """
    out = tws.copy()
    n = len(out)
    if beta_rel:
        out["beta11"] = out["beta11"] * (1 + rng.normal(0, beta_rel, n))
        out["beta22"] = out["beta22"] * (1 + rng.normal(0, beta_rel, n))
    if alfa_rel:
        out["alfa11"] = out["alfa11"] + rng.normal(0, alfa_rel, n) * np.abs(out["alfa11"]).mean()
        out["alfa22"] = out["alfa22"] + rng.normal(0, alfa_rel, n) * np.abs(out["alfa22"]).mean()
    if phase_abs:
        out["mu1"] = out["mu1"] + np.cumsum(rng.normal(0, phase_abs, n))
        out["mu2"] = out["mu2"] + np.cumsum(rng.normal(0, phase_abs, n))
    if disp_rel:
        for col in ("dx", "dpx", "dy", "dpy"):
            if col in out:
                scale = np.abs(out[col]).mean() or 1.0
                out[col] = out[col] + rng.normal(0, disp_rel, n) * scale
    return out
