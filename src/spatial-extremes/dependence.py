"""
Model-free spatial dependence diagnostics for max-stable data.

These work directly on unit-Frechet-transformed observations and let
you sanity-check a fitted model (e.g. Smith) against the empirical
dependence structure before trusting parameter estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None


def _to_numpy(df) -> tuple[np.ndarray, list[str]]:
    """Return (values as (n_obs, D) ndarray, column names) from pandas or polars."""
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_numpy(), df.columns
    return df.to_numpy(), list(df.columns)


def fmadogram(frechet_df, coords: np.ndarray) -> pd.DataFrame:
    """F-madogram: pairwise dependence summary on unit-Frechet margins.

        nu_F(h) = 0.5 * E[ |F(Z(s)) - F(Z(s+h))| ]

    where F is the unit-Frechet CDF, so F(Z(s)) is uniform on [0,1]
    marginally. Related to the extremal coefficient by:

        theta(h) = (1 + 2*nu_F(h)) / (1 - 2*nu_F(h))

    Parameters
    ----------
    frechet_df : pandas or polars DataFrame, columns = sites, rows = replicates,
        already on unit-Frechet margins (see gev.GEVResult.to_unit_frechet).
    coords : (D, 2) array_like of site coordinates, same order as columns.

    Returns
    -------
    pandas DataFrame with one row per site pair: site_i, site_j, distance,
    fmadogram, extremal_coefficient.
    """
    values, cols = _to_numpy(frechet_df)
    coords = np.asarray(coords, dtype=float)
    n, D = values.shape
    if coords.shape[0] != D:
        raise ValueError(f"coords has {coords.shape[0]} rows but frechet_df has {D} columns")

    # Uniform-margin transform via unit-Frechet CDF: F(z) = exp(-1/z)
    u = np.exp(-1.0 / values)

    rows = []
    for i in range(D):
        for j in range(i + 1, D):
            nu = 0.5 * np.mean(np.abs(u[:, i] - u[:, j]))
            theta = (1 + 2 * nu) / (1 - 2 * nu) if nu < 0.5 else 2.0
            dist = np.linalg.norm(coords[i] - coords[j])
            rows.append({
                "site_i": cols[i], "site_j": cols[j],
                "distance": dist, "fmadogram": nu,
                "extremal_coefficient": theta,
            })
    return pd.DataFrame(rows)


def empirical_extremal_coefficient(frechet_df, coords: np.ndarray) -> pd.DataFrame:
    """Empirical extremal coefficient via the madogram-based estimator.

    Thin wrapper around fmadogram() returning just the columns most
    people plot (distance vs extremal_coefficient), useful for
    overlaying against a fitted model's `extremal_coefficient(h)`.
    """
    result = fmadogram(frechet_df, coords)
    return result[["site_i", "site_j", "distance", "extremal_coefficient"]]


def binned_extremal_coefficient(frechet_df, coords: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin pairwise extremal coefficients by distance -- handy for plotting
    a smooth empirical curve against a fitted theoretical one."""
    ec = empirical_extremal_coefficient(frechet_df, coords)
    ec["dist_bin"] = pd.cut(ec["distance"], bins=n_bins)
    summary = (
        ec.groupby("dist_bin", observed=True)
        .agg(mean_distance=("distance", "mean"),
             mean_extremal_coefficient=("extremal_coefficient", "mean"),
             n_pairs=("extremal_coefficient", "size"))
        .reset_index(drop=True)
    )
    return summary
