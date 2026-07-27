"""
Pairwise (composite) likelihood fitting.

The full joint density of a max-stable process is analytically
intractable beyond a handful of sites, so parameters are estimated by
maximizing the sum of *pairwise* log-densities across all site pairs
and replicates instead -- this is the standard approach in the
spatial-extremes literature (Padoan, Ribatet & Sisson 2010) and is
what R's `SpatialExtremes::fitmaxstab` does under the hood.

Works with any model exposing:
  - `bivariate_logpdf(z1, z2, h)` (vectorized over replicates)
  - `from_free_params(free_params)` classmethod
  - `n_free_params` class attribute

i.e. SmithModel, SchlatherModel, BrownResnickModel all plug in directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import optimize

try:
    import polars as pl
except ImportError:
    pl = None


def _to_numpy(df):
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_numpy()
    return df.to_numpy()


@dataclass
class PairwiseFitResult:
    model: object
    free_params: np.ndarray
    nllh: float
    converged: bool
    n_pairs_used: int
    n_replicates: int
    optimizer_result: object


def fit_pairwise_likelihood(
    model_class,
    frechet_df,
    coords: np.ndarray,
    x0_free: np.ndarray | None = None,
    max_pairs: int | None = None,
    method: str = "Nelder-Mead",
    rng: np.random.Generator | None = None,
) -> PairwiseFitResult:
    """Fit a max-stable model by maximizing the pairwise log-likelihood.

    Parameters
    ----------
    model_class : SmithModel, SchlatherModel, or BrownResnickModel (the
        class itself, not an instance -- it's constructed internally at
        each optimizer step via `model_class.from_free_params(...)`).
    frechet_df : pandas or polars DataFrame, columns = sites (unit-Frechet
        margins already), rows = replicates.
    coords : (D, 2) array_like of site coordinates, same order as columns.
    x0_free : optional starting point in free-parameter space; defaults
        to a vector of zeros (which is a reasonable "unit scale, no
        anisotropy" starting guess under each model's reparametrization).
    max_pairs : if the number of sites is large, randomly subsample this
        many site pairs per optimizer evaluation (fixed once, not
        resampled per iteration) to keep fitting tractable. `None` uses
        all pairs.
    """
    values = _to_numpy(frechet_df)
    n_obs, D = values.shape
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] != D:
        raise ValueError(f"coords has {coords.shape[0]} rows but frechet_df has {D} columns")

    all_pairs = list(combinations(range(D), 2))
    if max_pairs is not None and max_pairs < len(all_pairs):
        rng = rng or np.random.default_rng()
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[k] for k in idx]
    else:
        pairs = all_pairs

    if x0_free is None:
        x0_free = np.zeros(model_class.n_free_params)

    def neg_composite_llh(free_params):
        try:
            model = model_class.from_free_params(free_params)
        except (ValueError, np.linalg.LinAlgError):
            return 1e12
        total = 0.0
        for i, j in pairs:
            h = coords[i] - coords[j]
            logpdf = model.bivariate_logpdf(values[:, i], values[:, j], h)
            if not np.all(np.isfinite(logpdf)):
                return 1e12
            total += np.sum(logpdf)
        return -total

    result = optimize.minimize(
        neg_composite_llh, x0=np.asarray(x0_free, dtype=float), method=method,
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6}
        if method == "Nelder-Mead" else {"maxiter": 5000},
    )

    fitted_model = model_class.from_free_params(result.x)
    return PairwiseFitResult(
        model=fitted_model, free_params=result.x, nllh=result.fun,
        converged=bool(result.success), n_pairs_used=len(pairs),
        n_replicates=n_obs, optimizer_result=result,
    )
