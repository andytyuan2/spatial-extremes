"""
Univariate Generalized Extreme Value (GEV) fitting.

This is the marginal building block for max-stable processes: every
location in a max-stable random field has GEV-distributed marginals.
Fit each site's block maxima independently here, then use the fitted
(mu, sigma, xi) to transform to unit Frechet margins before modeling
spatial dependence (see `transform.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats

try:
    import polars as pl
except ImportError:  # polars is optional
    pl = None


@dataclass
class GEVResult:
    """Fitted GEV parameters and diagnostics for a single site/column."""

    loc: float          # mu
    scale: float         # sigma (> 0)
    shape: float          # xi (0 => Gumbel limit)
    loc_se: float
    scale_se: float
    shape_se: float
    nllh: float           # negative log-likelihood at the optimum
    converged: bool
    n_obs: int

    def as_dict(self) -> dict:
        return {
            "loc": self.loc, "scale": self.scale, "shape": self.shape,
            "loc_se": self.loc_se, "scale_se": self.scale_se,
            "shape_se": self.shape_se, "nllh": self.nllh,
            "converged": self.converged, "n_obs": self.n_obs,
        }

    def cdf(self, x):
        return stats.genextreme.cdf(x, c=-self.shape, loc=self.loc, scale=self.scale)

    def to_unit_frechet(self, x):
        """Probability-integral-transform x onto unit Frechet margins."""
        u = self.cdf(np.asarray(x, dtype=float))
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return -1.0 / np.log(u)


def _neg_log_lik(params: np.ndarray, x: np.ndarray) -> float:
    """Negative log-likelihood of the GEV.

    Note: scipy.stats.genextreme uses the sign convention c = -xi
    (its 'c' is the negative of the standard extreme-value shape xi).
    We optimize in terms of the standard xi and flip sign only when
    calling scipy's genextreme functions.
    """
    mu, log_sigma, xi = params
    sigma = np.exp(log_sigma)  # keep sigma > 0 unconstrained in optimizer
    z = (x - mu) / sigma

    if abs(xi) < 1e-8:
        # Gumbel limit
        t = np.exp(-z)
        if not np.all(np.isfinite(t)):
            return 1e10
        nllh = np.sum(np.log(sigma) + z + t)
        return nllh

    support = 1 + xi * z
    if np.any(support <= 0):
        return 1e10  # outside GEV support -> invalid parameters

    t = support ** (-1.0 / xi)
    nllh = np.sum(np.log(sigma) + (1 + 1.0 / xi) * np.log(support) + t)
    return nllh if np.isfinite(nllh) else 1e10


def fit_gev(x, x0: tuple[float, float, float] | None = None) -> GEVResult:
    """Fit a GEV distribution to a 1-D array of block maxima via MLE.

    Parameters
    ----------
    x : array-like
        Block maxima (e.g. annual maxima at one site).
    x0 : optional initial guess (loc, scale, shape). If None, a
        method-of-moments-ish guess is derived from the data.

    Returns
    -------
    GEVResult
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        raise ValueError(f"Need at least ~10 observations for a stable GEV fit, got {n}")

    if x0 is None:
        loc0 = np.mean(x) - 0.5772 * np.std(x) * np.sqrt(6) / np.pi
        scale0 = np.std(x) * np.sqrt(6) / np.pi
        shape0 = 0.0
        x0 = (loc0, np.log(max(scale0, 1e-3)), shape0)
    else:
        x0 = (x0[0], np.log(max(x0[1], 1e-6)), x0[2])

    result = optimize.minimize(
        _neg_log_lik, x0=np.array(x0), args=(x,), method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
    )

    mu, log_sigma, xi = result.x
    sigma = np.exp(log_sigma)

    # Standard errors via numerical Hessian of the (sigma-not-log) nllh
    def nllh_raw(p):
        mu_, sigma_, xi_ = p
        if sigma_ <= 0:
            return 1e10
        return _neg_log_lik(np.array([mu_, np.log(sigma_), xi_]), x)

    se = _numerical_se(nllh_raw, np.array([mu, sigma, xi]))

    return GEVResult(
        loc=mu, scale=sigma, shape=xi,
        loc_se=se[0], scale_se=se[1], shape_se=se[2],
        nllh=result.fun, converged=bool(result.success), n_obs=n,
    )


def _numerical_se(fun, p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Standard errors from the inverse of a numerically-differenced Hessian."""
    k = len(p)
    hess = np.zeros((k, k))
    f0 = fun(p)
    for i in range(k):
        for j in range(k):
            pp = p.copy(); pp[i] += eps; pp[j] += eps
            pm = p.copy(); pm[i] += eps; pm[j] -= eps
            mp = p.copy(); mp[i] -= eps; mp[j] += eps
            mm = p.copy(); mm[i] -= eps; mm[j] -= eps
            hess[i, j] = (fun(pp) - fun(pm) - fun(mp) + fun(mm)) / (4 * eps ** 2)
    try:
        cov = np.linalg.inv(hess)
        diag = np.diag(cov)
        return np.sqrt(np.where(diag > 0, diag, np.nan))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


def fit_gev_by_site(df, site_cols: list[str] | None = None) -> pd.DataFrame:
    """Fit a GEV independently to each site (column) of block maxima.

    Accepts a pandas or polars DataFrame where each column is a site's
    time series of block maxima (e.g. one row per year).

    Returns a tidy pandas DataFrame, one row per site, with fitted
    parameters and diagnostics -- convenient for joining onto a
    GeoDataFrame of site coordinates afterwards.
    """
    if pl is not None and isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df

    cols = site_cols if site_cols is not None else list(pdf.columns)
    rows = []
    for col in cols:
        res = fit_gev(pdf[col].to_numpy())
        row = {"site": col, **res.as_dict()}
        rows.append(row)
    return pd.DataFrame(rows)
