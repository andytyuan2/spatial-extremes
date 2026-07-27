"""
Shared bivariate CDF/density for the "de Haan" exponent-function family
that both the Smith and Brown-Resnick models reduce to:

    V(z1, z2) = Phi(w)/z1 + Phi(v)/z2
    w = a/2 + log(z2/z1)/a,   v = a - w
    CDF(z1, z2) = exp(-V(z1, z2))

Only the definition of the single dependence parameter `a` differs:
  - Smith:          a = sqrt(h' Sigma^-1 h)   (Mahalanobis distance)
  - Brown-Resnick:  a = sqrt(2 * gamma(h))    (variogram-based)

The bivariate density is derived from V via the standard multivariate
extreme-value identity:

    f(z1, z2) = exp(-V) * (V1 * V2 - V12)

where V1 = dV/dz1, V2 = dV/dz2, V12 = d^2V/dz1 dz2. Verified
symbolically with sympy against finite-difference derivatives of the
CDF (see project notes) -- closed forms below are:

    V1  = -Phi(w) / z1^2
    V2  = -Phi(v) / z2^2
    V12 = -phi(w) / (a * z1^2 * z2)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def de_haan_bivariate_cdf(z1, z2, a: float) -> float:
    if a <= 0:
        # a -> 0 is the full-dependence limit
        return float(np.exp(-1.0 / min(z1, z2)))
    w = a / 2 + np.log(z2 / z1) / a
    v = a - w
    V = stats.norm.cdf(w) / z1 + stats.norm.cdf(v) / z2
    return float(np.exp(-V))


def de_haan_extremal_coefficient(a: float) -> float:
    """theta = 2 * Phi(a/2); shared closed form for Smith and Brown-Resnick."""
    return float(2 * stats.norm.cdf(a / 2))


def de_haan_bivariate_logpdf(z1: np.ndarray, z2: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Vectorized log bivariate density on unit-Frechet margins.

    z1, z2, a broadcast together (a is typically per-pair, constant
    across the replicate/row dimension of z1, z2).
    """
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    a = np.asarray(a, dtype=float)
    a = np.where(a <= 1e-8, 1e-8, a)  # avoid division by zero at h=0

    w = a / 2 + np.log(z2 / z1) / a
    v = a - w
    Phi_w = stats.norm.cdf(w)
    Phi_v = stats.norm.cdf(v)
    phi_w = stats.norm.pdf(w)

    V = Phi_w / z1 + Phi_v / z2
    V1 = -Phi_w / z1 ** 2
    V2 = -Phi_v / z2 ** 2
    V12 = -phi_w / (a * z1 ** 2 * z2)

    density_factor = V1 * V2 - V12
    density_factor = np.where(density_factor > 0, density_factor, 1e-300)
    return -V + np.log(density_factor)
