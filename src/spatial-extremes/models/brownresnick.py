"""
Brown-Resnick max-stable process (Kabluchko, Schlather & de Haan, 2009).

Spectral representation:

    Z(s) = max_i  xi_i * exp{ eps_i(s) - gamma(s - o) }

where eps(s) is a Gaussian process with stationary increments,
eps(o) = 0 at an arbitrary anchor point o, and semivariogram
gamma(h) = 0.5 * Var(eps(s+h) - eps(s)). Anchoring at o gives
Var(eps(s)) = 2*gamma(s - o), and the normalization
E[exp(eps(s) - gamma(s-o))] = exp(-gamma(s-o) + Var(eps(s))/2)
= exp(-gamma(s-o) + gamma(s-o)) = 1, which is exactly what's needed
for unit-Frechet margins. As with Schlather, no spatial windowing is
needed (each "storm" is a full field over all sites), only the
truncation of the (in principle infinite) sequence of Poisson points.

For any two sites, this model shares the exact same bivariate CDF
functional form as the Smith model (see _shared.py), with the single
dependence parameter a = sqrt(2 * gamma(h)) in place of Smith's
Mahalanobis distance. Brown-Resnick is, in this sense, a generalization
of Smith's model to non-Gaussian-kernel dependence structures; Smith's
model is recovered as a special (degenerate) case for particular
choices of gamma.

The classic choice of semivariogram is the power/fractal variogram
gamma(h) = (|h| / range)^alpha, alpha in (0, 2] -- alpha=2 gives a
smooth (differentiable) field, alpha=1 gives Brownian-motion-like
roughness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ._shared import de_haan_bivariate_cdf, de_haan_bivariate_logpdf, de_haan_extremal_coefficient


def power_variogram(h: np.ndarray, range_: float, alpha: float = 1.0) -> np.ndarray:
    """gamma(h) = (|h| / range)^alpha, the standard fractal/power variogram."""
    d = np.linalg.norm(h, axis=-1) if np.ndim(h) > 1 else np.linalg.norm(h)
    return (d / range_) ** alpha


@dataclass
class BrownResnickModel:
    """Brown-Resnick process with a power-law semivariogram.

    Parameters
    ----------
    range_ : float
        Variogram range (> 0).
    alpha : float
        Variogram smoothness exponent, in (0, 2].
    variogram_fn : callable, optional
        Override with a custom gamma(h, range_, alpha) -> float >= 0.
    """

    range_: float
    alpha: float = 1.0
    variogram_fn: Callable = power_variogram

    n_free_params = 2
    free_param_names = ("log_range", "logit_alpha")

    @classmethod
    def from_free_params(cls, free_params, **kwargs) -> "BrownResnickModel":
        log_range, logit_alpha = free_params
        range_ = np.exp(log_range)
        alpha = 2.0 / (1.0 + np.exp(-logit_alpha))
        return cls(range_=range_, alpha=alpha, **kwargs)

    def __post_init__(self):
        if self.range_ <= 0:
            raise ValueError("range_ must be positive")
        if not (0 < self.alpha <= 2):
            raise ValueError("alpha must be in (0, 2]")

    def _gamma(self, h: np.ndarray) -> float:
        h = np.asarray(h, dtype=float)
        return float(self.variogram_fn(h, self.range_, self.alpha))

    def _a(self, h: np.ndarray) -> float:
        return float(np.sqrt(2 * self._gamma(h)))

    def extremal_coefficient(self, h: np.ndarray) -> float:
        return de_haan_extremal_coefficient(self._a(h))

    def bivariate_cdf(self, z1: float, z2: float, h: np.ndarray) -> float:
        return de_haan_bivariate_cdf(z1, z2, self._a(h))

    def bivariate_logpdf(self, z1, z2, h: np.ndarray):
        return de_haan_bivariate_logpdf(z1, z2, self._a(h))

    def simulate(
        self,
        coords: np.ndarray,
        n_reps: int = 1,
        max_points: int = 20_000,
        batch_size: int = 256,
        practical_bound: float = 6.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate on unit-Frechet margins at given sites.

        Points are generated in vectorized batches (rather than one
        Python-level iteration per point) since exp(eps - gamma) has a
        much heavier tail than Smith's bounded kernel, so many more
        points are typically needed before the stopping rule triggers.

        Parameters
        ----------
        coords : (D, 2) array_like
        n_reps : int
        max_points : int
        batch_size : int
            Number of Poisson points generated per vectorized batch.
        practical_bound : float
            Standard-deviations-equivalent treated as an
            effectively-impossible upper bound (P(Z > 6) ~ 1e-9) for
            the early-stopping rule -- a probabilistic, not exact,
            truncation (same pattern as schlather.py). Verified
            empirically against the closed-form bivariate CDF in tests.
        """
        rng = rng or np.random.default_rng()
        coords = np.asarray(coords, dtype=float)
        D = coords.shape[0]

        origin = coords.mean(axis=0)
        gamma_to_origin = np.array([self._gamma(coords[i] - origin) for i in range(D)])

        diffs = coords[:, None, :] - coords[None, :, :]
        gamma_pairwise = np.array([
            [self._gamma(diffs[i, j]) for j in range(D)] for i in range(D)
        ])

        # Cov(eps(s_i), eps(s_j)) = gamma(s_i-o) + gamma(s_j-o) - gamma(s_i-s_j)
        # (standard construction of a valid covariance from a conditionally
        # negative-definite semivariogram, anchored at an arbitrary origin o)
        cov = gamma_to_origin[:, None] + gamma_to_origin[None, :] - gamma_pairwise
        cov = cov + np.eye(D) * 1e-8
        chol = np.linalg.cholesky(cov)

        var = np.diag(cov)
        w_bound = np.exp(practical_bound * np.sqrt(np.maximum(var, 0)) - gamma_to_origin)

        out = np.zeros((n_reps, D))
        for r in range(n_reps):
            running_max = np.zeros(D)
            t_cum = 0.0
            n_used = 0
            while n_used < max_points:
                b = min(batch_size, max_points - n_used)
                cum = np.cumsum(rng.exponential(1.0, size=b)) + t_cum
                t_cum = cum[-1]
                n_used += b

                xi = 1.0 / cum  # (b,)
                eps = chol @ rng.standard_normal((D, b))  # (D, b)
                w = np.exp(eps - gamma_to_origin[:, None])  # (D, b)
                contrib = xi[None, :] * w  # (D, b)
                np.maximum(running_max, contrib.max(axis=1), out=running_max)

                if np.all(xi[-1] * w_bound <= running_max):
                    break
            out[r] = running_max
        return out
