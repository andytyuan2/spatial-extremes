"""
Schlather (2002) extremal Gaussian max-stable process.

Spectral representation:

    Z(s) = max_i  xi_i * W_i(s),   W_i(s) = sqrt(2*pi) * max(0, eps_i(s))

where eps_i are iid copies of a stationary standard Gaussian process
with correlation function rho(h), and {xi_i} are points of a Poisson
process on (0, inf) with intensity xi^-2 dxi (xi_i = 1/Gamma_i,
Gamma_i the arrival times of a unit-rate Poisson process). The
sqrt(2*pi) normalizing constant makes E[W(s)] = 1, which is what
guarantees unit-Frechet margins.

Unlike Smith's model (a spatially *localized* kernel, which needs a
bounded sampling window + area-correction, see smith.py), each
Schlather "storm" is a full realization of a correlated field over
*all* sites at once -- there's no spatial windowing to worry about.

Known limitation of this model (not a bug): the extremal coefficient
saturates at 1 + sqrt(2)/2 ~= 1.707 as rho -> 0, so it can never
represent full independence (theta=2) between sites, however far
apart. This is a documented property of the Schlather model itself
(see Schlather 2002; Davison, Padoan & Ribatet 2012 Stat. Science
review), not a simulation artifact.

Bivariate CDF and its partial derivatives (needed for the pairwise
likelihood) were derived symbolically with sympy and cross-checked
against finite differences of the CDF; see fitting.py / project notes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats


def powered_exponential(h: np.ndarray, range_: float, smoothness: float = 1.0) -> np.ndarray:
    """rho(h) = exp(-(|h|/range)^smoothness), smoothness in (0, 2]."""
    d = np.linalg.norm(h, axis=-1) if np.ndim(h) > 1 else np.linalg.norm(h)
    return np.exp(-((d / range_) ** smoothness))


@dataclass
class SchlatherModel:
    """Extremal Gaussian process with a user-supplied correlation function.

    Parameters
    ----------
    range_ : float
        Correlation range (> 0).
    smoothness : float
        Powered-exponential smoothness parameter, in (0, 2]. 1.0 gives
        the exponential correlation, 2.0 gives the Gaussian (squared
        exponential) correlation.
    correlation_fn : callable, optional
        Override with a custom rho(h, range_, smoothness) -> [-1, 1].
        Defaults to the powered-exponential family above.
    """

    range_: float
    smoothness: float = 1.0
    correlation_fn: Callable = powered_exponential

    n_free_params = 2
    free_param_names = ("log_range", "logit_smoothness")

    @classmethod
    def from_free_params(cls, free_params, **kwargs) -> "SchlatherModel":
        log_range, logit_smoothness = free_params
        range_ = np.exp(log_range)
        # squash logit_smoothness (-inf, inf) -> smoothness in (0, 2]
        smoothness = 2.0 / (1.0 + np.exp(-logit_smoothness))
        return cls(range_=range_, smoothness=smoothness, **kwargs)

    def __post_init__(self):
        if self.range_ <= 0:
            raise ValueError("range_ must be positive")
        if not (0 < self.smoothness <= 2):
            raise ValueError("smoothness must be in (0, 2]")

    def _rho(self, h: np.ndarray) -> float:
        h = np.asarray(h, dtype=float)
        rho = float(self.correlation_fn(h, self.range_, self.smoothness))
        return np.clip(rho, -0.999999, 0.999999)

    def extremal_coefficient(self, h: np.ndarray) -> float:
        """theta(h) = 1 + sqrt((1 - rho(h)) / 2); saturates at ~1.707, not 2."""
        rho = self._rho(h)
        return 1 + np.sqrt((1 - rho) / 2)

    def bivariate_cdf(self, z1: float, z2: float, h: np.ndarray) -> float:
        rho = self._rho(h)
        s = 1 - 2 * (rho + 1) * z1 * z2 / (z1 + z2) ** 2
        s = max(s, 0.0)
        V = 0.5 * (1 / z1 + 1 / z2) * (1 + np.sqrt(s))
        return float(np.exp(-V))

    def bivariate_logpdf(self, z1, z2, h: np.ndarray):
        """Closed-form bivariate log-density on unit-Frechet margins.

        Derived symbolically (sympy) from V(z1,z2) and cross-checked
        against a central finite-difference of the CDF.
        """
        z1 = np.asarray(z1, dtype=float)
        z2 = np.asarray(z2, dtype=float)
        rho = self._rho(h)

        b = np.sqrt(z1 ** 2 + z2 ** 2 - 2 * rho * z1 * z2)
        b = np.where(b > 1e-10, b, 1e-10)

        V1 = (rho * z1 - z2 - b) / (2 * z1 ** 2 * b)
        V2 = (rho * z2 - z1 - b) / (2 * z2 ** 2 * b)
        V12 = (rho ** 2 - 1) / (2 * b ** 3)

        s = 1 - 2 * (rho + 1) * z1 * z2 / (z1 + z2) ** 2
        s = np.where(s > 0, s, 1e-12)
        V = 0.5 * (1 / z1 + 1 / z2) * (1 + np.sqrt(s))

        density_factor = V1 * V2 - V12
        density_factor = np.where(density_factor > 0, density_factor, 1e-300)
        return -V + np.log(density_factor)

    def simulate(
        self,
        coords: np.ndarray,
        n_reps: int = 1,
        max_points: int = 20_000,
        practical_bound: float = 10.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate on unit-Frechet margins at given sites.

        Parameters
        ----------
        coords : (D, 2) array_like
        n_reps : int
        max_points : int
            Hard cap on Poisson points per replicate.
        practical_bound : float
            Standard-normal z-score treated as an effectively-impossible
            upper bound (P(Z > 10) ~ 1e-23) for the early-stopping rule.
            This makes the truncation a probabilistic, not exact,
            approximation -- unlike Smith's model, W(s) here is
            unbounded, so no exact finite stopping rule exists. Verified
            empirically against the closed-form bivariate CDF in tests.
        """
        rng = rng or np.random.default_rng()
        coords = np.asarray(coords, dtype=float)
        D = coords.shape[0]

        # site-to-site correlation matrix, then its Cholesky factor for
        # fast iid-Gaussian-process sampling per Poisson point
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        R = np.exp(-((dists / self.range_) ** self.smoothness))
        R = R + np.eye(D) * 1e-10  # numerical PD safety margin
        chol = np.linalg.cholesky(R)

        w_bound = np.sqrt(2 * np.pi) * practical_bound

        out = np.zeros((n_reps, D))
        for r in range(n_reps):
            running_max = np.zeros(D)
            t_cum = 0.0
            n_used = 0
            while n_used < max_points:
                t_cum += rng.exponential(1.0)
                xi = 1.0 / t_cum
                n_used += 1

                if xi * w_bound <= running_max.min() and n_used > D:
                    break

                z = chol @ rng.standard_normal(D)
                w = np.sqrt(2 * np.pi) * np.maximum(0.0, z)
                np.maximum(running_max, xi * w, out=running_max)
            out[r] = running_max
        return out
