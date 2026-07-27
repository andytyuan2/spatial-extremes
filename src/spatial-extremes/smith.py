"""
Smith (1990) max-stable process.

Spectral (de Haan) representation:

    Z(s) = max_i  xi_i * phi(s - U_i; Sigma)

where {(xi_i, U_i)} are points of a Poisson process on (0, inf) x R^2
with intensity xi^-2 dxi x du, and phi(.; Sigma) is the density of a
bivariate N(0, Sigma) distribution. This gives Z with unit-Frechet
margins and spatial dependence controlled entirely by Sigma.

Simulation strategy (standard truncated point-process algorithm):
  1. Storm centers U_i are restricted to sample sites' bounding box,
     expanded by a buffer (a few kernel bandwidths) so that storms
     centered just outside the domain still contribute.
  2. xi_i = A / T_i where T_i is the i-th arrival time of a unit-rate
     Poisson process (cumulative sum of iid Exp(1)) and A is the area
     of the sampling box. Restricting storm centers to a finite box
     thins the (xi, U) process spatially; integrating the intensity
     xi^-2 dxi du over the box shows the resulting xi-marginal process
     has intensity A * xi^-2 dxi, hence the area factor.
  3. Because phi(.; Sigma) is bounded above by phi(0; Sigma), once
     xi_i * phi(0; Sigma) falls below the current running max at
     every site, no later point (i > i_stop) can ever change the
     max. This gives an exact, finite stopping rule for a fixed
     realization (not an approximation of the algorithm itself --
     the only approximation is the finite buffer around the domain).

This is a from-scratch reimplementation of the published algorithm
(Schlather 2002 discusses the general recipe for Smith/Schlather
processes); no code was ported from the R `SpatialExtremes` source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    gpd = None
    Point = None


@dataclass
class SmithModel:
    """Isotropic or anisotropic Smith max-stable process.

    Parameters
    ----------
    cov : (2, 2) array_like
        Covariance matrix Sigma of the Gaussian kernel. Larger
        eigenvalues -> smoother fields -> stronger spatial dependence.
    buffer_factor : float
        Storm centers are drawn from the sites' bounding box expanded
        by `buffer_factor * sqrt(max eigenvalue of Sigma)` on each
        side. Increase this if sites near the domain edge look
        under-dispersed.
    """

    cov: np.ndarray
    buffer_factor: float = 4.0

    def __post_init__(self):
        self.cov = np.asarray(self.cov, dtype=float)
        if self.cov.shape != (2, 2):
            raise ValueError("cov must be a 2x2 matrix")
        self._inv_cov = np.linalg.inv(self.cov)
        self._det_cov = np.linalg.det(self.cov)
        if self._det_cov <= 0:
            raise ValueError("cov must be positive definite")
        self._norm_const = 1.0 / (2 * np.pi * np.sqrt(self._det_cov))
        self._kernel_max = self._norm_const  # phi(0; Sigma), the sup of phi

    def _kernel(self, delta: np.ndarray) -> np.ndarray:
        """Gaussian kernel density phi(delta; Sigma) for an (n, 2) array of offsets."""
        quad = np.einsum("ij,jk,ik->i", delta, self._inv_cov, delta)
        return self._norm_const * np.exp(-0.5 * quad)

    def extremal_coefficient(self, h: np.ndarray) -> float:
        """Theoretical pairwise extremal coefficient theta(h) in [1, 2].

        theta=1: perfect dependence, theta=2: independence. Closed form
        for the Smith model: theta(h) = 2 * Phi(a/2), a = sqrt(h' Sigma^-1 h).
        """
        h = np.asarray(h, dtype=float)
        a = np.sqrt(h @ self._inv_cov @ h)
        return 2 * stats.norm.cdf(a / 2)

    def bivariate_cdf(self, z1: float, z2: float, h: np.ndarray) -> float:
        """Closed-form bivariate CDF on unit-Frechet margins, sites h apart."""
        h = np.asarray(h, dtype=float)
        a = np.sqrt(h @ self._inv_cov @ h)
        if a == 0:
            return np.exp(-1.0 / min(z1, z2))
        t1 = stats.norm.cdf(a / 2 + np.log(z2 / z1) / a)
        t2 = stats.norm.cdf(a / 2 + np.log(z1 / z2) / a)
        return np.exp(-t1 / z1 - t2 / z2)

    def simulate(
        self,
        coords: np.ndarray,
        n_reps: int = 1,
        max_points: int = 200_000,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Simulate the process on unit-Frechet margins at given sites.

        Parameters
        ----------
        coords : (D, 2) array_like
            Site coordinates.
        n_reps : int
            Number of independent replicates (e.g. independent storms/years).
        max_points : int
            Safety cap on Poisson points per replicate.

        Returns
        -------
        (n_reps, D) ndarray of simulated values on unit-Frechet margins.
        """
        rng = rng or np.random.default_rng()
        coords = np.asarray(coords, dtype=float)
        D = coords.shape[0]

        buffer = self.buffer_factor * np.sqrt(np.max(np.linalg.eigvalsh(self.cov)))
        lo = coords.min(axis=0) - buffer
        hi = coords.max(axis=0) + buffer
        area = np.prod(hi - lo)

        out = np.zeros((n_reps, D))

        for r in range(n_reps):
            running_max = np.zeros(D)
            t_cum = 0.0
            n_used = 0
            while n_used < max_points:
                t_cum += rng.exponential(1.0)
                # Restricting storm centers U to a box of area `area`
                # thins the (xi, U) Poisson process spatially. The
                # marginal process of xi-values for points landing in
                # the box then has intensity `area * xi^-2 dxi`
                # (integrating the constant-in-u intensity over the
                # box), so the i-th largest xi is `area / Gamma_i`
                # rather than `1 / Gamma_i` -- this area factor is what
                # keeps the margins exactly unit Frechet as the buffer
                # grows (it's not a free fudge factor; it falls out of
                # the Poisson process intensity calculation).
                xi = area / t_cum
                n_used += 1

                # Stopping rule: even a storm centered exactly on every
                # site simultaneously (impossible, but an upper bound)
                # cannot exceed xi * kernel_max at any site.
                if xi * self._kernel_max <= running_max.min() and n_used > D:
                    break

                center = lo + rng.random(2) * (hi - lo)
                contrib = xi * self._kernel(coords - center)
                np.maximum(running_max, contrib, out=running_max)
            out[r] = running_max

        # Correct for the finite simulation window: points outside the
        # buffered box contribute negligibly for well-chosen buffer, but
        # we rescale by (area / area) = 1 here -- kept explicit as a hook
        # in case you want importance-sampling corrections later.
        return out

    def simulate_geo(
        self,
        gdf,
        n_reps: int = 1,
        value_prefix: str = "sim",
        rng: np.random.Generator | None = None,
    ):
        """Simulate and return results attached to a GeoDataFrame of sites.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Point geometries giving site locations (must be in a
            *projected* CRS, e.g. metres, not lat/lon degrees, since
            the kernel distance is Euclidean).
        n_reps : int
            Number of replicates -> columns f"{value_prefix}_{r}".
        """
        if gpd is None:
            raise ImportError("geopandas is required for simulate_geo(); pip install geopandas")
        if gdf.crs is not None and gdf.crs.is_geographic:
            raise ValueError(
                "gdf is in a geographic CRS (degrees). Reproject to a "
                "projected CRS (e.g. gdf.to_crs(epsg=...)) so distances "
                "are in consistent linear units before simulating."
            )
        coords = np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])
        sims = self.simulate(coords, n_reps=n_reps, rng=rng)
        import pandas as pd
        sim_cols = pd.DataFrame(
            sims.T, columns=[f"{value_prefix}_{r}" for r in range(n_reps)], index=gdf.index
        )
        return gpd.GeoDataFrame(pd.concat([gdf, sim_cols], axis=1), crs=gdf.crs)
