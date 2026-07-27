import numpy as np
import pytest

from spatial_extremes.models.schlather import SchlatherModel


def test_extremal_coefficient_bounds_and_ceiling():
    model = SchlatherModel(range_=3.0, smoothness=1.0)
    theta_zero = model.extremal_coefficient(np.array([0.0, 0.0]))
    theta_far = model.extremal_coefficient(np.array([1000.0, 0.0]))
    assert theta_zero == pytest.approx(1.0, abs=1e-3)  # clipped rho -> tiny offset from exactly 1
    # Schlather's model cannot reach independence (theta=2); it saturates
    # at 1 + sqrt(2)/2 ~= 1.7071 as rho -> 0. This is a real model
    # property, not a bug -- assert it explicitly so nobody "fixes" it later.
    assert theta_far == pytest.approx(1 + np.sqrt(2) / 2, abs=1e-3)
    assert theta_far < 2.0


def test_simulate_marginals_are_unit_frechet():
    rng = np.random.default_rng(1)
    model = SchlatherModel(range_=2.0, smoothness=1.0)
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    sims = model.simulate(coords, n_reps=4000, rng=rng)

    for j in range(coords.shape[0]):
        z = sims[:, j]
        for test_z in [1.0, 2.0, 5.0]:
            empirical = np.mean(z <= test_z)
            theoretical = np.exp(-1.0 / test_z)
            assert abs(empirical - theoretical) < 0.03, (
                f"site {j}, z={test_z}: empirical={empirical:.3f} theoretical={theoretical:.3f}"
            )


def test_simulate_matches_closed_form_bivariate_cdf():
    rng = np.random.default_rng(11)
    model = SchlatherModel(range_=2.5, smoothness=1.0)
    h = np.array([1.5, 0.5])
    coords = np.array([[0.0, 0.0], h])

    n_reps = 20_000
    sims = model.simulate(coords, n_reps=n_reps, rng=rng)

    z1, z2 = 1.5, 2.0
    empirical = np.mean((sims[:, 0] <= z1) & (sims[:, 1] <= z2))
    theoretical = model.bivariate_cdf(z1, z2, h)

    assert abs(empirical - theoretical) < 0.03, (
        f"empirical={empirical:.4f} theoretical={theoretical:.4f}"
    )


def test_bivariate_logpdf_matches_finite_difference_of_cdf():
    """Cross-check the closed-form density against a numerical derivative
    of bivariate_cdf, independent of the sympy derivation used to write it."""
    model = SchlatherModel(range_=2.0, smoothness=1.0)
    h = np.array([1.0, 0.3])
    z1, z2 = 1.2, 2.3
    eps = 1e-4

    def cdf(a, b):
        return model.bivariate_cdf(a, b, h)

    fd = (cdf(z1 + eps, z2 + eps) - cdf(z1 + eps, z2 - eps)
          - cdf(z1 - eps, z2 + eps) + cdf(z1 - eps, z2 - eps)) / (4 * eps ** 2)
    analytic = np.exp(model.bivariate_logpdf(np.array([z1]), np.array([z2]), h))[0]

    assert abs(fd - analytic) / analytic < 1e-3
