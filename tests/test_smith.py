import numpy as np
import pytest

from maxstable.smith import SmithModel


def test_extremal_coefficient_bounds():
    model = SmithModel(cov=np.eye(2) * 2.0)
    theta_zero = model.extremal_coefficient(np.array([0.0, 0.0]))
    theta_far = model.extremal_coefficient(np.array([1000.0, 0.0]))
    assert theta_zero == pytest.approx(1.0, abs=1e-6)   # h=0 -> full dependence
    assert theta_far == pytest.approx(2.0, abs=1e-3)     # h large -> independence


def test_simulate_marginals_are_unit_frechet():
    """Each site's marginal simulated distribution should match unit Frechet:
    P(Z <= z) = exp(-1/z), regardless of spatial dependence structure."""
    rng = np.random.default_rng(42)
    model = SmithModel(cov=np.eye(2) * 1.5)
    coords = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    sims = model.simulate(coords, n_reps=4000, rng=rng)

    for j in range(coords.shape[0]):
        z = sims[:, j]
        # empirical CDF at a few test points vs theoretical unit-Frechet CDF
        for test_z in [1.0, 2.0, 5.0]:
            empirical = np.mean(z <= test_z)
            theoretical = np.exp(-1.0 / test_z)
            assert abs(empirical - theoretical) < 0.03, (
                f"site {j}, z={test_z}: empirical={empirical:.3f} "
                f"theoretical={theoretical:.3f}"
            )


def test_simulate_matches_closed_form_bivariate_cdf():
    """Validate the simulator against Smith's known closed-form bivariate CDF."""
    rng = np.random.default_rng(7)
    cov = np.array([[2.0, 0.3], [0.3, 1.5]])
    model = SmithModel(cov=cov)
    h = np.array([2.0, 1.0])
    coords = np.array([[0.0, 0.0], h])

    n_reps = 20_000
    sims = model.simulate(coords, n_reps=n_reps, rng=rng)

    z1, z2 = 1.5, 2.5
    empirical = np.mean((sims[:, 0] <= z1) & (sims[:, 1] <= z2))
    theoretical = model.bivariate_cdf(z1, z2, h)

    # Monte Carlo error at n=20000 for a probability ~0.3-0.6 is a few %
    assert abs(empirical - theoretical) < 0.03, (
        f"empirical={empirical:.4f} theoretical={theoretical:.4f}"
    )


def test_cov_must_be_positive_definite():
    with pytest.raises(ValueError):
        SmithModel(cov=np.array([[1.0, 2.0], [2.0, 1.0]]))  # not PSD
