import numpy as np
import pytest

from spatial_extremes.models.brownresnick import BrownResnickModel


def test_extremal_coefficient_bounds():
    model = BrownResnickModel(range_=3.0, alpha=1.0)
    theta_zero = model.extremal_coefficient(np.array([0.0, 0.0]))
    theta_far = model.extremal_coefficient(np.array([10000.0, 0.0]))
    assert theta_zero == pytest.approx(1.0, abs=1e-6)
    assert theta_far == pytest.approx(2.0, abs=1e-3)  # unlike Schlather, BR does reach independence


def test_simulate_marginals_are_unit_frechet():
    rng = np.random.default_rng(5)
    model = BrownResnickModel(range_=2.0, alpha=1.0)
    coords = np.array([[0.0, 0.0], [1.5, 0.0], [0.0, 1.5]])
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
    rng = np.random.default_rng(13)
    model = BrownResnickModel(range_=2.0, alpha=1.2)
    h = np.array([1.0, 0.8])
    coords = np.array([[0.0, 0.0], h])

    n_reps = 20_000
    sims = model.simulate(coords, n_reps=n_reps, rng=rng)

    z1, z2 = 1.4, 2.2
    empirical = np.mean((sims[:, 0] <= z1) & (sims[:, 1] <= z2))
    theoretical = model.bivariate_cdf(z1, z2, h)

    assert abs(empirical - theoretical) < 0.03, (
        f"empirical={empirical:.4f} theoretical={theoretical:.4f}"
    )


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        BrownResnickModel(range_=1.0, alpha=2.5)
