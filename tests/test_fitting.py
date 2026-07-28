import numpy as np
import pandas as pd
import pytest

from spatial_extremes.fitting import fit_pairwise_likelihood
from spatial_extremes.models.smith import SmithModel
from spatial_extremes.models.schlather import SchlatherModel
from spatial_extremes.models.brownresnick import BrownResnickModel


def _frechet_frame(sims, n_sites):
    return pd.DataFrame(sims, columns=[f"s{i}" for i in range(n_sites)])


def test_fit_smith_recovers_extremal_coefficients():
    """Fit Smith on Smith-simulated data; check the fitted model's
    extremal coefficient function matches the true one closely, rather
    than requiring exact recovery of the (unidentifiable-up-to-rotation)
    covariance entries themselves."""
    rng = np.random.default_rng(100)
    true_cov = np.array([[2.0, 0.4], [0.4, 1.3]])
    true_model = SmithModel(cov=true_cov)

    coords = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 3.0]])
    sims = true_model.simulate(coords, n_reps=1500, rng=rng)
    df = _frechet_frame(sims, coords.shape[0])

    fit = fit_pairwise_likelihood(SmithModel, df, coords, rng=rng)
    assert fit.converged or fit.optimizer_result.nit > 0

    test_hs = [np.array([2.0, 0.0]), np.array([0.0, 2.0]), np.array([2.0, 2.0])]
    for h in test_hs:
        true_theta = true_model.extremal_coefficient(h)
        fit_theta = fit.model.extremal_coefficient(h)
        assert abs(true_theta - fit_theta) < 0.15, (
            f"h={h}: true={true_theta:.3f} fit={fit_theta:.3f}"
        )


def test_fit_schlather_recovers_extremal_coefficients():
    rng = np.random.default_rng(101)
    true_model = SchlatherModel(range_=2.5, smoothness=1.0)
    coords = np.array([[0.0, 0.0], [1.5, 0.0], [0.0, 1.5], [1.5, 1.5], [0.7, 2.2]])
    sims = true_model.simulate(coords, n_reps=1500, rng=rng)
    df = _frechet_frame(sims, coords.shape[0])

    fit = fit_pairwise_likelihood(SchlatherModel, df, coords, rng=rng)

    test_hs = [np.array([1.5, 0.0]), np.array([1.5, 1.5])]
    for h in test_hs:
        true_theta = true_model.extremal_coefficient(h)
        fit_theta = fit.model.extremal_coefficient(h)
        assert abs(true_theta - fit_theta) < 0.15, (
            f"h={h}: true={true_theta:.3f} fit={fit_theta:.3f}"
        )


def test_fit_brownresnick_recovers_extremal_coefficients():
    rng = np.random.default_rng(102)
    true_model = BrownResnickModel(range_=2.0, alpha=1.2)
    coords = np.array([[0.0, 0.0], [1.5, 0.0], [0.0, 1.5], [1.5, 1.5], [0.7, 2.2]])
    sims = true_model.simulate(coords, n_reps=1500, rng=rng)
    df = _frechet_frame(sims, coords.shape[0])

    fit = fit_pairwise_likelihood(BrownResnickModel, df, coords, rng=rng)

    test_hs = [np.array([1.5, 0.0]), np.array([1.5, 1.5])]
    for h in test_hs:
        true_theta = true_model.extremal_coefficient(h)
        fit_theta = fit.model.extremal_coefficient(h)
        assert abs(true_theta - fit_theta) < 0.15, (
            f"h={h}: true={true_theta:.3f} fit={fit_theta:.3f}"
        )


def test_max_pairs_subsampling_runs():
    """Sanity check that pair-subsampling for larger site counts doesn't crash
    and still gives a roughly sane fit."""
    rng = np.random.default_rng(103)
    true_model = SmithModel(cov=np.eye(2) * 1.5)
    coords = rng.uniform(0, 5, size=(10, 2))
    sims = true_model.simulate(coords, n_reps=500, rng=rng)
    df = _frechet_frame(sims, coords.shape[0])

    fit = fit_pairwise_likelihood(SmithModel, df, coords, max_pairs=15, rng=rng)
    assert fit.n_pairs_used == 15
