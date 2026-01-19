import numpy as np
import pytest

from rkhs_shap.sampling import (
    sample_coalitions_full,
    sample_coalitions_weighted,
)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_sample_coalitions_full(m):
    Z = sample_coalitions_full(m)

    assert Z.shape == (2**m, m)
    assert Z.dtype == bool
    assert np.all(~Z[0])
    assert np.all(Z[-1])

    unique_rows = np.unique(Z, axis=0)
    assert len(unique_rows) == 2**m


@pytest.mark.parametrize("m,n_samples", [(5, 50), (10, 100), (8, 200)])
def test_sample_coalitions_weighted(m, n_samples):
    Z = sample_coalitions_weighted(m, n_samples)

    n_unique = Z.shape[0] - 2
    assert n_unique <= n_samples
    assert Z.shape == (n_unique + 2, m)
    assert Z.dtype == bool
    assert np.all(~Z[-2])
    assert np.all(Z[-1])

    coalition_sizes = Z[:-2].sum(axis=1)
    assert len(np.unique(coalition_sizes)) > 1

    # All sampled coalitions should be unique
    sampled_coalitions = Z[:-2]
    unique_rows = np.unique(sampled_coalitions, axis=0)
    assert len(unique_rows) == n_unique


@pytest.mark.parametrize("m,n_samples", [(5, 50), (10, 100)])
def test_sample_coalitions_weighted_reproducibility_default(m, n_samples):
    """Test that passing explicit RNG produces identical results."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    Z1 = sample_coalitions_weighted(m, n_samples, rng=rng1)
    Z2 = sample_coalitions_weighted(m, n_samples, rng=rng2)
    np.testing.assert_array_equal(Z1, Z2)


@pytest.mark.parametrize("m,n_samples", [(5, 50), (10, 100)])
def test_sample_coalitions_weighted_reproducibility_with_rng(m, n_samples):
    """Test that passing same seed produces identical results."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    Z1 = sample_coalitions_weighted(m, n_samples, rng=rng1)
    Z2 = sample_coalitions_weighted(m, n_samples, rng=rng2)
    np.testing.assert_array_equal(Z1, Z2)


@pytest.mark.parametrize("m,n_samples", [(5, 50), (10, 100)])
def test_sample_coalitions_weighted_different_seeds(m, n_samples):
    """Test that different seeds produce different results."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(456)
    Z1 = sample_coalitions_weighted(m, n_samples, rng=rng1)
    Z2 = sample_coalitions_weighted(m, n_samples, rng=rng2)
    assert not np.array_equal(Z1, Z2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
