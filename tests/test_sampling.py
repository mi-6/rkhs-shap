import numpy as np
import pytest

from rkhs_shap.sampling import (
    sample_coalitions_full,
    sample_coalitions_weighted,
)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_sample_coalitions_full(m):
    Z, is_sampled = sample_coalitions_full(m)

    assert Z.shape == (2**m, m)
    assert Z.dtype == bool
    assert is_sampled.shape == (2**m,)
    assert is_sampled.dtype == bool
    assert np.all(~is_sampled)
    assert np.all(~Z[0])
    assert np.all(Z[-1])

    unique_rows = np.unique(Z, axis=0)
    assert len(unique_rows) == 2**m


@pytest.mark.parametrize("m,n_samples", [(5, 50), (10, 100), (8, 200)])
def test_sample_coalitions_weighted(m, n_samples):
    Z, is_sampled = sample_coalitions_weighted(m, n_samples)

    n_unique = Z.shape[0] - 2
    assert n_unique <= n_samples
    assert Z.shape == (n_unique + 2, m)
    assert Z.dtype == bool
    assert is_sampled.shape == (n_unique + 2,)
    assert is_sampled.dtype == bool
    assert np.all(is_sampled[:-2])
    assert not is_sampled[-2]
    assert not is_sampled[-1]
    assert np.all(~Z[-2])
    assert np.all(Z[-1])

    coalition_sizes = Z[:-2].sum(axis=1)
    assert len(np.unique(coalition_sizes)) > 1

    # All sampled coalitions should be unique
    sampled_coalitions = Z[:-2]
    unique_rows = np.unique(sampled_coalitions, axis=0)
    assert len(unique_rows) == n_unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
