import numpy as np
import pytest

from rkhs_shap.sampling import (
    sample_coalitions_full,
    sample_coalitions_uniform,
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

    assert Z.shape == (n_samples + 2, m)
    assert Z.dtype == bool
    assert np.all(~Z[-2])
    assert np.all(Z[-1])

    coalition_sizes = Z[:-2].sum(axis=1)
    assert len(np.unique(coalition_sizes)) > 1


@pytest.mark.parametrize("m,n_samples", [(5, 50), (10, 100), (8, 200)])
def test_sample_coalitions_uniform(m, n_samples):
    Z = sample_coalitions_uniform(m, n_samples)

    assert Z.shape == (n_samples + 2, m)
    assert Z.dtype == bool
    assert np.all(~Z[-2])
    assert np.all(Z[-1])

    coalition_sizes = Z[:-2].sum(axis=1)
    assert np.all(coalition_sizes > 0)
    assert np.all(coalition_sizes < m)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
