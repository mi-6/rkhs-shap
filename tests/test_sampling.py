import numpy as np
import pytest

from rkhs_shap.sampling import (
    sample_coalitions_full,
    sample_coalitions_hybrid,
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


@pytest.mark.parametrize("m,n_samples", [(3, 10), (5, 50), (10, 100)])
def test_sample_coalitions_hybrid(m, n_samples):
    Z = sample_coalitions_hybrid(m, n_samples)

    assert Z.shape[1] == m
    assert Z.shape[0] >= min(2**m, n_samples + 2)
    assert Z.dtype == bool
    assert np.all(~Z[-2])
    assert np.all(Z[-1])

    coalition_sizes = Z[:-2].sum(axis=1)
    assert np.all(coalition_sizes > 0)
    assert np.all(coalition_sizes < m)


def test_sample_coalitions_hybrid_exhaustive():
    """Test that hybrid sampling exhaustively enumerates when budget allows."""
    m = 5
    n_samples = 1000

    Z = sample_coalitions_hybrid(m, n_samples)

    size_1_coalitions = Z[Z.sum(axis=1) == 1]
    size_4_coalitions = Z[Z.sum(axis=1) == 4]

    unique_size_1 = np.unique(size_1_coalitions, axis=0)
    unique_size_4 = np.unique(size_4_coalitions, axis=0)

    assert len(unique_size_1) == m
    assert len(unique_size_4) == m


def test_sample_coalitions_hybrid_small_budget():
    """Test that hybrid sampling switches to random sampling with small budget."""
    m = 10
    n_samples = 50

    Z = sample_coalitions_hybrid(m, n_samples)

    assert Z.shape[0] == n_samples + 2
    assert np.all(~Z[-2])
    assert np.all(Z[-1])

    coalition_sizes = Z[:-2].sum(axis=1)
    assert len(np.unique(coalition_sizes)) > 1


def test_sample_coalitions_hybrid_deterministic_part():
    """Test that the exhaustively enumerated part is deterministic."""
    m = 4
    n_samples = 100

    Z1 = sample_coalitions_hybrid(m, n_samples)
    Z2 = sample_coalitions_hybrid(m, n_samples)

    size_1_mask_1 = Z1[:-2].sum(axis=1) == 1
    size_1_mask_2 = Z2[:-2].sum(axis=1) == 1

    size_1_coalitions_1 = Z1[:-2][size_1_mask_1]
    size_1_coalitions_2 = Z2[:-2][size_1_mask_2]

    sorted_1 = size_1_coalitions_1[np.lexsort(size_1_coalitions_1.T[::-1])]
    sorted_2 = size_1_coalitions_2[np.lexsort(size_1_coalitions_2.T[::-1])]

    assert np.array_equal(sorted_1, sorted_2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
