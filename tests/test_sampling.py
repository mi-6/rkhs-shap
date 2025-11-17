import numpy as np
import pytest

from rkhs_shap.sampling import (
    generate_full_Z,
    subset_full_Z,
    large_scale_sample_alternative,
    _get_weights,
    _propose_func,
    generate_samples_Z,
)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_generate_full_Z(m):
    Z = generate_full_Z(m)

    assert Z.shape == (2**m, m)
    assert Z.dtype == bool
    assert np.all(Z[0] == False)
    assert np.all(Z[-1] == True)

    unique_rows = np.unique(Z, axis=0)
    assert len(unique_rows) == 2**m


@pytest.mark.parametrize("m,samples", [(3, 10), (4, 50), (5, 100)])
def test_subset_full_Z(m, samples):
    Z = generate_full_Z(m)
    Z_subset = subset_full_Z(Z, samples=samples)

    assert Z_subset.shape == (samples + 2, m)
    assert Z_subset.dtype == bool
    assert np.all(Z_subset[-2] == False) or np.all(Z_subset[-1] == False)
    assert np.all(Z_subset[-2] == True) or np.all(Z_subset[-1] == True)


@pytest.mark.parametrize("m,n_samples", [(5, 50), (10, 100), (8, 200)])
def test_large_scale_sample_alternative(m, n_samples):
    Z = large_scale_sample_alternative(m, n_samples)

    assert Z.shape == (n_samples + 2, m)
    assert Z.dtype == bool
    assert np.all(Z[-2] == False)
    assert np.all(Z[-1] == True)

    coalition_sizes = Z[:-2].sum(axis=1)
    assert len(np.unique(coalition_sizes)) > 1


@pytest.mark.parametrize("s,m", [(1, 5), (2, 5), (4, 5), (1, 10), (5, 10), (9, 10)])
def test_get_weights(s, m):
    weight = _get_weights(s, m)

    assert weight > 0
    assert np.isfinite(weight)


def test_get_weights_boundary():
    weight_low = _get_weights(1, 10)
    weight_high = _get_weights(9, 10)

    assert weight_low > 0
    assert weight_high > 0


def test_propose_func():
    np.random.seed(42)
    z = np.array([1, -1, 1, -1, 1])
    z_original = z.copy()

    z_proposed = _propose_func(z)

    assert z_proposed is z

    diff = (z_proposed != z_original).sum()
    assert diff == 1

    assert set(z_proposed) == {1, -1}


@pytest.mark.parametrize("m,mcmc_run,warm_up_cut", [(3, 50, 10), (5, 100, 20)])
def test_generate_samples_Z(m, mcmc_run, warm_up_cut):
    np.random.seed(42)
    Z = generate_samples_Z(m, mcmc_run, warm_up_cut)

    assert Z.shape[1] == m
    assert Z.shape[0] == mcmc_run - warm_up_cut + 4
    assert Z.dtype == bool

    assert np.all(Z[-2] == True)
    assert np.all(Z[-1] == False)


def test_generate_samples_Z_warmup():
    np.random.seed(42)
    m = 4
    mcmc_run = 100
    warm_up_cut = 30

    Z = generate_samples_Z(m, mcmc_run, warm_up_cut)

    expected_samples = mcmc_run - warm_up_cut + 4
    assert Z.shape[0] == expected_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
