import numpy as np
import pytest

from rkhs_shap.sampling import (
    _get_weights,
    generate_full_Z,
    generate_samples_Z,
    large_scale_sample_alternative,
    large_scale_sample_uniform,
    subset_full_Z,
)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_generate_full_Z(m):
    Z = generate_full_Z(m)

    assert Z.shape == (2**m, m)
    assert Z.dtype == bool
    assert np.all(~Z[0])
    assert np.all(Z[-1])

    unique_rows = np.unique(Z, axis=0)
    assert len(unique_rows) == 2**m


@pytest.mark.parametrize("m,samples", [(3, 10), (4, 50), (5, 100)])
def test_subset_full_Z(m, samples):
    Z = generate_full_Z(m)
    Z_subset = subset_full_Z(Z, samples=samples)

    assert Z_subset.shape == (samples + 2, m)
    assert Z_subset.dtype == bool
    assert np.all(~Z_subset[-2]) or np.all(~Z_subset[-1])
    assert np.all(Z_subset[-2]) or np.all(Z_subset[-1])


@pytest.mark.parametrize("m,n_samples", [(5, 50), (10, 100), (8, 200)])
def test_large_scale_sample_alternative(m, n_samples):
    Z = large_scale_sample_alternative(m, n_samples)

    assert Z.shape == (n_samples + 2, m)
    assert Z.dtype == bool
    assert np.all(~Z[-2])
    assert np.all(Z[-1])

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


@pytest.mark.parametrize("m,mcmc_run,warm_up_cut", [(3, 50, 10), (5, 100, 20)])
def test_generate_samples_Z(m, mcmc_run, warm_up_cut):
    np.random.seed(42)
    Z = generate_samples_Z(m, mcmc_run, warm_up_cut)

    assert Z.shape[1] == m
    assert Z.shape[0] == mcmc_run - warm_up_cut + 4
    assert Z.dtype == bool

    assert np.all(Z[-2])
    assert np.all(~Z[-1])


def test_generate_samples_Z_warmup():
    np.random.seed(42)
    m = 4
    mcmc_run = 100
    warm_up_cut = 30

    Z = generate_samples_Z(m, mcmc_run, warm_up_cut)

    expected_samples = mcmc_run - warm_up_cut + 4
    assert Z.shape[0] == expected_samples


@pytest.mark.parametrize("m,n_samples", [(5, 50), (10, 100), (8, 200)])
def test_large_scale_sample_uniform(m, n_samples):
    Z = large_scale_sample_uniform(m, n_samples)

    assert Z.shape == (n_samples + 2, m)
    assert Z.dtype == bool
    assert np.all(~Z[-2])
    assert np.all(Z[-1])

    coalition_sizes = Z[:-2].sum(axis=1)
    assert np.all(coalition_sizes > 0)
    assert np.all(coalition_sizes < m)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
