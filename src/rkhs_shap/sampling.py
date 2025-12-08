from itertools import combinations
from math import comb

import numpy as np


def generate_full_Z(m: int) -> np.ndarray:
    """Generate the 2^m possible ordering of the binary matrix

    Args:
        m: Number of features

    Returns:
        ls of binary orderings
    """

    Z = np.zeros((2**m, m))
    Z[-1, :] = True  # Make sure the last row is all ones - part of the hard constraint

    elements = [i for i in range(m)]
    count = 1
    for s in range(1, m):
        perm_s = list(combinations(elements, s))
        for i in list(perm_s):
            Z[count, i] = True
            count += 1

    return Z.astype(np.bool_)


def subset_full_Z(Z: np.ndarray, samples: int = 1000) -> np.ndarray:
    """Sampling from the full permutation based on the shapley kernel weight

    *note: if M is large this is not very feasible, especially if one wants to deal with Data Shapley

    Args:
        Z: The array of permutations
        samples: Number of samples to draw. Defaults to 1000.

    Returns:
        Z_subset: The subsetted Z matrix according to the kernel shap weight
    """

    # Set probability weight to sample according to the kernelshap weight - however the all ones and all zeros are ignored
    prob_vec = [_get_weights(row.sum(), len(row)) for row in Z[1:-1,]]

    index = np.random.choice(
        2 ** Z.shape[1] - 2,
        replace=True,
        p=prob_vec / np.array(prob_vec).sum(),
        size=samples,
    )
    index = np.append(index, [0, -1])

    return Z[index]


def large_scale_sample_alternative(m: int, n_samples: int) -> np.ndarray:
    """Sample the permutation according to the kernel weight

    We first sample the size of the permutation = number of ones in a permutation, and then we randomly select
    1 particular permutation there since their distribution is uniform conditioned on the size of the permutation.

    Args:
        m: number of features
        n_samples: number of samples you want

    Returns:
        Z: The matrix of boolean values
    """

    def prob_per_size(m: int, s: int) -> float:
        return (m - 1) / (s * (m - s))

    samples_container = []

    prob_vec_per_size = np.array([prob_per_size(m, s) for s in range(1, m)])
    prob_vec_per_size /= prob_vec_per_size.sum()
    size_ls = np.random.choice(
        range(1, m), p=prob_vec_per_size, size=n_samples + 2, replace=True
    )

    for size in size_ls:
        all_ones = np.zeros(m)
        all_ones[:size] = 1
        shuffled_ind = np.random.choice(range(m), size=m, replace=False)
        new_sample = all_ones[shuffled_ind]
        samples_container.append(new_sample)

    samples_container = np.array(samples_container).astype(np.bool_)

    samples_container[-2, :] = False
    samples_container[-1, :] = True

    return samples_container


def large_scale_sample_uniform(m: int, n_samples: int) -> np.ndarray:
    """Sample coalitions uniformly (i.i.d.) from all non-trivial coalitions.

    Each coalition has equal probability, regardless of size. This is the simplest
    form of Monte Carlo sampling. Use with full Shapley kernel weights in regression
    for unbiased estimation.

    This approach is more transparent than size-stratified sampling (used in SHAP's
    KernelExplainer) but may have higher variance since it doesn't concentrate
    samples on subset sizes with higher Shapley kernel weights.

    Args:
        m: number of features
        n_samples: number of samples you want

    Returns:
        Z: The matrix of boolean values, shape (n_samples + 2, m)
           Last two rows are empty and full coalitions
    """
    samples_container = []

    for _ in range(n_samples):
        coalition = np.random.rand(m) < 0.5

        # Reject empty and full coalitions (re-sample if needed)
        while coalition.sum() == 0 or coalition.sum() == m:
            coalition = np.random.rand(m) < 0.5

        samples_container.append(coalition)

    samples_container = np.array(samples_container, dtype=np.bool_)

    # Always include empty and full coalitions
    empty = np.zeros(m, dtype=bool)
    full = np.ones(m, dtype=bool)
    samples_container = np.vstack([samples_container, empty, full])

    return samples_container


def _get_weights(s: int, m: int) -> float:
    """The unnormalised probability weight to sample a particular permutation

    Args:
        s: size
        m: number of active features

    Returns:
        Unnormalised probability weight
    """
    return (m - 1) / (comb(m, s) * s * (m - s))
