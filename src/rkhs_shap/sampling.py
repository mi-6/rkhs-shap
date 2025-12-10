from itertools import combinations
from math import comb

import numpy as np


def sample_coalitions_full(m: int) -> np.ndarray:
    """Generate all 2^m coalitions exhaustively.

    Args:
        m: Number of features

    Returns:
        Boolean array of shape (2^m, m) with all possible coalitions
    """
    Z = np.zeros((2**m, m), dtype=bool)

    count = 1
    for s in range(1, m):
        for combo in combinations(range(m), s):
            Z[count, combo] = True
            count += 1

    Z[-1, :] = True
    return Z


def sample_coalitions_weighted(m: int, n_samples: int) -> np.ndarray:
    """Sample coalitions according to Shapley kernel weights.

    First samples coalition sizes proportional to their Shapley kernel weights,
    then uniformly samples a specific coalition of that size.

    Args:
        m: Number of features
        n_samples: Number of samples

    Returns:
        Boolean array of shape (n_samples + 2, m) with sampled coalitions.
        Last two rows are empty and full coalitions.
    """
    prob_vec = np.array([_shapley_kernel_weight(m, s) for s in range(1, m)])
    prob_vec /= prob_vec.sum()

    sizes = np.random.choice(range(1, m), p=prob_vec, size=n_samples, replace=True)

    Z = np.zeros((n_samples + 2, m), dtype=bool)
    for i, size in enumerate(sizes):
        Z[i, np.random.choice(m, size=size, replace=False)] = True

    Z[-2, :] = False
    Z[-1, :] = True

    return Z


def _shapley_kernel_weight(m: int, s: int) -> float:
    """Compute Shapley kernel weight for coalition size s.

    This is the weight used in the weighted least squares formulation
    of Shapley value estimation.
    """
    return (m - 1) / (s * (m - s))


def _enumerate_coalitions_of_size(m: int, size: int) -> np.ndarray:
    """Generate all coalitions of a specific size."""

    n_combs = comb(m, size)
    coalitions = np.zeros((n_combs, m), dtype=bool)

    for i, combo in enumerate(combinations(range(m), size)):
        coalitions[i, combo] = True

    return coalitions


def sample_coalitions_hybrid(m: int, n_samples: int) -> np.ndarray:
    """Hybrid exhaustive/random sampling following SHAP's KernelExplainer strategy.

    Exhaustively enumerates coalition sizes when the sample budget allows, then
    switches to random sampling for remaining sizes. Processes sizes from extremes
    (1, m-1) inward (2, m-2, etc.) since these have highest Shapley kernel weights.

    Args:
        m: Number of features
        n_samples: Requested number of samples

    Returns:
        Boolean array of shape (n_total, m) where n_total >= min(2^m, n_samples + 2).
        When exhaustive enumeration is possible, returns all 2^m coalitions.
        Last two rows are always empty and full coalitions.
    """
    from math import comb

    samples_container = []
    num_samples_left = n_samples

    weight_vec = np.array([_shapley_kernel_weight(m, s) for s in range(1, m)])
    remaining_weight_vec = weight_vec.copy()

    num_subset_sizes = int(np.ceil((m - 1) / 2.0))

    for i in range(num_subset_sizes):
        size_small = i + 1
        size_large = m - 1 - i

        num_combs_small = comb(m, size_small)
        num_combs_large = comb(m, size_large)

        remaining_weight_sum = remaining_weight_vec.sum()
        if remaining_weight_sum == 0:
            break

        weight_small = remaining_weight_vec[size_small - 1]
        weight_large = remaining_weight_vec[size_large - 1]

        samples_allocated_small = num_samples_left * weight_small / remaining_weight_sum
        samples_allocated_large = num_samples_left * weight_large / remaining_weight_sum

        if samples_allocated_small >= num_combs_small - 1e-8:
            coalitions = _enumerate_coalitions_of_size(m, size_small)
            samples_container.append(coalitions)
            num_samples_left -= num_combs_small
            remaining_weight_vec[size_small - 1] = 0

        if (
            size_small != size_large
            and samples_allocated_large >= num_combs_large - 1e-8
        ):
            coalitions = _enumerate_coalitions_of_size(m, size_large)
            samples_container.append(coalitions)
            num_samples_left -= num_combs_large
            remaining_weight_vec[size_large - 1] = 0

    if num_samples_left > 0 and remaining_weight_vec.sum() > 0:
        remaining_weight_vec /= remaining_weight_vec.sum()

        sizes = np.random.choice(
            range(1, m), p=remaining_weight_vec, size=num_samples_left
        )
        random_samples = np.zeros((num_samples_left, m), dtype=bool)
        for i, size in enumerate(sizes):
            random_samples[i, np.random.choice(m, size=size, replace=False)] = True

        samples_container.append(random_samples)

    if samples_container:
        Z = np.vstack(samples_container)
    else:
        Z = np.empty((0, m), dtype=bool)

    empty = np.zeros((1, m), dtype=bool)
    full = np.ones((1, m), dtype=bool)
    Z = np.vstack([Z, empty, full])

    return Z
