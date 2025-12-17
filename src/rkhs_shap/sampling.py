from itertools import combinations

import numpy as np


def sample_coalitions_full(m: int) -> np.ndarray:
    """Generate all 2^m coalitions exhaustively.

    Args:
        m: Number of features

    Returns:
        Z: Boolean array of shape (2^m, m) with all possible coalitions
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
    """Sample unique coalitions according to Shapley kernel weights.

    First samples coalition sizes proportional to their Shapley kernel weights,
    then uniformly samples a specific coalition of that size. Duplicates are
    removed to ensure each coalition appears exactly once.

    Args:
        m: Number of features
        n_samples: Target number of unique coalition samples

    Returns:
        Z: Boolean array of shape (n_unique + 2, m) with unique sampled coalitions.
        Last two rows are empty and full coalitions.
    """
    prob_vec = np.array([_shapley_kernel_weight(m, s) for s in range(1, m)])
    prob_vec /= prob_vec.sum()

    seen: set[tuple[bool, ...]] = set()
    coalitions: list[np.ndarray] = []

    # Oversample to account for duplicates, then deduplicate
    batch_size = n_samples
    max_attempts = 10

    for _ in range(max_attempts):
        sizes = np.random.choice(range(1, m), p=prob_vec, size=batch_size, replace=True)

        for size in sizes:
            coalition = np.zeros(m, dtype=bool)
            coalition[np.random.choice(m, size=size, replace=False)] = True
            key = tuple(coalition)

            if key in seen:
                continue
            seen.add(key)
            coalitions.append(coalition)
            if len(coalitions) >= n_samples:
                break

        if len(coalitions) >= n_samples:
            break

    n_unique = len(coalitions)
    Z = np.zeros((n_unique + 2, m), dtype=bool)
    for i, coalition in enumerate(coalitions):
        Z[i] = coalition

    Z[-2, :] = False
    Z[-1, :] = True

    return Z


def _shapley_kernel_weight(m: int, s: int) -> float:
    """Compute Shapley kernel weight for coalition size s.

    This is the weight used in the weighted least squares formulation
    of Shapley value estimation.
    """
    return (m - 1) / (s * (m - s))


def compute_kernelshap_weights(Z: np.ndarray) -> np.ndarray:
    """Compute weights matching KernelSHAP's per-size normalization.

    For each coalition size s, the total weight equals the Shapley kernel
    weight for that size: (m-1) / (s * (m-s)), normalized to sum to 1.
    Each sampled coalition of size s shares this total weight equally.

    Args:
        Z: Boolean array of shape (n_coalitions, m) with coalition masks

    Returns:
        Weight array of shape (n_coalitions,)
    """
    m = Z.shape[1]
    coalition_sizes = Z.sum(axis=1).astype(int)

    target_weight_per_size = np.zeros(m + 1)
    for s in range(1, m):
        target_weight_per_size[s] = (m - 1) / (s * (m - s))

    total_target = target_weight_per_size.sum()
    target_weight_per_size /= total_target

    size_counts = np.bincount(coalition_sizes, minlength=m + 1)

    weight_per_coalition_by_size = np.zeros(m + 1)
    for s in range(1, m):
        if size_counts[s] > 0:
            weight_per_coalition_by_size[s] = target_weight_per_size[s] / size_counts[s]

    weights = weight_per_coalition_by_size[coalition_sizes]

    is_empty_or_full = (coalition_sizes == 0) | (coalition_sizes == m)
    weights[is_empty_or_full] = 1e10

    return weights
