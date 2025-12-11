from itertools import combinations

import numpy as np


def sample_coalitions_full(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate all 2^m coalitions exhaustively.

    Args:
        m: Number of features

    Returns:
        Tuple of (Z, is_sampled) where:
        - Z: Boolean array of shape (2^m, m) with all possible coalitions
        - is_sampled: Boolean array of shape (2^m,) indicating which coalitions
          were randomly sampled (all False for exhaustive enumeration)
    """
    Z = np.zeros((2**m, m), dtype=bool)

    count = 1
    for s in range(1, m):
        for combo in combinations(range(m), s):
            Z[count, combo] = True
            count += 1

    Z[-1, :] = True
    is_sampled = np.zeros(2**m, dtype=bool)
    return Z, is_sampled


def sample_coalitions_weighted(m: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample unique coalitions according to Shapley kernel weights.

    First samples coalition sizes proportional to their Shapley kernel weights,
    then uniformly samples a specific coalition of that size. Duplicates are
    removed to ensure each coalition appears exactly once.

    Args:
        m: Number of features
        n_samples: Target number of unique coalition samples

    Returns:
        Tuple of (Z, is_sampled) where:
        - Z: Boolean array of shape (n_unique + 2, m) with unique sampled coalitions.
          Last two rows are empty and full coalitions.
        - is_sampled: Boolean array of shape (n_unique + 2,) indicating which
          coalitions were randomly sampled (True for all except empty/full)
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

            if key not in seen:
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

    is_sampled = np.ones(n_unique + 2, dtype=bool)
    is_sampled[-2:] = False

    return Z, is_sampled


def _shapley_kernel_weight(m: int, s: int) -> float:
    """Compute Shapley kernel weight for coalition size s.

    This is the weight used in the weighted least squares formulation
    of Shapley value estimation.
    """
    return (m - 1) / (s * (m - s))
