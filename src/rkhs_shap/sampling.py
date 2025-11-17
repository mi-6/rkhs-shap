from itertools import combinations
from scipy.special import binom
import numpy as np

"""This file contains code for sampling coalitions to run the weighed least square approach in RKHS-SHAP."""


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


#################
# MCMC Sampling #
#################


def _get_weights(s: int, m: int) -> float:
    """The unnormalised probability weight to sample a particular permutation

    Args:
        s: size
        m: number of active features

    Returns:
        Unnormalised probability weight
    """
    return (m - 1) / ((binom(m, s) * s * (m - s)))


def _propose_func(z: np.ndarray) -> np.ndarray:
    """Propose a new MCMC state by flipping one element.

    WARNING: This function modifies z in-place.

    Args:
        z: Current state vector with values in {-1, 1}

    Returns:
        The modified array (same object as input)
    """
    m = len(z)
    index = np.random.choice(m)
    z[index] = -z[index]
    return z


def generate_samples_Z(m: int, mcmc_run: int, warm_up_cut: int) -> np.ndarray:
    """Generate Samples of zs' for the KS4K

    Args:
        m: number of features
        mcmc_run: number of MCMC runs
        warm_up_cut: number of warmup to discard (initial runs not converging to stationary dist)

    Returns:
        Z: samples of zs' for KS4K returned as the Z matrix in KernelSHAP
    """
    z_init = np.array([np.random.choice([1, -1]) for _ in range(m)])
    z_vec = []
    z_vec.append(z_init)

    count = 0
    while count <= mcmc_run:
        propose = _propose_func(z_vec[count].copy())
        propose_s = (propose == 1).sum()
        current_s = (z_vec[count] == 1).sum()

        # These two cases should have probability 0 reaching there
        if propose_s == m or propose_s == 0:
            continue
        else:
            # the alpha score in MCMC
            a_t = np.min(
                [1, _get_weights(int(propose_s), int(m)) / _get_weights(int(current_s), int(m))]
            )

            flip = np.random.uniform()

            if flip <= a_t:
                # Append the proposed one
                z_vec.append(propose)
                count += 1
            else:
                # Append the current one
                z_vec.append(z_vec[count])
                count += 1

    z_vec = z_vec[warm_up_cut:]
    z_vec.append(np.ones_like(propose))
    z_vec.append(-np.ones_like(propose))
    z_vec = (np.array(z_vec) + 1) / 2

    return z_vec.astype(np.bool_)
