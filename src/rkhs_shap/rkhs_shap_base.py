from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from math import comb

import numpy as np
import torch
from gpytorch.kernels import Kernel
from sklearn.linear_model import Ridge
from torch import Tensor
from tqdm import tqdm

from rkhs_shap.sampling import (
    sample_coalitions_full,
    sample_coalitions_weighted,
)
from rkhs_shap.subset_kernel import SubsetKernel
from rkhs_shap.utils import freeze_parameters, to_tensor


class RKHSSHAPBase(ABC):
    """Base class with shared fit method for RKHS-SHAP implementations"""

    n: int
    m: int
    X: Tensor
    y: Tensor
    cme_reg: Tensor
    mean_function: Callable[[Tensor], Tensor]
    kernel: Kernel
    krr_weights: Tensor
    ypred: Tensor
    rmse: float
    reference: float

    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        kernel: Kernel,
        cme_reg: float,
        mean_function: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize common RKHS-SHAP attributes.

        Args:
            X: Training features of shape (n, m)
            y: Training targets of shape (n,) or (n, 1)
            kernel: Fitted kernel (e.g., RBFKernel, MaternKernel)
            cme_reg: Regularization for conditional/marginal mean embeddings
            mean_function: Optional mean function m(x). If provided, KRR will fit
                residuals (y - m(X)) and predictions will be m(x) + k(x,X)α.
        """
        self.n, self.m = X.shape
        self.X, self.y = X.double(), y.double()
        self.cme_reg = to_tensor(cme_reg)
        self.mean_function = (
            mean_function if mean_function else lambda x: torch.zeros(x.shape[0])
        )
        self.kernel = deepcopy(kernel)
        freeze_parameters(self.kernel)

    @abstractmethod
    def _value_observation(self, z: np.ndarray, X_test: Tensor) -> Tensor:
        """Compute observational Shapley value function for coalition z.

        Args:
            z: Binary coalition vector of shape (m,) indicating active features
            X_test: Test points of shape (n_test, m)

        Returns:
            Value function evaluated at X_test, shape (1, n_test)
        """
        ...

    @abstractmethod
    def _value_intervention(self, z: np.ndarray, X_test: Tensor) -> Tensor:
        """Compute interventional Shapley value function for coalition z.

        Args:
            z: Binary coalition vector of shape (m,) indicating active features
            X_test: Test points of shape (n_test, m)

        Returns:
            Value function evaluated at X_test, shape (1, n_test)
        """
        ...

    def _eval_mean(self, X: Tensor) -> Tensor:
        """Evaluate mean function with proper shape handling."""
        with torch.inference_mode():
            mean = self.mean_function(X).detach()
        if mean.dim() == 0:
            mean = mean.expand(X.shape[0])
        return mean

    def _get_subset_kernels(self, z: np.ndarray):
        """Extract coalition and complement kernels from binary coalition vector.

        Args:
            z: Binary coalition vector of shape (m,) indicating active features

        Returns:
            Tuple of (S_kernel, Sc_kernel) - SubsetKernel instances for coalition and complement
        """
        S = np.where(z)[0]
        Sc = np.where(~z)[0]
        S_kernel = SubsetKernel(self.kernel, subset_dims=S)
        Sc_kernel = SubsetKernel(self.kernel, subset_dims=Sc)
        return S_kernel, Sc_kernel

    def fit(
        self,
        X_test: Tensor,
        method: str,
        sample_method: str,
        num_samples: int = 100,
        wls_reg: float = 0.01,  # TODO: consider decreasing the default
    ) -> np.ndarray:
        """Compute RKHS-SHAP values for test points.

        Args:
            X_test: Test points to explain, shape (n_test, m)
            method: "O" (Observational) or "I" (Interventional) Shapley values
            sample_method: Sampling strategy for coalitions:
                - "weighted": Monte Carlo sampling weighted by Shapley kernel
                - "full" or None: Enumerate all 2^m coalitions
            num_samples: Number of coalition samples (if using MC sampling)
            wls_reg: Regularization for weighted least squares fitting

        Returns:
            SHAP values of shape (n_test, m)
        """
        m = self.m
        if sample_method == "weighted":
            Z, is_sampled = sample_coalitions_weighted(m, num_samples)
        elif sample_method == "full":
            Z, is_sampled = sample_coalitions_full(m)
        else:
            raise ValueError("sample_method must be either 'weighted' or 'full'")

        n_coalitions = Z.shape[0]
        n_test = X_test.shape[0]
        Y_target = np.zeros((n_coalitions, n_test))

        coalition_sizes = Z.sum(axis=1)
        is_empty_or_full = (coalition_sizes == 0) | (coalition_sizes == m)

        binomial_coeffs = np.array([comb(m, int(s)) for s in coalition_sizes])

        # Compute Shapley kernel weights only for non-trivial coalitions
        shapley_kernel = np.zeros_like(coalition_sizes, dtype=float)
        non_trivial = ~is_empty_or_full
        shapley_kernel[non_trivial] = (m - 1) / (
            binomial_coeffs[non_trivial]
            * coalition_sizes[non_trivial]
            * (m - coalition_sizes[non_trivial])
        )

        # Importance sampling correction only for randomly sampled coalitions
        sampling_correction = np.ones(n_coalitions)

        if sample_method == "MC":
            # For MC: correct for sampling bias only on randomly sampled coalitions
            # Sampling uses P(size s) ∝ (m-1)/(s*(m-s)), then uniform within size
            # P(coalition of size s) = P(size s) / C(m,s)
            # For unbiased estimation: weight = shapley_kernel / sampling_probability

            prob_per_size = np.array([(m - 1) / (s * (m - s)) for s in range(1, m)])
            prob_per_size /= prob_per_size.sum()
            size_to_prob = {s: prob_per_size[s - 1] for s in range(1, m)}

            # Apply correction ONLY to randomly sampled coalitions
            for i in range(n_coalitions):
                if is_sampled[i] and 0 < coalition_sizes[i] < m:
                    s = int(coalition_sizes[i])
                    if s in size_to_prob:
                        sampling_correction[i] = binomial_coeffs[i] / size_to_prob[s]

        weights = np.where(is_empty_or_full, 1e10, shapley_kernel * sampling_correction)

        if method == "O":
            value_fn = self._value_observation
        elif method == "I":
            value_fn = self._value_intervention
        else:
            raise ValueError("Must be either interventional or observational")

        for idx, row in enumerate(tqdm(Z)):
            Y_target[idx, :] = value_fn(row, X_test)

        clf = Ridge(wls_reg)
        clf.fit(Z, Y_target, sample_weight=weights)

        shap_values = clf.coef_

        return shap_values
