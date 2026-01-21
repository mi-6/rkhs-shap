from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import torch
from gpytorch.kernels import Kernel
from torch import Tensor

from rkhs_shap.sampling import (
    compute_kernelshap_weights,
    sample_coalitions_full,
    sample_coalitions_weighted,
)
from rkhs_shap.subset_kernel import SubsetKernel
from rkhs_shap.utils import freeze_parameters, to_tensor


def weighted_ridge(
    X: Tensor, y: Tensor, sample_weight: Tensor, alpha: float = 1.0
) -> Tensor:
    """Fit weighted ridge regression using closed-form solution.

    Solves: argmin_w ||W^(1/2)(Xw - y)||^2 + alpha||w||^2
    where W = diag(sample_weight)

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target matrix of shape (n_samples, n_targets)
        sample_weight: Sample weights of shape (n_samples,)
        alpha: Regularization strength

    Returns:
        Coefficients of shape (n_targets, n_features)
    """
    W_sqrt = torch.sqrt(sample_weight).unsqueeze(1)

    X_weighted = X * W_sqrt
    y_weighted = y * W_sqrt

    n_features = X.shape[1]
    regularizer = alpha * torch.eye(n_features, dtype=torch.float64)

    # Solve (X^T W X + alpha*I) w = X^T W y
    XtWX = X_weighted.T @ X_weighted
    Xtwy = X_weighted.T @ y_weighted

    coef = torch.linalg.solve(XtWX + regularizer, Xtwy)

    return coef.T


class RKHSSHAPBase(ABC):
    """Base class with shared fit method for RKHS-SHAP implementations"""

    _n: int
    _m: int
    _X: Tensor
    _y: Tensor
    _cme_reg: Tensor
    _mean_function: Callable[[Tensor], Tensor]
    _kernel: Kernel
    _krr_weights: Tensor
    _ypred: Tensor
    _rmse: float
    _reference: float

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
        self._n, self._m = X.shape
        self._X, self._y = X.double(), y.double()
        self._cme_reg = to_tensor(cme_reg)
        self._mean_function = (
            mean_function if mean_function else lambda x: torch.zeros(x.shape[0])
        )
        self._kernel = deepcopy(kernel)
        freeze_parameters(self._kernel)

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
            mean = self._mean_function(X).detach()
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
        S_kernel = SubsetKernel(self._kernel, subset_dims=S)
        Sc_kernel = SubsetKernel(self._kernel, subset_dims=Sc)
        return S_kernel, Sc_kernel

    def fit(
        self,
        X_test: Tensor,
        method: str,
        sample_method: str,
        num_samples: int = 100,
        wls_reg: float = 1e-8,
        rng: np.random.Generator | None = None,
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
            rng: Optional numpy random generator. If None, uses non-deterministic randomness.

        Returns:
            SHAP values of shape (n_test, m)
        """
        m = self._m
        if sample_method == "weighted":
            Z = sample_coalitions_weighted(m, num_samples, rng=rng)
        elif sample_method == "full":
            Z = sample_coalitions_full(m)
        else:
            raise ValueError("sample_method must be either 'weighted' or 'full'")

        n_coalitions = Z.shape[0]
        n_test = X_test.shape[0]
        Y_target = torch.zeros((n_coalitions, n_test), dtype=torch.float64)

        weights = compute_kernelshap_weights(Z)
        weights = torch.from_numpy(weights)

        if method == "O":
            value_fn = self._value_observation
        elif method == "I":
            value_fn = self._value_intervention
        else:
            raise ValueError("Must be either interventional or observational")

        for idx, row in enumerate(Z):
            Y_target[idx, :] = value_fn(row, X_test)

        Z = torch.from_numpy(Z)
        shap_values = weighted_ridge(Z, Y_target, sample_weight=weights, alpha=wls_reg)

        return shap_values.numpy()
