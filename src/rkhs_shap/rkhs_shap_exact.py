from collections.abc import Callable
from copy import deepcopy

import numpy as np
import torch
from gpytorch.kernels import Kernel
from torch import Tensor

from rkhs_shap.rkhs_shap_base import RKHSSHAPBase
from rkhs_shap.subset_kernel import SubsetKernel
from rkhs_shap.utils import freeze_parameters, to_tensor


class RKHSSHAP(RKHSSHAPBase):
    """Implement the exact RKHS SHAP algorithm with no kernel approximation"""

    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        kernel: Kernel,
        noise_var: float = 1e-2,
        cme_reg: float = 1e-4,
        mean_function: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize exact RKHS-SHAP with Kernel Ridge Regression.

        Args:
            X: Training features of shape (n, m)
            y: Training targets of shape (n,) or (n, 1)
            kernel: Fitted kernel (e.g., RBFKernel, MaternKernel)
            noise_var: KRR regularization parameter. Corresponds to the noise variance
                in Gaussian Process formulation. If you've fit a GP model, set this to
                model.likelihood.noise_covar.noise for equivalent predictions.
            cme_reg: Regularization for conditional/marginal mean embeddings
            mean_function: Optional mean function m(x). If provided, KRR will fit
                residuals (y - m(X)) and predictions will be m(x) + k(x,X)α.
                Pass model.mean_module from a fitted GP to ensure prediction alignment.
        """
        self.n, self.m = X.shape
        self.X, self.y = X.float(), y

        self.cme_reg = to_tensor(cme_reg)
        self.mean_function = (
            mean_function if mean_function else lambda x: torch.zeros(x.shape[0])
        )

        self.kernel = deepcopy(kernel)
        freeze_parameters(self.kernel)

        # Run Kernel Ridge Regression on residuals (y - mean(X))
        K_train = self.kernel(self.X).to_dense()
        mean_train = self._eval_mean(self.X)
        y_centered = self.y - mean_train

        krr_weights: Tensor = torch.linalg.solve(
            K_train + noise_var * torch.eye(self.n), y_centered
        )
        self.krr_weights = krr_weights.reshape(-1, 1)

        # Predictions include mean function
        self.ypred = K_train @ krr_weights + mean_train
        self.rmse = torch.sqrt(torch.mean((self.ypred - self.y) ** 2)).item()
        self.reference = self.ypred.mean().item()

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

    def _eval_mean(self, X: Tensor) -> Tensor:
        """Evaluate mean function with proper shape handling."""
        with torch.no_grad():
            mean = self.mean_function(X).detach()
        if mean.dim() == 0:
            mean = mean.repeat(X.shape[0])
        return mean

    def _value_intervention(self, z: np.ndarray, X_test: Tensor) -> Tensor:
        """Compute interventional Shapley value function for coalition z.

        Computes E[f(X) | X_S = x_S] - E[f(X)] where S is the coalition defined by z.
        Uses Kernel Mean Embedding (KME) to marginalize over complement features.

        Args:
            z: Binary coalition vector of shape (m,) indicating active features
            X_test: Test points of shape (n_test, m)

        Returns:
            Value function evaluated at X_test, shape (1, n_test)
        """
        n_test = X_test.shape[0]

        if z.sum() == 0:
            return torch.zeros(1, X_test.shape[0])

        if z.sum() == self.m:
            # If all features are active, return full prediction
            K_test = self.kernel(self.X, X_test).to_dense()
            ypred_test = self.krr_weights.T @ K_test + self._eval_mean(
                X_test
            ).unsqueeze(0)
            return ypred_test - self.reference

        # Naming conventions:
        # S ⊆ {1, 2, ..., m} is a subset of all m features (coalition)
        # S represents which features are "present" or "known" when computing the value function
        # Sp (S prime) refers to test points where we evaluate the value function
        # Sc (S complement) is the complement set - features NOT in S
        S_kernel, Sc_kernel = self._get_subset_kernels(z)

        K_SSp = S_kernel(self.X, X_test).to_dense()
        K_Sc = Sc_kernel(self.X, self.X).to_dense()
        KME_mat = K_Sc.mean(axis=1, keepdim=True).repeat(1, n_test)

        ypred_partial = self.krr_weights.T @ (K_SSp * KME_mat) + self._eval_mean(
            X_test
        ).unsqueeze(0)
        return ypred_partial - self.reference

    def _value_observation(self, z: np.ndarray, X_test: Tensor) -> Tensor:
        """Compute observational Shapley value function for coalition z.

        Computes E[f(X) | X_S = x_S] - E[f(X)] where S is the coalition defined by z.
        Uses Conditional Mean Embedding (CME) to condition on observed features.

        Args:
            z: Binary coalition vector of shape (m,) indicating active features
            X_test: Test points of shape (n_test, m)

        Returns:
            Value function evaluated at X_test, shape (1, n_test)
        """
        if z.sum() == 0:
            return torch.zeros(1, X_test.shape[0])

        if z.sum() == self.m:
            K_test = self.kernel(self.X, X_test).to_dense()
            ypred_test = self.krr_weights.T @ K_test + self._eval_mean(
                X_test
            ).unsqueeze(0)
            return ypred_test - self.reference

        S_kernel, Sc_kernel = self._get_subset_kernels(z)

        K_SSp = S_kernel(self.X, X_test).to_dense().float()
        K_Sc = Sc_kernel(self.X, self.X).to_dense()
        K_SS = S_kernel(self.X, self.X).to_dense()

        # Conditional Mean Embedding operator: maps complement features to coalition features
        Xi_S = torch.linalg.solve(
            K_SS + self.n * self.cme_reg * torch.eye(self.n), K_Sc
        ).T

        ypred_partial = self.krr_weights.T @ (K_SSp * (Xi_S @ K_SSp)) + self._eval_mean(
            X_test
        ).unsqueeze(0)
        return ypred_partial - self.reference
