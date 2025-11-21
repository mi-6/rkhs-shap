from collections.abc import Callable
from copy import deepcopy

import numpy as np
import torch
from gpytorch.kernels import Kernel
from torch import Tensor

from rkhs_shap.kernel_approx import Nystroem
from rkhs_shap.rkhs_shap_base import RKHSSHAPBase
from rkhs_shap.subset_kernel import SubsetKernel
from rkhs_shap.utils import freeze_parameters, to_tensor


class RKHSSHAPApprox(RKHSSHAPBase):
    """Implement the RKHS SHAP algorithm with Nyström kernel approximation"""

    def __init__(
        self,
        X: Tensor,
        y: Tensor,
        kernel: Kernel,
        noise_var: float = 1e-2,
        cme_reg: float = 1e-3,
        n_components: int = 100,
        mean_function: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize approximate RKHS-SHAP with Nyström-approximated Kernel Ridge Regression.

        Args:
            X: Training features of shape (n, m)
            y: Training targets of shape (n,) or (n, 1)
            kernel: Fitted kernel (e.g., RBFKernel, MaternKernel)
            noise_var: KRR regularization parameter. Corresponds to the noise variance
                in Gaussian Process formulation. If you've fit a GP model, set this to
                model.likelihood.noise_covar.noise for equivalent predictions.
            cme_reg: Regularization for conditional/marginal mean embeddings
            n_components: Number of landmark points for Nyström approximation
            mean_function: Optional mean function m(x). If provided, KRR will fit
                residuals (y - m(X)) and predictions will be m(x) + k(x,X)α.
                Pass model.mean_module from a fitted GP to ensure prediction alignment.

        Note:
            When using GPyTorch kernels with n>800, GPyTorch switches from Cholesky
            decomposition to conjugate gradient (CG) solver, which can cause numerical
            differences.
        """
        self.n, self.m = X.shape
        self.X, self.y = X.float(), y
        self.cme_reg = to_tensor(cme_reg)
        self.mean_function = (
            mean_function if mean_function else lambda x: torch.zeros(x.shape[0])
        )

        self.kernel = deepcopy(kernel)
        freeze_parameters(self.kernel)

        # Run Nyström approximation and Kernel Ridge Regression on residuals
        self.nystroem = Nystroem(kernel=self.kernel, n_components=n_components)
        self.nystroem.fit(self.X)
        Z = self.nystroem.transform(self.X)
        K_train = Z @ Z.T
        n = K_train.shape[0]

        mean_train = self._eval_mean(self.X)
        y_centered = self.y.reshape(-1, 1) - mean_train.reshape(-1, 1)

        jitter = 1e-6
        K_reg = K_train + (noise_var + jitter) * torch.eye(n, dtype=K_train.dtype)

        krr_weights: Tensor = torch.linalg.solve(K_reg, y_centered)

        self.krr_weights = krr_weights.reshape(-1, 1)
        self.ypred = K_train @ krr_weights + mean_train.reshape(-1, 1)
        self.rmse = torch.sqrt(
            torch.mean((self.ypred - self.y.reshape(-1, 1)) ** 2)
        ).item()
        self.reference = self.ypred.mean().item()
        self.Z = Z

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
            return 0

        if z.sum() == self.m:
            ypred_test = self.krr_weights.T @ self.Z @ self.nystroem.transform(
                X_test
            ).T + self._eval_mean(X_test).unsqueeze(0)
            return ypred_test - self.reference

        # Naming conventions:
        # S ⊆ {1, 2, ..., m} is a subset of all m features (coalition)
        # S represents which features are "present" or "known" when computing the value function
        # Sp (S prime) refers to test points where we evaluate the value function
        # Sc (S complement) is the complement set - features NOT in S
        S = np.where(z)[0]
        Sc = np.where(~z)[0]

        # Create subset kernels for the coalition and its complement
        S_kernel = SubsetKernel(self.kernel, subset_dims=S)
        Sc_kernel = SubsetKernel(self.kernel, subset_dims=Sc)

        # Transform using Nyström with subsetted data
        Z_S = self._nystroem_transform_subset(self.X, S_kernel)
        Z_S_new = self._nystroem_transform_subset(X_test, S_kernel)
        K_SSp = Z_S @ Z_S_new.T

        Z_Sc = self._nystroem_transform_subset(self.X, Sc_kernel)
        K_Sc = Z_Sc @ Z_Sc.T
        KME_mat = K_Sc.mean(dim=1, keepdim=True).repeat(1, n_test)

        ypred_partial = self.krr_weights.T @ (K_SSp * KME_mat) + self._eval_mean(
            X_test
        ).unsqueeze(0)
        return ypred_partial - self.reference

    def _nystroem_transform_subset(
        self, X: Tensor, subset_kernel: SubsetKernel
    ) -> Tensor:
        """Apply Nyström transformation with a subset kernel.

        Note: SubsetKernel handles the subsetting internally, so we pass full-dimensional
        data and let the kernel handle feature selection.
        """
        X_tensor = to_tensor(X)

        # Apply Nyström transformation with subset kernel
        # SubsetKernel will internally select the active dimensions
        ZT = (
            subset_kernel(self.nystroem.landmarks)
            .add_jitter()
            .cholesky()
            .solve(subset_kernel(self.nystroem.landmarks, X_tensor).to_dense())
        )
        return ZT.T

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
            return 0

        if z.sum() == self.m:
            ypred_test = self.krr_weights.T @ self.Z @ self.nystroem.transform(
                X_test
            ).T + self._eval_mean(X_test).unsqueeze(0)
            return ypred_test - self.reference

        S = np.where(z)[0]
        Sc = np.where(~z)[0]

        # Create subset kernels for the coalition and its complement
        S_kernel = SubsetKernel(self.kernel, subset_dims=S)
        Sc_kernel = SubsetKernel(self.kernel, subset_dims=Sc)

        Z_S = self._nystroem_transform_subset(self.X, S_kernel)
        Z_S_new = self._nystroem_transform_subset(X_test, S_kernel)
        K_SSp = Z_S @ Z_S_new.T

        Z_Sc = self._nystroem_transform_subset(self.X, Sc_kernel)
        K_Sc = Z_Sc @ Z_Sc.T

        # Conditional Mean Embedding operator: maps complement features to coalition features
        ZS_gram = Z_S.T @ Z_S
        n_comp = ZS_gram.shape[0]
        Xi_S = torch.linalg.solve(ZS_gram + self.cme_reg * torch.eye(n_comp), Z_S_new.T)

        ypred_partial = self.krr_weights.T @ (
            K_SSp * (K_Sc @ Z_S @ Xi_S)
        ) + self._eval_mean(X_test).unsqueeze(0)
        return ypred_partial - self.reference
