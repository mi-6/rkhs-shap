from collections.abc import Callable

import numpy as np
import torch
from gpytorch.kernels import Kernel
from torch import Tensor

from rkhs_shap.kernel_approx import Nystroem
from rkhs_shap.rkhs_shap_base import RKHSSHAPBase
from rkhs_shap.subset_kernel import SubsetKernel


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
        random_state: int | None = 42,
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
            random_state: Random state for KMeans clustering in Nyström approximation.
                If None, uses 42 for reproducibility.

        Note:
            When using GPyTorch kernels with n>800, GPyTorch switches from Cholesky
            decomposition to conjugate gradient (CG) solver, which can cause numerical
            differences.
        """
        super().__init__(X, y, kernel, cme_reg, mean_function)

        # Run Nyström approximation and Kernel Ridge Regression on residuals
        self.nystroem = Nystroem(
            kernel=self.kernel, n_components=n_components, random_state=random_state
        )
        self.nystroem.fit(self.X)
        Z = self.nystroem.transform(self.X)
        K_train = Z @ Z.T
        n = K_train.shape[0]

        mean_train = self._eval_mean(self.X)
        y_centered = (self.y - mean_train).reshape(-1, 1)

        jitter = 1e-8
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
        coalition_size = z.sum()

        if coalition_size == 0:
            return torch.zeros(1, X_test.shape[0])

        if coalition_size == self.m:
            return self._compute_full_prediction(X_test)

        # Naming conventions:
        # S ⊆ {1, 2, ..., m} is a subset of all m features (coalition)
        # S represents which features are "present" or "known" when computing the value function
        # Sp (S prime) refers to test points where we evaluate the value function
        # Sc (S complement) is the complement set - features NOT in S
        S_kernel, Sc_kernel = self._get_subset_kernels(z)

        # Transform using Nyström with subsetted data
        Z_S = self._nystroem_transform_subset(self.X, S_kernel)
        Z_S_new = self._nystroem_transform_subset(X_test, S_kernel)
        K_SSp = Z_S @ Z_S_new.T

        Z_Sc = self._nystroem_transform_subset(self.X, Sc_kernel)
        K_Sc = Z_Sc @ Z_Sc.T
        KME_vec = K_Sc.mean(dim=1, keepdim=True)

        ypred_partial = self.krr_weights.T @ (K_SSp * KME_vec) + self._eval_mean(
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
        # Apply Nyström transformation with subset kernel
        # SubsetKernel will internally select the active dimensions
        ZT = (
            subset_kernel(self.nystroem.landmarks)
            .add_jitter()
            .cholesky()
            .solve(subset_kernel(self.nystroem.landmarks, X).to_dense())
        )
        return ZT.T

    def _compute_full_prediction(self, X_test: Tensor) -> Tensor:
        """Compute full prediction for all features active (z.sum() == m).

        Args:
            X_test: Test points of shape (n_test, m)

        Returns:
            Prediction minus reference, shape (1, n_test)
        """
        ypred_test = self.krr_weights.T @ self.Z @ self.nystroem.transform(
            X_test
        ).T + self._eval_mean(X_test).unsqueeze(0)
        return ypred_test - self.reference

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
        coalition_size = z.sum()

        if coalition_size == 0:
            return torch.zeros(1, X_test.shape[0])

        if coalition_size == self.m:
            return self._compute_full_prediction(X_test)

        S_kernel, Sc_kernel = self._get_subset_kernels(z)

        Z_S = self._nystroem_transform_subset(self.X, S_kernel)
        Z_S_new = self._nystroem_transform_subset(X_test, S_kernel)
        K_SSp = Z_S @ Z_S_new.T

        Z_Sc = self._nystroem_transform_subset(self.X, Sc_kernel)
        K_Sc = Z_Sc @ Z_Sc.T

        # Conditional Mean Embedding operator: maps complement features to coalition features
        ZS_gram = Z_S.T @ Z_S
        n_comp = ZS_gram.shape[0]
        jitter = 1e-8
        Xi_S = torch.linalg.solve(
            ZS_gram + (self.cme_reg + jitter) * torch.eye(n_comp, dtype=ZS_gram.dtype),
            Z_S_new.T,
        )

        ypred_partial = self.krr_weights.T @ (
            K_SSp * (K_Sc @ Z_S @ Xi_S)
        ) + self._eval_mean(X_test).unsqueeze(0)
        return ypred_partial - self.reference
