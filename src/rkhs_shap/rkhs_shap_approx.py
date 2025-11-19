###############################
# KernelSHAP4K For Regression #
###############################

from copy import deepcopy

import numpy as np
import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import lazify
from torch import Tensor

from rkhs_shap.kernel_approx import Nystroem
from rkhs_shap.rkhs_shap_base import RKHSSHAPBase
from rkhs_shap.subset_kernel import SubsetKernel
from rkhs_shap.utils import freeze_parameters


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
        """
        self.n, self.m = X.shape
        self.X, self.y = X.float(), y

        noise_var = torch.tensor(noise_var, dtype=torch.float32)
        cme_reg = torch.tensor(cme_reg, dtype=torch.float32)
        self.cme_reg = cme_reg

        self.kernel = deepcopy(kernel)
        freeze_parameters(self.kernel)

        # Run Nyström approximation and Kernel Ridge Regression
        self.nystroem = Nystroem(kernel=self.kernel, n_components=n_components)
        self.nystroem.fit(self.X.numpy())
        Z = self.nystroem.transform(self.X.numpy())
        K_train = Z @ Z.T

        krr_weights: Tensor = (
            lazify(K_train).add_diag(noise_var).inv_matmul(self.y.reshape(-1, 1))
        )

        self.krr_weights = krr_weights.reshape(-1, 1)
        self.ypred = K_train @ krr_weights
        self.rmse = torch.sqrt(
            torch.mean((self.ypred - self.y.reshape(-1, 1)) ** 2)
        ).item()
        self.reference = self.ypred.mean().item()
        self.Z = Z

    def _value_intervention(self, z: np.ndarray, X_test: np.ndarray) -> Tensor:
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
            ypred_test = self.krr_weights.T @ self.Z @ self.nystroem.transform(X_test).T
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
        Z_S = self._nystroem_transform_subset(self.X.numpy(), S_kernel)
        Z_S_new = self._nystroem_transform_subset(X_test, S_kernel)
        K_SSp = Z_S @ Z_S_new.T

        Z_Sc = self._nystroem_transform_subset(self.X.numpy(), Sc_kernel)
        K_Sc = Z_Sc @ Z_Sc.T
        KME_mat = K_Sc.mean(dim=1, keepdim=True).repeat(1, n_test)

        return self.krr_weights.T @ (K_SSp * KME_mat) - self.reference

    def _nystroem_transform_subset(
        self, X: np.ndarray, subset_kernel: SubsetKernel
    ) -> Tensor:
        """Apply Nyström transformation with a subset kernel.

        Note: SubsetKernel handles the subsetting internally, so we pass full-dimensional
        data and let the kernel handle feature selection.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Apply Nyström transformation with subset kernel
        # SubsetKernel will internally select the active dimensions
        ZT = (
            subset_kernel(self.nystroem.landmarks)
            .add_jitter()
            .cholesky()
            .inv_matmul(subset_kernel(self.nystroem.landmarks, X_tensor).evaluate())
        )
        return ZT.T

    def _value_observation(self, z: np.ndarray, X_test: np.ndarray) -> Tensor:
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
            ypred_test = self.krr_weights.T @ self.Z @ self.nystroem.transform(X_test).T
            return ypred_test - self.reference

        S = np.where(z)[0]
        Sc = np.where(~z)[0]

        # Create subset kernels for the coalition and its complement
        S_kernel = SubsetKernel(self.kernel, subset_dims=S)
        Sc_kernel = SubsetKernel(self.kernel, subset_dims=Sc)

        Z_S = self._nystroem_transform_subset(self.X.numpy(), S_kernel)
        Z_S_new = self._nystroem_transform_subset(X_test, S_kernel)
        K_SSp = Z_S @ Z_S_new.T

        Z_Sc = self._nystroem_transform_subset(self.X.numpy(), Sc_kernel)
        K_Sc = Z_Sc @ Z_Sc.T

        # Conditional Mean Embedding operator: maps complement features to coalition features
        Xi_S = lazify(Z_S.T @ Z_S).add_diag(self.cme_reg).inv_matmul(Z_S_new.T)

        return self.krr_weights.T @ (K_SSp * (K_Sc @ Z_S @ Xi_S)) - self.reference
