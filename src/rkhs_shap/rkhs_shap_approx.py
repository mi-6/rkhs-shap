###############################
# KernelSHAP4K For Regression #
###############################

from copy import deepcopy

import numpy as np
import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import lazify
from scipy.special import binom
from sklearn.linear_model import Ridge
from torch import Tensor
from tqdm import tqdm

from rkhs_shap.kernel_approx import Nystroem_gpytorch
from rkhs_shap.sampling import (
    generate_full_Z,
    large_scale_sample_alternative,
    subset_full_Z,
)
from rkhs_shap.subset_kernel import SubsetKernel
from rkhs_shap.utils import freeze_parameters


class RKHSSHAP_Approx(object):
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
        self.nystroem = Nystroem_gpytorch(kernel=self.kernel, n_components=n_components)
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
        Z_S = self._nystroem_transform_subset(self.X.numpy(), S_kernel, S)
        Z_S_new = self._nystroem_transform_subset(X_test, S_kernel, S)
        K_SSp = Z_S @ Z_S_new.T

        Z_Sc = self._nystroem_transform_subset(self.X.numpy(), Sc_kernel, Sc)
        K_Sc = Z_Sc @ Z_Sc.T
        KME_mat = K_Sc.mean(dim=1, keepdim=True).repeat(1, n_test)

        return self.krr_weights.T @ (K_SSp * KME_mat) - self.reference

    def _nystroem_transform_subset(
        self, X: np.ndarray, subset_kernel: SubsetKernel, active_indices: np.ndarray
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

        Z_S = self._nystroem_transform_subset(self.X.numpy(), S_kernel, S)
        Z_S_new = self._nystroem_transform_subset(X_test, S_kernel, S)
        K_SSp = Z_S @ Z_S_new.T

        Z_Sc = self._nystroem_transform_subset(self.X.numpy(), Sc_kernel, Sc)
        K_Sc = Z_Sc @ Z_Sc.T

        # Conditional Mean Embedding operator: maps complement features to coalition features
        Xi_S = lazify(Z_S.T @ Z_S).add_diag(self.cme_reg).inv_matmul(Z_S_new.T)

        return self.krr_weights.T @ (K_SSp * (K_Sc @ Z_S @ Xi_S)) - self.reference

    def fit(
        self,
        X_test: Tensor | np.ndarray,
        method: str,
        sample_method: str,
        num_samples: int = 100,
        wls_reg: float = 1e-10,
    ) -> np.ndarray:
        """Compute RKHS-SHAP values for test points.

        Args:
            X_test: Test points to explain, shape (n_test, m)
            method: "O" (Observational) or "I" (Interventional) Shapley values
            sample_method: Sampling strategy for coalitions:
                - "MC": Monte Carlo sampling weighted by Shapley kernel
                - "MC2": Sample from full coalition space
                - "full" or None: Enumerate all 2^m coalitions
            num_samples: Number of coalition samples (if using MC sampling)
            wls_reg: Regularization for weighted least squares fitting

        Returns:
            SHAP values of shape (n_test, m)
        """
        if isinstance(X_test, Tensor):
            X_test = X_test.numpy()

        if sample_method == "MC":
            Z = large_scale_sample_alternative(self.m, num_samples)
        elif sample_method == "MC2":
            Z = generate_full_Z(self.m)
            Z = subset_full_Z(Z, samples=num_samples)
        else:
            Z = generate_full_Z(self.m)

        n_coalitions = Z.shape[0]
        n_test = X_test.shape[0]
        Y_target = np.zeros((n_coalitions, n_test))
        count = 0
        weights = []

        for row in tqdm(Z):
            if np.sum(row) == 0 or np.sum(row) == self.m:
                weights.append(1e5)
            else:
                z = row
                weights.append(
                    (self.m - 1)
                    / (binom(self.m, np.sum(z)) * np.sum(z) * (self.m - np.sum(z)))
                )

            if method == "O":
                Y_target[count, :] = self._value_observation(row, X_test)
            elif method == "I":
                Y_target[count, :] = self._value_intervention(row, X_test)
            else:
                raise ValueError("Must be either interventional or observational")

            count += 1

        clf = Ridge(wls_reg)
        clf.fit(Z, Y_target, sample_weight=weights)

        return clf.coef_
