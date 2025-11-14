###############################
# KernelSHAP4K For Regression #
###############################

import torch
from scipy.special import binom
import numpy as np
from gpytorch.kernels import Kernel
from copy import deepcopy

# from gpytorch.lazy import lazify
from sklearn.linear_model import Ridge
from numpy import sum
from tqdm import tqdm
from rkhs_shap.sampling import (
    large_scale_sample_alternative,
    generate_full_Z,
    subsetting_full_Z,
)
from rkhs_shap.subset_kernel import SubsetKernel
from rkhs_shap.utils import freeze_parameters


class RKHSSHAP(object):
    """Implement the exact RKHS SHAP algorithm with no kernel approximation"""

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        kernel: Kernel,
        noise_var: float = 1e-2,
        cme_reg: float = 1e-4,
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
        """
        self.n, self.m = X.shape
        self.X, self.y = X.float(), y

        noise_var = torch.tensor(noise_var, dtype=torch.float32)
        cme_reg = torch.tensor(cme_reg, dtype=torch.float32)
        self.cme_reg = cme_reg

        self.kernel = deepcopy(kernel)
        freeze_parameters(self.kernel)

        # Run Kernel Ridge Regression
        K_train = self.kernel(self.X)
        krr_weights = K_train.add_diag(noise_var).inv_matmul(self.y)
        self.krr_weights = krr_weights.reshape(-1, 1)
        self.y_pred: torch.Tensor = K_train @ krr_weights
        self.rmse = torch.sqrt(torch.mean(self.y_pred - self.y) ** 2)
        self.reference = self.y_pred.mean()

    def _value_intervention(
        self,
        z: np.ndarray | torch.Tensor,
        X_new: torch.Tensor,
    ) -> torch.Tensor:
        """Compute interventional Shapley value function for coalition z.

        Computes E[f(X) | X_S = x_S] - E[f(X)] where S is the coalition defined by z.
        Uses Kernel Mean Embedding (KME) to marginalize over complement features.

        Args:
            z: Binary coalition vector of shape (m,) indicating active features
            X_new: Test points of shape (n_new, m)

        Returns:
            Value function evaluated at X_new, shape (1, n_new)
        """
        # TODO accept only numpy array as argument
        z = z.cpu().numpy() if isinstance(z, torch.Tensor) else z
        n_new = X_new.shape[0]

        if z.sum() == 0:
            # If no features are active, return 0
            return 0

        if z.sum() == self.m:
            # If all features are active, return full prediction
            new_ypred = self.krr_weights.T @ self.kernel(self.X, X_new).evaluate()
            return new_ypred - self.reference

        coalition_dims = np.where(z)[0].tolist()
        complement_dims = np.where(~z)[0].tolist()

        k_S = SubsetKernel(self.kernel, subset_dims=coalition_dims)
        k_Sc = SubsetKernel(self.kernel, subset_dims=complement_dims)

        K_SSp = k_S(self.X, X_new).evaluate().float()
        K_Sc = k_Sc(self.X, self.X)

        KME_mat = K_Sc.evaluate().mean(axis=1)[:, np.newaxis] * torch.ones(
            (self.n, n_new)
        )

        return self.krr_weights.T @ (K_SSp * KME_mat) - self.reference

    def _value_observation(
        self,
        z: np.ndarray | torch.Tensor,
        X_new: torch.Tensor,
    ) -> torch.Tensor:
        """Compute observational Shapley value function for coalition z.

        Computes E[f(X) | X_S = x_S] - E[f(X)] where S is the coalition defined by z.
        Uses Conditional Mean Embedding (CME) to condition on observed features.

        Args:
            z: Binary coalition vector of shape (m,) indicating active features
            X_new: Test points of shape (n_new, m)

        Returns:
            Value function evaluated at X_new, shape (1, n_new)
        """
        z = z.cpu().numpy() if isinstance(z, torch.Tensor) else z

        if z.sum() == 0:
            return 0

        if z.sum() == self.m:
            new_ypred = self.krr_weights.T @ self.kernel(self.X, X_new).evaluate()
            return new_ypred - self.reference

        coalition_dims = np.where(z)[0].tolist()
        complement_dims = np.where(~z)[0].tolist()

        coalition_k = SubsetKernel(self.kernel, subset_dims=coalition_dims)
        complement_k = SubsetKernel(self.kernel, subset_dims=complement_dims)

        K_SSp = coalition_k(self.X, X_new).evaluate().float()
        K_Sc = complement_k(self.X, self.X)
        K_SS = coalition_k(self.X, self.X)

        Xi_S = (K_SS.add_diag(self.n * self.cme_reg).inv_matmul(K_Sc.evaluate())).T

        return self.krr_weights.T @ (K_SSp * (Xi_S @ K_SSp)) - self.reference

    def fit(self, X_new, method, sample_method, num_samples=100, wls_reg=1e-10):

        n_ = X_new.shape[0]

        if sample_method == "MC":
            Z = large_scale_sample_alternative(self.m, num_samples)
        elif sample_method == "MC2":
            Z = generate_full_Z(self.m)
            Z = subsetting_full_Z(Z, samples=num_samples)
        else:
            Z = generate_full_Z(self.m)

        # Set up containers
        epoch = Z.shape[0]
        Y_target = np.zeros((epoch, n_))

        count = 0
        weights = []

        for row in tqdm(Z):
            if np.sum(row) == 0 or np.sum(row) == self.m:
                weights.append(1e5)

            else:
                z = row
                weights.append(
                    (self.m - 1) / (binom(self.m, sum(z)) * sum(z) * (self.m - sum(z)))
                )

            if method == "O":
                Y_target[count, :] = self._value_observation(row, X_new)
            elif method == "I":
                Y_target[count, :] = self._value_intervention(row, X_new)
            else:
                raise ValueError("Must be either interventional or observational")

            count += 1

        clf = Ridge(wls_reg)
        clf.fit(Z, Y_target, sample_weight=weights)

        return clf.coef_
