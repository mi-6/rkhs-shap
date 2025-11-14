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
        self.y_pred = K_train @ krr_weights
        self.rmse = torch.sqrt(torch.mean(self.y_pred - self.y) ** 2)

    def _value_intervention(self, z, X_new):

        n_ = X_new.shape[0]
        zc = z == False

        reference = (self.y_pred.mean() * torch.ones((1, n_))).float()
        self.reference = reference

        if z.sum() == self.m:
            new_ypred = self.krr_weights.T @ self.kernel(self.X, X_new).evaluate()
            return new_ypred - reference

        elif z.sum() == 0:
            return 0

        else:
            z_tensor = torch.from_numpy(z) if isinstance(z, np.ndarray) else z
            zc_tensor = torch.from_numpy(zc) if isinstance(zc, np.ndarray) else zc

            active_S = torch.where(z_tensor)[0].tolist()
            active_Sc = torch.where(zc_tensor)[0].tolist()

            k_S = SubsetKernel(self.kernel, subset_dims=active_S)
            k_Sc = SubsetKernel(self.kernel, subset_dims=active_Sc)

            K_SSp = k_S(self.X, X_new).evaluate().float()
            K_Sc = k_Sc(self.X, self.X)

            KME_mat = K_Sc.evaluate().mean(axis=1)[:, np.newaxis] * torch.ones(
                (self.n, n_)
            )

            return self.krr_weights.T @ (K_SSp * KME_mat) - reference

    def _value_observation(self, z, X_new):

        n_ = X_new.shape[0]
        zc = z == False

        reference = (self.y_pred.mean() * torch.ones((1, n_))).float()
        self.reference = reference

        if z.sum() == self.m:
            new_ypred = self.krr_weights.T @ self.kernel(self.X, X_new).evaluate()

            return new_ypred - reference

        elif z.sum() == 0:
            return 0

        else:
            z_tensor = torch.from_numpy(z) if isinstance(z, np.ndarray) else z
            zc_tensor = torch.from_numpy(zc) if isinstance(zc, np.ndarray) else zc

            active_S = torch.where(z_tensor)[0].tolist()
            active_Sc = torch.where(zc_tensor)[0].tolist()

            k_S = SubsetKernel(self.kernel, subset_dims=active_S)
            k_Sc = SubsetKernel(self.kernel, subset_dims=active_Sc)

            K_SSp = k_S(self.X, X_new).evaluate().float()
            K_Sc = k_Sc(self.X, self.X)
            K_SS = k_S(self.X, self.X)

            Xi_S = (
                K_SS.add_diag(self.n * self.cme_reg).inv_matmul(K_Sc.evaluate())
            ).T

            return self.krr_weights.T @ (K_SSp * (Xi_S @ K_SSp)) - reference

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
