###############################
# KernelSHAP4K For Regression #
###############################

import torch
from scipy.special import binom
import numpy as np
from gpytorch.kernels import RBFKernel

# from gpytorch.lazy import lazify
from sklearn.linear_model import Ridge
from numpy import sum
from tqdm import tqdm
from rkhs_shap.sampling import (
    large_scale_sample_alternative,
    generate_full_Z,
    subsetting_full_Z,
)


class RKHSSHAP(object):
    """Implement the exact RKHS SHAP algorithm with no kernel approximation"""

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lambda_krr: float = 1e-2,
        lambda_cme: float = 1e-4,
        lengthscale: torch.Tensor | None = None,
    ) -> None:
        """Initialize exact RKHS-SHAP with Kernel Ridge Regression.

        Args:
            X: Training features of shape (n, m)
            y: Training targets of shape (n,) or (n, 1)
            lambda_krr: KRR regularization parameter. Corresponds to the noise variance
                in Gaussian Process formulation. If you've fit a GP model, set this to
                model.likelihood.noise_covar.noise for equivalent predictions.
            lambda_cme: Regularization for conditional/marginal mean embeddings
            lengthscale: Optional lengthscale parameter for the RBF kernel. If provided,
                the kernel's lengthscale will be set to this value.
        """

        # Store data
        self.n, self.m = X.shape
        self.X, self.y = X.float(), y

        lambda_krr = torch.tensor(lambda_krr, dtype=torch.float32)
        lambda_cme = torch.tensor(lambda_cme, dtype=torch.float32)
        self.lambda_krr = lambda_krr
        self.lambda_cme = lambda_cme

        # Set up kernel
        if lengthscale is not None and lengthscale.numel() > 1:
            rbf = RBFKernel(ard_num_dims=self.m)
            rbf.lengthscale = lengthscale
        else:
            rbf = RBFKernel()
            if lengthscale is not None:
                rbf.lengthscale = lengthscale
        rbf.raw_lengthscale.requires_grad = False
        self.k = rbf

        # Run Kernel Ridge Regression (need alphas!)
        Kxx = rbf(self.X)
        alphas = Kxx.add_diag(lambda_krr).inv_matmul(self.y)
        self.alphas = alphas.float().reshape(-1, 1)

        self.ypred = Kxx @ alphas
        self.rmse = torch.sqrt(torch.mean(self.ypred - self.y) ** 2)

    def _create_subkernel(self, active_dims: list[int]) -> RBFKernel:
        """Create a sub-kernel with specified active dimensions.

        Args:
            active_dims: List of dimension indices to include in the sub-kernel

        Returns:
            RBFKernel configured for the active dimensions with appropriate lengthscales
        """
        if hasattr(self.k, 'ard_num_dims') and self.k.ard_num_dims is not None:
            k_sub = RBFKernel(ard_num_dims=len(active_dims), active_dims=active_dims)
            k_sub.lengthscale = self.k.lengthscale[:, active_dims]
        else:
            k_sub = RBFKernel(active_dims=active_dims)
            k_sub.lengthscale = self.k.lengthscale

        k_sub.raw_lengthscale.requires_grad = False
        return k_sub

    def _value_intervention(self, z, X_new):

        n_ = X_new.shape[0]
        zc = z == False

        reference = (self.ypred.mean() * torch.ones((1, n_))).float()
        self.reference = reference

        if z.sum() == self.m:
            new_ypred = self.alphas.T @ self.k(self.X, X_new).evaluate()
            return new_ypred - reference

        elif z.sum() == 0:
            return 0

        else:
            z_tensor = torch.from_numpy(z) if isinstance(z, np.ndarray) else z
            zc_tensor = torch.from_numpy(zc) if isinstance(zc, np.ndarray) else zc

            active_S = torch.where(z_tensor)[0].tolist()
            active_Sc = torch.where(zc_tensor)[0].tolist()

            k_S = self._create_subkernel(active_S)
            k_Sc = self._create_subkernel(active_Sc)

            K_SSp = k_S(self.X, X_new).evaluate().float()
            K_Sc = k_Sc(self.X, self.X)

            KME_mat = K_Sc.evaluate().mean(axis=1)[:, np.newaxis] * torch.ones(
                (self.n, n_)
            )

            return self.alphas.T @ (K_SSp * KME_mat) - reference

    def _value_observation(self, z, X_new):

        n_ = X_new.shape[0]
        zc = z == False

        reference = (self.ypred.mean() * torch.ones((1, n_))).float()
        self.reference = reference

        if z.sum() == self.m:
            new_ypred = self.alphas.T @ self.k(self.X, X_new).evaluate()

            return new_ypred - reference

        elif z.sum() == 0:
            return 0

        else:
            z_tensor = torch.from_numpy(z) if isinstance(z, np.ndarray) else z
            zc_tensor = torch.from_numpy(zc) if isinstance(zc, np.ndarray) else zc

            active_S = torch.where(z_tensor)[0].tolist()
            active_Sc = torch.where(zc_tensor)[0].tolist()

            k_S = self._create_subkernel(active_S)
            k_Sc = self._create_subkernel(active_Sc)

            K_SSp = k_S(self.X, X_new).evaluate().float()
            K_Sc = k_Sc(self.X, self.X)
            K_SS = k_S(self.X, self.X)

            Xi_S = (
                K_SS.add_diag(self.n * self.lambda_cme).inv_matmul(K_Sc.evaluate())
            ).T

            return self.alphas.T @ (K_SSp * (Xi_S @ K_SSp)) - reference

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
