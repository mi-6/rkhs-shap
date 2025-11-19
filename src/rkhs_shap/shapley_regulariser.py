import copy

import numpy as np
import torch
from gpytorch.kernels import RBFKernel
from linear_operator.operators import to_linear_operator
from tqdm import tqdm

from rkhs_shap.kernel_approx import Nystroem
from rkhs_shap.sampling import (
    generate_full_Z,
    large_scale_sample_alternative,
)
from rkhs_shap.subset_kernel import SubsetKernel
from rkhs_shap.utils import to_tensor


def insert_i(ls, feature_to_exclude):
    """
    A helping function for computing Shapley functional in ShapleyRegulariser
    """
    subset_vec = np.array([True for _ in range(len(ls) + 1)])
    subset_vec[feature_to_exclude] = False

    holder = np.zeros(len(ls) + 1)
    holder[subset_vec] = ls

    Sui = copy.deepcopy(holder)
    Sui[feature_to_exclude] = True

    S = copy.deepcopy(holder)
    S[feature_to_exclude] = False

    return [Sui == 1, S == 1]


class ShapleyRegulariser(object):
    def __init__(
        self, lambda_sv: float, lambda_krr: float, lambda_cme: float, n_components: int
    ):
        """[Initialisation]

        Args:
            lambda_sv (float): [regularisation parameter for Shapley Regulariser]
            lambda_krr (float): [regularisation parameter for kernel ridge regression]
            lambda_cme (float): [regularisation parameter for conditional mean embedding]
            n_components (int): [number of landmark points for kernel approximation]
        """

        self.lambda_cme = lambda_cme
        self.lambda_krr = lambda_krr
        self.lambda_sv = lambda_sv
        self.n_components = n_components

        self.Z = None
        self.nystroem = None
        self.kernel = None

    def _get_int_embedding(self, z, X):
        """
        Compute the interventional embedding
        """

        zc = ~z
        n, m = X.shape

        if z.sum() == m:
            return self.Z @ self.Z.T
        elif z.sum() == 0:
            return (self.Kx.mean(axis=1) * torch.ones((n, n))).T
        else:
            S = np.where(z)[0]
            Sc = np.where(zc)[0]

            S_kernel = SubsetKernel(self.kernel, subset_dims=S)
            Sc_kernel = SubsetKernel(self.kernel, subset_dims=Sc)

            Z_S = self._nystroem_transform_subset(X, S_kernel)
            K_SS = Z_S @ Z_S.T

            Z_Sc = self._nystroem_transform_subset(X, Sc_kernel)
            K_Sc = Z_Sc @ Z_Sc.T

            KME_mat = K_Sc.mean(axis=1)[:, np.newaxis] * torch.ones((n, n))

            return K_SS * KME_mat

    def _get_obsv_embedding(self, z, X):
        """
        Compute the observational embedding
        """

        zc = ~z
        n = X.shape[0]

        if z.sum() == 0:
            return self.Kx
        elif z.sum() == 0:
            return self.Kx.mean(axis=1) * torch.ones((n, n)).T
        else:
            S = np.where(z)[0]
            Sc = np.where(zc)[0]

            S_kernel = SubsetKernel(self.kernel, subset_dims=S)
            Sc_kernel = SubsetKernel(self.kernel, subset_dims=Sc)

            Z_S = self._nystroem_transform_subset(X, S_kernel)
            K_SS = Z_S @ Z_S.T

            Z_Sc = self._nystroem_transform_subset(X, Sc_kernel)
            cme_latter_part = (
                to_linear_operator(Z_S.T @ Z_S)
                .add_diagonal(to_tensor(self.lambda_cme))
                .solve(Z_S.T)
            )
            holder = K_SS * (Z_Sc @ Z_Sc.T @ Z_S @ cme_latter_part)

            return holder

    def _nystroem_transform_subset(self, X, subset_kernel: SubsetKernel):
        """Apply Nyström transformation with a subset kernel."""
        X_tensor = to_tensor(X)

        ZT = (
            subset_kernel(self.nystroem.landmarks)
            .add_jitter()
            .cholesky()
            .solve(subset_kernel(self.nystroem.landmarks, X_tensor).to_dense())
        )
        return ZT.T

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        ls: np.ndarray | torch.Tensor,
        features_index: list,
        method: str = "O",
        num_samples: int = 300,
        sample_method: str = "MC",
    ):
        """[summary]

        Args:
            X (float): [training data]
            y (float): [training labels]
            ls (float): [lengthscales for the kernel]
            features_index (list): [list containing which feature to minimise]
            method (str, optional): ["O" stands for OSV-Reg and "I" stands for ISV-Reg]. Defaults to "O".
            num_samples (int, optional): [number of samples to estimate the shapley functionals]. Defaults to 300.
            sample_method (str, optional): [sampling method to compute shapley functionals]. Defaults to "MC".
        """

        self.X = X
        self.n, self.m = X.shape
        self.ls = ls

        rbf = RBFKernel()
        rbf.lengthscale = to_tensor(ls)
        rbf.raw_lengthscale.requires_grad = False
        self.kernel = rbf

        ny = Nystroem(kernel=rbf, n_components=self.n_components)
        ny.fit(to_tensor(self.X))
        Phi = ny.transform(to_tensor(self.X))
        self.Z = Phi
        Kx = Phi @ Phi.T

        self.nystroem = ny
        self.Kx = Kx

        m_exclude_i = self.m - len(features_index)
        if sample_method == "MC":
            Z_exclude_i = large_scale_sample_alternative(m_exclude_i, num_samples)
        else:
            Z_exclude_i = generate_full_Z(m_exclude_i)
        A = np.zeros((self.n, self.n))

        for row in tqdm(Z_exclude_i):
            Sui, S = insert_i(row, feature_to_exclude=features_index)

            if method == "O":
                sui = self._get_obsv_embedding(Sui, X)
                s = self._get_obsv_embedding(S, X)

                A += (sui - s).numpy()

            elif method == "I":
                sui = self._get_int_embedding(Sui, X)
                s = self._get_int_embedding(S, X)

                A += (sui - s).numpy()

            else:
                raise ValueError(
                    "Method must either be Observational or Interventional"
                )

        A = A / Z_exclude_i.shape[0]

        self.A = A

        # Formulate the regression
        y_ten = to_tensor(y).reshape(-1, 1)
        A = to_linear_operator(to_tensor(A))
        self.AAt = A @ A.t()
        K = rbf(to_tensor(X / ls))
        alphas = (
            (K @ K + self.lambda_krr * K + self.lambda_sv * self.AAt)
            .add_jitter()
            .solve(K.matmul(y_ten))
        )

        self.alphas = alphas
        ypred = K.matmul(alphas)

        print("RMSE: %.2f" % torch.sqrt(torch.mean((y_ten - ypred) ** 2)))

        self.RMSE = torch.sqrt(torch.mean((y_ten - ypred) ** 2)).numpy()
        self.ypred = ypred

    def predict(self, X_test):
        # obtain kernel K_{test, train} first
        Z_test = self.nystroem.transform(to_tensor(X_test))
        K_xpx = Z_test @ self.Z.T

        return K_xpx @ self.alphas
