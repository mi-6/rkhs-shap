#############
#
#############

import math

import numpy as np
from sklearn.linear_model import Ridge
from tqdm import tqdm

from rkhs_shap.sampling import sample_coalitions_full, sample_coalitions_weighted


class CorrelatedLinearModel(object):
    def __init__(
        self,
        n: np.int,
        noise: np.float = 1e-1,
        mean_vec: np.array = None,
        Sigma: np.array = None,
        beta: np.array = None,
        random_state: np.int = None,
    ):
        """[Correlated Linear Model]
        This is a correlated linear model class where one could simulate observations where
        inputs are correlated and compute its Observational and Interventional Shapley Values as well.

        Args:
            n (np.int): [number of data points]
            Sigma (np.array): [user specified Covariance matrix of shape n x n]
            beta (np.array): [user specified linear model coefficients]
        """

        self.n = n
        self.m = Sigma.shape[0]
        if mean_vec is None:
            mean_vec = np.zeros(self.m)
        self.Sigma = Sigma
        self.beta = beta
        self.seed = random_state

        # Simulate the data
        np.random.seed(random_state)

        # Mean 0 multivariate
        X = np.random.multivariate_normal(mean=mean_vec, cov=Sigma, size=n)
        y = X @ beta + noise * np.random.normal(size=(n))

        self.X, self.y = X, y

    def _observational_value_function(self, z: np.ndarray, X: np.array):
        """[Compute Observational Value function evaluation]

        Args:
            z (np.ndarray): [subsetting]
            X (np.array): [the data]
        """

        # Subset
        zc = ~z

        # Mean 0 multivariate Gaussian
        reference = self.y.mean()

        if z.sum() == self.m:
            # E[f(X_S)|X_S=x] basically means everything is conditioned
            return X @ self.beta - reference
        elif z.sum() == 0:
            return 0
        else:
            X_S = X[:, z]
            beta_S, beta_Sc = self.beta[z], self.beta[zc]
            Sigma_S = self.Sigma[z][:, z]
            Sigma_SSc = self.Sigma[z][:, zc]

            return (
                X_S @ beta_S
                + X_S @ np.linalg.inv(Sigma_S) @ Sigma_SSc @ beta_Sc
                - reference
            )

    def kernelSHAP(
        self,
        X: np.array,
        sample_method: str = "full",
        num_samples: int = 1000,
        verbose: bool = True,
    ):
        """[Run the KernelSHAP algorithm to retrieve shapley values]

        Args:
            X (np.array): [data]
            sample_method (str, optional): [description]. Defaults to "weighted".
            num_samples (int, optional): [description]. Defaults to 1000.
            verbose (bool, optional): [description]. Defaults to True.
        """

        n = X.shape[0]

        if sample_method == "weighted":
            Z, _ = sample_coalitions_weighted(self.m, num_samples)
        elif sample_method == "full":
            Z, _ = sample_coalitions_full(self.m)

        epoch = Z.shape[0]
        Y_target = np.zeros((epoch, n))

        count, weights = 0, []

        if verbose:
            for row in tqdm(Z):
                if np.sum(row) == 0 or np.sum(row) == self.m:
                    weights.append(1e-5)
                else:
                    z = row
                    weights.append(
                        1
                        / (
                            (self.m - 1)
                            / (math.comb(self.m, sum(z)) * sum(z) * (self.m - sum(z)))
                        )
                    )

                Y_target[count, :] = self._observational_value_function(row, X)

                count += 1

        else:
            for row in Z:
                if np.sum(row) == 0 or np.sum(row) == self.m:
                    weights.append(1e-5)
                else:
                    z = row
                    weights.append(
                        1
                        / (
                            (self.m - 1)
                            / (math.comb(self.m, sum(z)) * sum(z) * (self.m - sum(z)))
                        )
                    )

                Y_target[count, :] = self._observational_value_function(row, X)

                count += 1

        clf = Ridge(1e-5)
        clf.fit(Z, Y_target, sample_weight=weights)

        self.full_shapley_values_ = np.concatenate(
            [clf.intercept_.reshape(-1, 1), clf.coef_], axis=1
        )
        self.SHAP_LM = clf

        return clf.coef_
