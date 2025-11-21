from abc import ABC, abstractmethod

import numpy as np
from scipy.special import binom
from sklearn.linear_model import Ridge
from torch import Tensor
from tqdm import tqdm

from rkhs_shap.sampling import (
    generate_full_Z,
    large_scale_sample_alternative,
    subset_full_Z,
)


class RKHSSHAPBase(ABC):
    """Base class with shared fit method for RKHS-SHAP implementations"""

    m: int
    reference: float

    @abstractmethod
    def _value_observation(self, z: np.ndarray, X_test: Tensor) -> Tensor:
        """Compute observational Shapley value function for coalition z.

        Args:
            z: Binary coalition vector of shape (m,) indicating active features
            X_test: Test points of shape (n_test, m)

        Returns:
            Value function evaluated at X_test, shape (1, n_test)
        """
        ...

    @abstractmethod
    def _value_intervention(self, z: np.ndarray, X_test: Tensor) -> Tensor:
        """Compute interventional Shapley value function for coalition z.

        Args:
            z: Binary coalition vector of shape (m,) indicating active features
            X_test: Test points of shape (n_test, m)

        Returns:
            Value function evaluated at X_test, shape (1, n_test)
        """
        ...

    def fit(
        self,
        X_test: Tensor,
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
        m = self.m
        if sample_method == "MC":
            Z = large_scale_sample_alternative(m, num_samples)
        elif sample_method == "MC2":
            Z = generate_full_Z(m)
            Z = subset_full_Z(Z, samples=num_samples)
        else:
            Z = generate_full_Z(m)

        n_coalitions = Z.shape[0]
        n_test = X_test.shape[0]
        Y_target = np.zeros((n_coalitions, n_test))
        count = 0
        weights = []

        for row in tqdm(Z):
            if np.sum(row) == 0 or np.sum(row) == m:
                weights.append(1e5)
            else:
                weights.append(
                    (m - 1) / (binom(m, np.sum(row)) * np.sum(row) * (m - np.sum(row)))
                )

            if method == "O":
                Y_target[count, :] = self._value_observation(row, X_test)
            elif method == "I":
                Y_target[count, :] = self._value_intervention(row, X_test)
            else:
                raise ValueError("Must be either interventional or observational")

            count += 1

        # clf = Ridge(wls_reg, solver="sag")
        clf = Ridge(wls_reg)
        clf.fit(Z, Y_target, sample_weight=weights)

        shap_values = clf.coef_

        return shap_values
