"""Shared test utilities and fixtures for RKHS-SHAP tests."""

import gpytorch
import numpy as np
import pytest
import shap
import torch

from rkhs_shap.exact_gp import ExactGPModel
from rkhs_shap.utils import to_tensor

N_TRAIN_SAMPLES = 100


@pytest.fixture
def diabetes_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the diabetes dataset from SHAP."""
    X, y = shap.datasets.diabetes()
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Normalize X to [0, 1] range
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X = (X - X_min) / (X_max - X_min + 1e-8)

    # Standardize y (zero mean, unit variance)
    y_mean, y_std = y.mean(), y.std()
    y = (y - y_mean) / y_std

    return X, y


def train_gp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    covar_module: gpytorch.kernels.Kernel | None = None,
    training_iter: int = 50,
) -> tuple[ExactGPModel, torch.Tensor, torch.Tensor]:
    X_train_tensor = to_tensor(X_train)
    y_train_tensor = to_tensor(y_train)
    gp = ExactGPModel(X_train_tensor, y_train_tensor, covar_module=covar_module)
    gp.fit(training_iter=training_iter)
    return gp, X_train_tensor, y_train_tensor


def get_train_subset(diabetes_data) -> tuple[np.ndarray, np.ndarray]:
    X, y = diabetes_data
    return X[:N_TRAIN_SAMPLES], y[:N_TRAIN_SAMPLES]
