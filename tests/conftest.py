"""Shared test utilities and fixtures for RKHS-SHAP tests."""

import gpytorch
import numpy as np
import pytest
import shap
import torch

from rkhs_shap.exact_gp import ExactGPModel

N_TRAIN_SAMPLES = 100


def calculate_additivity_mae(
    shap_values: np.ndarray, model_preds: torch.Tensor, baseline: float
) -> float:
    """
    Calculate the mean absolute error of the additivity property.

    The additivity property states that sum of SHAP values plus baseline
    should equal the model prediction: baseline + sum(shap_values) = f(x)

    Args:
        shap_values: Array of SHAP values (n_samples, n_features)
        model_preds: Model predictions for each sample
        baseline: Baseline prediction (usually mean of training predictions)

    Returns:
        float: Mean absolute additivity error
    """
    shap_sums = shap_values.sum(axis=1)
    pred_diffs = model_preds.numpy() - baseline
    return np.mean(np.abs(shap_sums - pred_diffs))


def calculate_correlation(values1: np.ndarray, values2: np.ndarray) -> float:
    """
    Calculate mean correlation between two sets of SHAP values.

    Args:
        values1: First set of SHAP values (n_samples, n_features)
        values2: Second set of SHAP values (n_samples, n_features)

    Returns:
        float: Mean correlation across all samples
    """
    correlations = []
    for i in range(values1.shape[0]):
        corr = np.corrcoef(values1[i], values2[i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    return np.mean(correlations)


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
) -> tuple[ExactGPModel, torch.Tensor, torch.Tensor]:
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    gp = ExactGPModel(X_train_tensor, y_train_tensor, covar_module=covar_module)
    gp.fit()
    return gp, X_train_tensor, y_train_tensor


def get_train_subset(diabetes_data) -> tuple[np.ndarray, np.ndarray]:
    X, y = diabetes_data
    return X[:N_TRAIN_SAMPLES], y[:N_TRAIN_SAMPLES]
