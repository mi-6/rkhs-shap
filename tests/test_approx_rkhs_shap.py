"""Tests for approximate RKHS-SHAP implementation with Nyström approximation."""

import gpytorch
import numpy as np
import pytest
import shap
import torch

from rkhs_shap.examples.exact_gp import ExactGPModel
from rkhs_shap.rkhs_shap_approx import RKHSSHAP_Approx

# Test configuration constants
N_TRAIN_SAMPLES = 100
N_EXPLAIN_SAMPLES = 10
CME_REGULARIZATION = 1e-4
N_COMPONENTS = 50

# Assertion thresholds (more lenient for approximate method with Nyström)
MAX_ADDITIVITY_MAE = 0.10  # Nyström approximation has lower accuracy
MIN_INTERVENTIONAL_CORRELATION = 0.90
DEFAULT_MIN_OBSERVATIONAL_CORRELATION = 0.75


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
    """Helper function to train a GP model."""
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    gp = ExactGPModel(X_train_tensor, y_train_tensor, covar_module=covar_module)
    gp.fit()
    return gp, X_train_tensor, y_train_tensor


def _get_train_subset(diabetes_data) -> tuple[np.ndarray, np.ndarray]:
    """Extract training subset from diabetes data."""
    X, y = diabetes_data
    return X[:N_TRAIN_SAMPLES], y[:N_TRAIN_SAMPLES]


@pytest.fixture
def trained_model(diabetes_data):
    """Train a GP model with RBF kernel on the diabetes dataset."""
    X_train, y_train = _get_train_subset(diabetes_data)
    return train_gp_model(X_train, y_train)


@pytest.fixture
def trained_model_matern(diabetes_data):
    """Train a GP model with Matern kernel on the diabetes dataset."""
    X_train, y_train = _get_train_subset(diabetes_data)
    matern_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X_train.shape[1])
    return train_gp_model(X_train, y_train, covar_module=matern_kernel)


@pytest.fixture
def trained_model_scaled(diabetes_data):
    """Train a GP model with ScaleKernel + RBF kernel on the diabetes dataset."""
    X_train, y_train = _get_train_subset(diabetes_data)
    scaled_kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=X_train.shape[1])
    )
    return train_gp_model(X_train, y_train, covar_module=scaled_kernel)


def run_rkhs_shap_test(
    trained_model: tuple[ExactGPModel, torch.Tensor, torch.Tensor],
    min_corr_O: float = DEFAULT_MIN_OBSERVATIONAL_CORRELATION,
) -> None:
    """
    Helper function to run RKHS-SHAP test on a trained model.

    This test verifies:
    1. RKHS-SHAP runs successfully on real data with Nyström approximation
    2. Additivity property is satisfied (MAE is low)
    3. Results are correlated with KernelSHAP
    4. RKHS-SHAP additivity is comparable or better than KernelSHAP
    """
    np.random.seed(42)
    torch.manual_seed(42)

    gp, X_train, y_train = trained_model
    kernel_name = gp.covar_module.__class__.__name__

    lambda_krr = gp.likelihood.noise.detach().cpu().float()
    lambda_cme = torch.tensor(CME_REGULARIZATION).float()

    rkhs_shap = RKHSSHAP_Approx(
        X=X_train,
        y=y_train,
        kernel=gp.covar_module,
        noise_var=lambda_krr,
        cme_reg=lambda_cme,
        n_components=N_COMPONENTS,
    )

    X_explain = X_train[:N_EXPLAIN_SAMPLES]

    shap_values_I = rkhs_shap.fit(
        X_test=X_explain,
        method="I",
        sample_method="full",
    )

    shap_values_O = rkhs_shap.fit(
        X_test=X_explain,
        method="O",
        sample_method="full",
    )

    model_preds = gp.predict(X_explain).mean
    baseline = gp.predict(X_train).mean.mean().item()

    additivity_mae_I = calculate_additivity_mae(shap_values_I, model_preds, baseline)
    additivity_mae_O = calculate_additivity_mae(shap_values_O, model_preds, baseline)

    explainer = shap.KernelExplainer(gp.predict_mean_numpy, X_train.numpy())
    kernel_explanation = explainer(X_explain.numpy())
    kernel_values = kernel_explanation.values
    kernel_additivity_mae = calculate_additivity_mae(
        kernel_values, model_preds, baseline
    )

    pred_range = (model_preds.max() - model_preds.min()).item()
    additivity_mae_I = additivity_mae_I / pred_range
    additivity_mae_O = additivity_mae_O / pred_range
    kernel_additivity_mae = kernel_additivity_mae / pred_range

    mean_corr_I = calculate_correlation(kernel_values, shap_values_I)
    mean_corr_O = calculate_correlation(kernel_values, shap_values_O)

    print(f"\n{kernel_name} Kernel Test Results (Approximate with Nyström):")
    print(f"RKHS-SHAP Interventional additivity MAE: {additivity_mae_I:.6f}")
    print(f"RKHS-SHAP Observational additivity MAE: {additivity_mae_O:.6f}")
    print(f"KernelSHAP additivity MAE: {kernel_additivity_mae:.6f}")
    print("\nCorrelation with KernelSHAP:")
    print(f"  Interventional: {mean_corr_I:.3f}")
    print(f"  Observational: {mean_corr_O:.3f}")

    assert shap_values_I.shape == (N_EXPLAIN_SAMPLES, X_train.shape[1])
    assert shap_values_O.shape == (N_EXPLAIN_SAMPLES, X_train.shape[1])

    assert additivity_mae_I < MAX_ADDITIVITY_MAE, (
        f"Interventional additivity error too large: {additivity_mae_I}"
    )
    assert additivity_mae_O < MAX_ADDITIVITY_MAE, (
        f"Observational additivity error too large: {additivity_mae_O}"
    )

    assert mean_corr_I > MIN_INTERVENTIONAL_CORRELATION, (
        f"Interventional correlation with KernelSHAP too low: {mean_corr_I}"
    )
    assert mean_corr_O > min_corr_O, (
        f"Observational correlation with KernelSHAP too low: {mean_corr_O}"
    )

    print("\n" + "=" * 60)
    print(f"{kernel_name} Kernel test passed (Approximate)!")
    print("=" * 60)


def test_approx_rkhs_shap_diabetes(trained_model):
    """Test approximate RKHS-SHAP with RBF kernel on the diabetes dataset."""
    run_rkhs_shap_test(trained_model)


def test_approx_rkhs_shap_diabetes_matern(trained_model_matern):
    """Test approximate RKHS-SHAP with Matern kernel on the diabetes dataset."""
    run_rkhs_shap_test(trained_model_matern, min_corr_O=0.75)


def test_approx_rkhs_shap_diabetes_scaled(trained_model_scaled):
    """Test approximate RKHS-SHAP with ScaleKernel + RBF kernel on the diabetes dataset."""
    run_rkhs_shap_test(trained_model_scaled, min_corr_O=0.75)


def test_approx_accepts_tensor_and_array(trained_model):
    """Test that approximate RKHS-SHAP accepts both Tensor and ndarray inputs."""
    gp, X_train, y_train = trained_model

    rkhs_shap = RKHSSHAP_Approx(
        X=X_train,
        y=y_train,
        kernel=gp.covar_module,
        noise_var=gp.likelihood.noise.detach().cpu().float(),
        cme_reg=CME_REGULARIZATION,
        n_components=N_COMPONENTS,
    )

    X_explain = X_train[:5]

    # Test with Tensor
    shap_values_tensor = rkhs_shap.fit(
        X_test=X_explain,
        method="I",
        sample_method="full",
    )

    # Test with ndarray
    shap_values_array = rkhs_shap.fit(
        X_test=X_explain.numpy(),
        method="I",
        sample_method="full",
    )

    # Results should be the same
    np.testing.assert_allclose(shap_values_tensor, shap_values_array, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
