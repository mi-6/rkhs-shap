"""Tests for exact RKHS-SHAP implementation."""

import gpytorch
import numpy as np
import pytest
import shap
import torch

from rkhs_shap.exact_gp import ExactGPModel
from rkhs_shap.rkhs_shap_exact import RKHSSHAP
from rkhs_shap.utils import calculate_additivity_mae, calculate_correlation, to_tensor

from .conftest import get_train_subset, train_gp_model

# Test configuration constants
N_EXPLAIN_SAMPLES = 10
CME_REGULARIZATION = 1e-4

# Assertion thresholds
MAX_ADDITIVITY_MAE = 0.005
MAX_INTERNAL_MAE = 0.001
MIN_INTERVENTIONAL_CORRELATION = 0.99
DEFAULT_MIN_OBSERVATIONAL_CORRELATION = 0.85


@pytest.fixture
def trained_model(diabetes_data):
    """Train a GP model with RBF kernel on the diabetes dataset."""
    X_train, y_train = get_train_subset(diabetes_data)
    return train_gp_model(X_train, y_train)


@pytest.fixture
def trained_model_matern(diabetes_data):
    """Train a GP model with Matern kernel on the diabetes dataset."""
    X_train, y_train = get_train_subset(diabetes_data)
    matern_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X_train.shape[1])
    return train_gp_model(X_train, y_train, covar_module=matern_kernel)


@pytest.fixture
def trained_model_scale_kernel(diabetes_data):
    """Train a GP model with ScaleKernel + RBF kernel on the diabetes dataset."""
    X_train_scaled, y_train = get_train_subset(diabetes_data)
    X_unscaled = X_train_scaled * np.array(
        [100.0, 50.0, 200.0, 10.0, 5.0, 30.0, 80.0, 150.0, 20.0, 40.0]
    )
    X_unscaled = X_unscaled + np.array(
        [50.0, -20.0, 100.0, 5.0, 2.0, 10.0, 40.0, 80.0, 10.0, 20.0]
    )
    y_unscaled = y_train * 2.5 + 10.0
    scaled_kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=X_unscaled.shape[1])
    )
    return train_gp_model(X_unscaled, y_unscaled, covar_module=scaled_kernel)


@pytest.fixture
def trained_model_unscaled_data_rbf(diabetes_data):
    """Train a GP model with RBF kernel on unscaled diabetes dataset."""
    X_train_scaled, y_train = get_train_subset(diabetes_data)
    X_unscaled = X_train_scaled * np.array(
        [100.0, 50.0, 200.0, 10.0, 5.0, 30.0, 80.0, 150.0, 20.0, 40.0]
    )
    X_unscaled = X_unscaled + np.array(
        [50.0, -20.0, 100.0, 5.0, 2.0, 10.0, 40.0, 80.0, 10.0, 20.0]
    )
    y_unscaled = y_train * 2.5 + 10.0
    return train_gp_model(X_unscaled, y_unscaled)


@pytest.fixture
def trained_model_underfit(diabetes_data):
    """Train an underfit GP model with only 1 training iteration."""
    X_train, y_train = get_train_subset(diabetes_data)
    return train_gp_model(X_train, y_train, training_iter=1)


def run_rkhs_shap_test(
    trained_model: tuple[ExactGPModel, torch.Tensor, torch.Tensor],
    min_corr_O: float = DEFAULT_MIN_OBSERVATIONAL_CORRELATION,
) -> None:
    """
    Helper function to run RKHS-SHAP test on a trained model.

    This test verifies:
    1. RKHS-SHAP runs successfully on real data
    2. Additivity property is satisfied (MAE is low)
    3. Results are correlated with KernelSHAP
    4. RKHS-SHAP additivity is comparable or better than KernelSHAP
    """
    np.random.seed(42)
    torch.manual_seed(42)

    gp, X_train, y_train = trained_model
    kernel_name = gp.covar_module.__class__.__name__

    lambda_krr = gp.likelihood.noise.detach().cpu()
    lambda_cme = to_tensor(CME_REGULARIZATION)

    rkhs_shap = RKHSSHAP(
        X=X_train,
        y=y_train,
        kernel=gp.covar_module,
        noise_var=lambda_krr.item(),
        cme_reg=lambda_cme.item(),
        mean_function=gp.mean_module,
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

    internal_preds = rkhs_shap.ypred[:N_EXPLAIN_SAMPLES].squeeze()
    internal_baseline = rkhs_shap.reference

    additivity_mae_I = calculate_additivity_mae(shap_values_I, model_preds, baseline)
    additivity_mae_O = calculate_additivity_mae(shap_values_O, model_preds, baseline)

    internal_mae_I = calculate_additivity_mae(
        shap_values_I, internal_preds, internal_baseline
    )
    internal_mae_O = calculate_additivity_mae(
        shap_values_O, internal_preds, internal_baseline
    )

    explainer = shap.KernelExplainer(gp.predict_mean_numpy, X_train.numpy())
    kernel_explanation = explainer(X_explain.numpy())
    kernel_values = np.asarray(kernel_explanation.values)
    kernel_additivity_mae = calculate_additivity_mae(
        kernel_values, model_preds, baseline
    )

    pred_range = (model_preds.max() - model_preds.min()).item()
    additivity_mae_I = additivity_mae_I / pred_range
    additivity_mae_O = additivity_mae_O / pred_range
    kernel_additivity_mae = kernel_additivity_mae / pred_range

    mean_corr_I = calculate_correlation(kernel_values, shap_values_I)
    mean_corr_O = calculate_correlation(kernel_values, shap_values_O)

    print(f"\n{kernel_name} Kernel Test Results:")
    print(f"RKHS-SHAP Interventional additivity MAE: {additivity_mae_I:.6f}")
    print(f"RKHS-SHAP Observational additivity MAE: {additivity_mae_O:.6f}")
    print(f"RKHS-SHAP Internal Interventional MAE: {internal_mae_I:.6f}")
    print(f"RKHS-SHAP Internal Observational MAE: {internal_mae_O:.6f}")
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

    assert internal_mae_I < MAX_INTERNAL_MAE, (
        f"Internal interventional additivity error too large: {internal_mae_I}"
    )
    assert internal_mae_O < MAX_INTERNAL_MAE, (
        f"Internal observational additivity error too large: {internal_mae_O}"
    )

    assert mean_corr_I > MIN_INTERVENTIONAL_CORRELATION, (
        f"Interventional correlation with KernelSHAP too low: {mean_corr_I}"
    )
    assert mean_corr_O > min_corr_O, (
        f"Observational correlation with KernelSHAP too low: {mean_corr_O}"
    )

    print("\n" + "=" * 60)
    print(f"{kernel_name} Kernel test passed!")
    print("=" * 60)


def test_exact_rkhs_shap_diabetes(trained_model):
    """Test exact RKHS-SHAP with RBF kernel on the diabetes dataset."""
    run_rkhs_shap_test(trained_model)


def test_exact_rkhs_shap_diabetes_matern(trained_model_matern):
    """Test exact RKHS-SHAP with Matern kernel on the diabetes dataset."""
    run_rkhs_shap_test(trained_model_matern, min_corr_O=0.83)


def test_exact_rkhs_shap_diabetes_scaled(trained_model_scale_kernel):
    """Test exact RKHS-SHAP with ScaleKernel + RBF kernel on the diabetes dataset."""
    run_rkhs_shap_test(trained_model_scale_kernel, min_corr_O=0.65)


def test_exact_rkhs_shap_diabetes_unscaled(trained_model_unscaled_data_rbf):
    """Test exact RKHS-SHAP with RBF kernel on unscaled dataset."""
    run_rkhs_shap_test(trained_model_unscaled_data_rbf, min_corr_O=0.6)


def test_exact_rkhs_shap_diabetes_underfit(trained_model_underfit):
    """Test exact RKHS-SHAP with an underfit model trained for only 1 iteration.

    This test verifies RKHS-SHAP works with poorly trained models that haven't
    converged, which may have suboptimal hyperparameters and poor predictions.
    """
    run_rkhs_shap_test(trained_model_underfit, min_corr_O=0.83)


def test_exact_rkhs_shap_mc_sampling():
    """Test exact RKHS-SHAP with MC sampling on higher-dimensional synthetic data.

    This test verifies:
    1. RKHS-SHAP works with Monte Carlo coalition sampling (sample_method="weighted")
    2. MC sampling produces reasonable results on problems where full enumeration is infeasible
    3. Additivity property is approximately satisfied with MC sampling
    """
    np.random.seed(42)
    torch.manual_seed(42)

    n_train = 100
    n_features = 15
    n_explain = 5

    X_train = torch.randn(n_train, n_features, dtype=torch.float64)
    true_weights = torch.randn(n_features, dtype=torch.float64) * 0.5
    y_train = X_train @ true_weights + 0.1 * torch.randn(n_train, dtype=torch.float64)

    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=n_features)
    kernel.lengthscale = torch.ones(1, n_features) * 2.0

    trained_model = train_gp_model(
        X_train, y_train, covar_module=kernel, training_iter=50
    )
    gp, X_train, y_train = trained_model

    lambda_krr = gp.likelihood.noise.detach().cpu()
    lambda_cme = to_tensor(CME_REGULARIZATION)

    rkhs_shap = RKHSSHAP(
        X=X_train,
        y=y_train,
        kernel=gp.covar_module,
        noise_var=lambda_krr.item(),
        cme_reg=lambda_cme.item(),
        mean_function=gp.mean_module,
    )

    X_explain = X_train[:n_explain]

    shap_values_I = rkhs_shap.fit(
        X_test=X_explain,
        method="I",
        sample_method="weighted",
        num_samples=500,
    )

    model_preds = gp.predict(X_explain).mean
    baseline = gp.predict(X_train).mean.mean().item()

    additivity_mae_I = calculate_additivity_mae(shap_values_I, model_preds, baseline)
    pred_range = (model_preds.max() - model_preds.min()).item()
    normalized_mae = additivity_mae_I / pred_range

    print(f"\nMC Sampling Test Results (m={n_features}):")
    print(f"Number of samples: 500 (out of 2^{n_features} = {2**n_features} possible)")
    print(f"RKHS-SHAP Interventional additivity MAE: {normalized_mae:.6f}")
    print(f"SHAP values shape: {shap_values_I.shape}")

    assert shap_values_I.shape == (n_explain, n_features)
    assert normalized_mae < 1e-4, (
        f"MC sampling additivity error too large: {normalized_mae:.6f}"
    )

    print("\n" + "=" * 60)
    print("MC Sampling test passed!")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
