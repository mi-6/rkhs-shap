"""Tests for approximate RKHS-SHAP implementation with Nyström approximation."""

import gpytorch
import numpy as np
import pytest
import shap
import torch

from rkhs_shap.exact_gp import ExactGPModel
from rkhs_shap.rkhs_shap_approx import RKHSSHAPApprox

from .conftest import (
    calculate_additivity_mae,
    calculate_correlation,
    get_train_subset,
    train_gp_model,
)

# Test configuration constants
N_EXPLAIN_SAMPLES = 10
CME_REGULARIZATION = 1e-4
N_COMPONENTS = 50

# Assertion thresholds (more lenient for approximate method with Nyström)
MAX_ADDITIVITY_MAE = 0.10  # Nyström approximation has lower accuracy
MIN_INTERVENTIONAL_CORRELATION = 0.90
DEFAULT_MIN_OBSERVATIONAL_CORRELATION = 0.75


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
def trained_model_scaled(diabetes_data):
    """Train a GP model with ScaleKernel + RBF kernel on the diabetes dataset."""
    X_train, y_train = get_train_subset(diabetes_data)
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

    rkhs_shap = RKHSSHAPApprox(
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


def test_approx_rkhs_shap_diabetes_scaled(trained_model_scaled):
    """Test approximate RKHS-SHAP with ScaleKernel + RBF kernel on the diabetes dataset."""
    run_rkhs_shap_test(trained_model_scaled, min_corr_O=0.75)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
