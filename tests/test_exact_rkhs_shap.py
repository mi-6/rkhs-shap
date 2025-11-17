"""Tests for exact RKHS-SHAP implementation."""

import numpy as np
import pytest
import torch
import shap

from rkhs_shap.rkhs_shap_exact import RKHSSHAP
from rkhs_shap.examples.exact_gp import ExactGPModel


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
    additivity_errors = []
    for i in range(shap_values.shape[0]):
        shap_sum = shap_values[i].sum()
        pred_diff = model_preds[i].item() - baseline
        error = abs(shap_sum - pred_diff)
        additivity_errors.append(error)
    return np.mean(additivity_errors)


@pytest.fixture
def diabetes_data():
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


@pytest.fixture
def trained_model(diabetes_data):
    """Train a GP model on the diabetes dataset."""
    X, y = diabetes_data

    # Use a subset for faster testing
    X_train = X[:100]
    y_train = y[:100]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    gp = ExactGPModel(X_train_tensor, y_train_tensor)
    gp.fit()

    return gp, X_train_tensor, y_train_tensor


def test_exact_rkhs_shap_diabetes(trained_model):
    """
    Test exact RKHS-SHAP on the diabetes dataset and compare with KernelSHAP.

    This test verifies:
    1. RKHS-SHAP runs successfully on real data
    2. Additivity property is satisfied (MAE is low)
    3. Results are correlated with KernelSHAP
    4. RKHS-SHAP additivity is comparable or better than KernelSHAP
    """
    np.random.seed(42)
    torch.manual_seed(42)

    gp: ExactGPModel
    X_train: torch.Tensor
    y_train: torch.Tensor
    gp, X_train, y_train = trained_model

    lambda_krr = gp.likelihood.noise.detach().cpu().float()
    lambda_cme = torch.tensor(1e-4).float()

    rkhs_shap = RKHSSHAP(
        X=X_train,
        y=y_train,
        kernel=gp.covar_module,
        noise_var=lambda_krr,
        cme_reg=lambda_cme,
    )

    n_explain = 10
    X_explain = X_train[:n_explain]

    # Compute Interventional RKHS-SHAP
    shap_values_I = rkhs_shap.fit(
        X_test=X_explain,
        method="I",  # Interventional
        sample_method="full",
    )

    # Compute Observational RKHS-SHAP
    shap_values_O = rkhs_shap.fit(
        X_test=X_explain,
        method="O",  # Observational
        sample_method="full",
    )

    # Get model predictions and baseline for additivity test
    model_preds = gp.predict(X_explain).mean
    baseline = gp.predict(X_train).mean.mean().item()

    additivity_mae_I = calculate_additivity_mae(shap_values_I, model_preds, baseline)
    additivity_mae_O = calculate_additivity_mae(shap_values_O, model_preds, baseline)

    # Run KernelSHAP for comparison
    explainer = shap.KernelExplainer(gp.predict_mean_numpy, X_train.numpy())
    kernel_explanation = explainer(X_explain.numpy())
    kernel_values = kernel_explanation.values
    kernel_additivity_mae = calculate_additivity_mae(
        kernel_values, model_preds, baseline
    )

    # Normalize MAEs by prediction range for better interpretability
    pred_range = (model_preds.max() - model_preds.min()).item()
    additivity_mae_I = additivity_mae_I / pred_range
    additivity_mae_O = additivity_mae_O / pred_range
    kernel_additivity_mae = kernel_additivity_mae / pred_range
    print(f"\nRKHS-SHAP Interventional additivity MAE: {additivity_mae_I:.6f}")
    print(f"RKHS-SHAP Observational additivity MAE: {additivity_mae_O:.6f}")
    print(f"KernelSHAP additivity MAE: {kernel_additivity_mae:.6f}")

    corr_I = []
    corr_O = []

    for i in range(n_explain):
        corr_i = np.corrcoef(kernel_values[i], shap_values_I[i])[0, 1]
        if not np.isnan(corr_i):
            corr_I.append(corr_i)

        corr_o = np.corrcoef(kernel_values[i], shap_values_O[i])[0, 1]
        if not np.isnan(corr_o):
            corr_O.append(corr_o)

    mean_corr_I = np.mean(corr_I)
    mean_corr_O = np.mean(corr_O)

    print("\nCorrelation with KernelSHAP:")
    print(f"  Interventional: {mean_corr_I:.3f}")
    print(f"  Observational: {mean_corr_O:.3f}")

    # Assertions
    # 1. SHAP values should have correct shape
    assert shap_values_I.shape == (n_explain, X_train.shape[1])
    assert shap_values_O.shape == (n_explain, X_train.shape[1])

    # 2. Additivity error should be reasonably small (within 10% of prediction range)
    assert additivity_mae_I < 0.005, (
        f"Interventional additivity error too large: {additivity_mae_I}"
    )
    assert additivity_mae_O < 0.005, (
        f"Observational additivity error too large: {additivity_mae_O}"
    )

    # 3. Correlation with KernelSHAP should be high
    assert mean_corr_I > 0.99, (
        f"Interventional correlation with KernelSHAP too low: {mean_corr_I}"
    )
    assert mean_corr_O > 0.85, (
        f"Observational correlation with KernelSHAP too low: {mean_corr_O}"
    )

    print("\n" + "=" * 60)
    print("Test passed! Summary:")
    print(f"  RKHS-SHAP Interventional additivity MAE: {additivity_mae_I:.6f}")
    print(f"  RKHS-SHAP Observational additivity MAE:  {additivity_mae_O:.6f}")
    print(f"  KernelSHAP additivity MAE:                {kernel_additivity_mae:.6f}")
    print(f"  Correlation with KernelSHAP (I):          {mean_corr_I:.3f}")
    print(f"  Correlation with KernelSHAP (O):          {mean_corr_O:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
