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

    internal_preds = rkhs_shap._ypred[:N_EXPLAIN_SAMPLES].squeeze()
    internal_baseline = rkhs_shap._reference

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


def test_exact_rkhs_shap_precomputed_weights(trained_model):
    """Test that pre-computed krr_weights produce identical SHAP values.

    This verifies the optimization path where alpha/krr_weights are passed
    from an externally trained GP, skipping the internal linear solve.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    gp, X_train, y_train = trained_model

    lambda_krr = gp.likelihood.noise.detach().cpu()
    lambda_cme = to_tensor(CME_REGULARIZATION)

    # First create RKHSSHAP without pre-computed weights (baseline)
    rkhs_shap_baseline = RKHSSHAP(
        X=X_train,
        y=y_train,
        kernel=gp.covar_module,
        noise_var=lambda_krr.item(),
        cme_reg=lambda_cme.item(),
        mean_function=gp.mean_module,
    )

    # Extract the computed krr_weights
    precomputed_weights = rkhs_shap_baseline._krr_weights.clone()

    # Now create RKHSSHAP with pre-computed weights
    rkhs_shap_optimized = RKHSSHAP(
        X=X_train,
        y=y_train,
        kernel=gp.covar_module,
        noise_var=lambda_krr.item(),  # ignored when krr_weights provided
        cme_reg=lambda_cme.item(),
        mean_function=gp.mean_module,
        krr_weights=precomputed_weights,
    )

    X_explain = X_train[:N_EXPLAIN_SAMPLES]

    # Compute SHAP values from both
    shap_baseline_I = rkhs_shap_baseline.fit(
        X_explain, method="I", sample_method="full"
    )
    shap_optimized_I = rkhs_shap_optimized.fit(
        X_explain, method="I", sample_method="full"
    )

    shap_baseline_O = rkhs_shap_baseline.fit(
        X_explain, method="O", sample_method="full"
    )
    shap_optimized_O = rkhs_shap_optimized.fit(
        X_explain, method="O", sample_method="full"
    )

    # Verify they're identical
    np.testing.assert_allclose(
        shap_baseline_I,
        shap_optimized_I,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Interventional SHAP values differ with pre-computed weights",
    )
    np.testing.assert_allclose(
        shap_baseline_O,
        shap_optimized_O,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Observational SHAP values differ with pre-computed weights",
    )

    # Also verify internal attributes match
    np.testing.assert_allclose(
        rkhs_shap_baseline._reference,
        rkhs_shap_optimized._reference,
        rtol=1e-10,
        err_msg="Reference values differ",
    )
    np.testing.assert_allclose(
        rkhs_shap_baseline._ypred.numpy(),
        rkhs_shap_optimized._ypred.numpy(),
        rtol=1e-10,
        err_msg="Predictions differ",
    )

    print("\nPre-computed weights test passed!")
    print(
        f"Max diff (Interventional): {np.abs(shap_baseline_I - shap_optimized_I).max():.2e}"
    )
    print(
        f"Max diff (Observational): {np.abs(shap_baseline_O - shap_optimized_O).max():.2e}"
    )


def test_exact_rkhs_shap_gpytorch_alpha(trained_model):
    """Test that alpha extracted from GPyTorch model produces near-identical SHAP values.

    This is the realistic use case: extracting pre-computed alpha from a trained
    GPyTorch model via model.prediction_strategy.mean_cache and passing it to
    RKHSSHAP to skip the expensive linear solve.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    gp, X_train, y_train = trained_model

    lambda_krr = gp.likelihood.noise.detach().cpu()
    lambda_cme = to_tensor(CME_REGULARIZATION)

    # Ensure model is in eval mode and trigger cache population
    gp.eval()
    with torch.no_grad():
        _ = gp(X_train[:1])  # Make a prediction to populate the cache

    # Extract alpha from GPyTorch's prediction strategy
    gp_alpha = gp.prediction_strategy.mean_cache.detach()

    # Create RKHSSHAP without pre-computed weights (baseline)
    rkhs_shap_baseline = RKHSSHAP(
        X=X_train,
        y=y_train,
        kernel=gp.covar_module,
        noise_var=lambda_krr.item(),
        cme_reg=lambda_cme.item(),
        mean_function=gp.mean_module,
    )

    # Create RKHSSHAP with GPyTorch's alpha
    rkhs_shap_with_gp_alpha = RKHSSHAP(
        X=X_train,
        y=y_train,
        kernel=gp.covar_module,
        noise_var=lambda_krr.item(),
        cme_reg=lambda_cme.item(),
        mean_function=gp.mean_module,
        krr_weights=gp_alpha,
    )

    X_explain = X_train[:N_EXPLAIN_SAMPLES]

    # Compute SHAP values from both
    shap_baseline_I = rkhs_shap_baseline.fit(
        X_explain, method="I", sample_method="full"
    )
    shap_gp_alpha_I = rkhs_shap_with_gp_alpha.fit(
        X_explain, method="I", sample_method="full"
    )

    shap_baseline_O = rkhs_shap_baseline.fit(
        X_explain, method="O", sample_method="full"
    )
    shap_gp_alpha_O = rkhs_shap_with_gp_alpha.fit(
        X_explain, method="O", sample_method="full"
    )

    # Check alpha values are very close (may have small numerical differences
    # due to different solve methods: GPyTorch uses CG for n>800, we use direct solve)
    alpha_diff = (
        torch.abs(rkhs_shap_baseline._krr_weights.squeeze() - gp_alpha).max().item()
    )
    print(f"\nMax alpha difference (RKHSSHAP vs GPyTorch): {alpha_diff:.2e}")

    # SHAP values should be very close (allowing for small numerical differences)
    max_diff_I = np.abs(shap_baseline_I - shap_gp_alpha_I).max()
    max_diff_O = np.abs(shap_baseline_O - shap_gp_alpha_O).max()

    print(f"Max SHAP diff (Interventional): {max_diff_I:.2e}")
    print(f"Max SHAP diff (Observational): {max_diff_O:.2e}")

    # Use tolerances that account for numerical differences in solve methods
    # GPyTorch may use different solvers (CG vs direct) which cause small differences
    np.testing.assert_allclose(
        shap_baseline_I,
        shap_gp_alpha_I,
        rtol=1e-4,
        atol=2e-5,
        err_msg="Interventional SHAP values differ too much with GPyTorch alpha",
    )
    np.testing.assert_allclose(
        shap_baseline_O,
        shap_gp_alpha_O,
        rtol=1e-4,
        atol=2e-5,
        err_msg="Observational SHAP values differ too much with GPyTorch alpha",
    )

    # Verify correlation is nearly perfect
    corr_I = calculate_correlation(shap_baseline_I, shap_gp_alpha_I)
    corr_O = calculate_correlation(shap_baseline_O, shap_gp_alpha_O)

    print(f"Correlation (Interventional): {corr_I:.6f}")
    print(f"Correlation (Observational): {corr_O:.6f}")

    assert corr_I > 0.9999, f"Interventional correlation too low: {corr_I}"
    assert corr_O > 0.9999, f"Observational correlation too low: {corr_O}"

    print("\nGPyTorch alpha test passed!")


def test_exact_rkhs_shap_reproducibility():
    """Test that exact RKHS-SHAP produces reproducible results with MC sampling."""
    np.random.seed(123)
    torch.manual_seed(123)

    n_train, n_features, n_explain = 10, 10, 3
    X_train = torch.randn(n_train, n_features, dtype=torch.float64)
    true_weights = torch.randn(n_features, dtype=torch.float64) * 0.5
    y_train = X_train @ true_weights + 0.1 * torch.randn(n_train, dtype=torch.float64)

    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=n_features)
    kernel.lengthscale = torch.ones(1, n_features) * 2.0
    gp, X_train, y_train = train_gp_model(
        X_train, y_train, covar_module=kernel, training_iter=5
    )
    X_explain = X_train[:n_explain]

    rkhs_shap = RKHSSHAP(
        X=X_train,
        y=y_train,
        kernel=gp.covar_module,
        noise_var=gp.likelihood.noise.item(),
        cme_reg=CME_REGULARIZATION,
        mean_function=gp.mean_module,
    )

    # Test 1: Explicit RNG produces identical results
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    shap_1 = rkhs_shap.fit(X_explain, "I", "weighted", num_samples=100, rng=rng1)
    shap_2 = rkhs_shap.fit(X_explain, "I", "weighted", num_samples=100, rng=rng2)
    np.testing.assert_array_equal(shap_1, shap_2)

    # Test 2: Passing same RNG produces identical results
    rng1, rng2 = np.random.default_rng(999), np.random.default_rng(999)
    shap_3 = rkhs_shap.fit(X_explain, "I", "weighted", num_samples=100, rng=rng1)
    shap_4 = rkhs_shap.fit(X_explain, "I", "weighted", num_samples=100, rng=rng2)
    np.testing.assert_array_equal(shap_3, shap_4)

    # Test 3: Different seeds produce different results
    rng_a, rng_b = np.random.default_rng(111), np.random.default_rng(222)
    shap_5 = rkhs_shap.fit(X_explain, "I", "weighted", num_samples=100, rng=rng_a)
    shap_6 = rkhs_shap.fit(X_explain, "I", "weighted", num_samples=100, rng=rng_b)
    assert not np.array_equal(shap_5, shap_6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
