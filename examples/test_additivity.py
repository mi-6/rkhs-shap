"""Minimal script to reproduce the additivity error bug at n>800."""

import numpy as np
import shap
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rkhs_shap.exact_gp import ExactGPModel
from rkhs_shap.rkhs_shap_exact import RKHSSHAP
from rkhs_shap.utils import to_tensor


def load_scaled_dataset(n_train: int, n_val: int = 500):
    """Load and scale California housing dataset - same as notebook."""
    X, y = shap.datasets.california()
    rng = np.random.default_rng(42)

    n_indices = n_train + n_val
    both_indices = rng.choice(len(X), size=n_indices, replace=False)
    train_indices = both_indices[:n_train]

    X_train = X.iloc[train_indices].values
    y_train = y[train_indices]

    feature_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(X_train)

    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    return X_train, y_train


def test_sample_size(n_train: int, explain_size: int = 100):
    """Test RKHS-SHAP at a specific sample size."""
    print(f"\n{'=' * 60}")
    print(f"Testing sample size: {n_train}")
    print(f"{'=' * 60}")

    # Load data
    X_train, y_train = load_scaled_dataset(n_train)
    train_x = to_tensor(X_train)
    train_y = to_tensor(y_train)

    # Train GP model
    print("Training GP model...")
    model = ExactGPModel(train_x, train_y)
    model.fit(training_iter=50, lr=0.1)

    # Setup RKHS-SHAP
    noise_var = model.likelihood.noise.detach().cpu().float()
    rkhs_shap = RKHSSHAP(
        X=train_x,
        y=train_y,
        kernel=model.covar_module,
        noise_var=noise_var,
        cme_reg=1e-4,
    )

    # Explain first 100 points
    X_explain = X_train[:explain_size]
    X_explain_tensor = to_tensor(X_explain)

    # Compute SHAP values
    print("Computing RKHS-SHAP...")
    shap_values = rkhs_shap.fit(
        X_test=X_explain_tensor,
        method="I",  # Interventional
        sample_method="full",
    )

    # Check additivity using GP model predictions (like notebook)
    pred_explain = model.predict_mean_numpy(X_explain)
    baseline = model.predict_mean_numpy(X_train).mean()

    shap_sums = shap_values.sum(axis=1)
    pred_diffs = pred_explain - baseline
    additivity_errors = np.abs(shap_sums - pred_diffs)

    # Print results
    print("\nResults:")
    print(f"  GP baseline: {baseline:.6f}")
    print(f"  RKHS reference: {rkhs_shap.reference:.6f}")
    print(f"  Baseline difference: {abs(baseline - rkhs_shap.reference):.6f}")
    print(f"  Additivity MAE: {np.mean(additivity_errors):.6f}")
    print(f"  Max additivity error: {np.max(additivity_errors):.6f}")

    return np.mean(additivity_errors)


if __name__ == "__main__":
    # Test various sample sizes around the threshold
    sample_sizes = [100, 500, 799, 800, 801, 802, 1000]

    results = {}
    for n in sample_sizes:
        mae = test_sample_size(n)
        results[n] = mae

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print("Sample Size | Additivity MAE")
    print("-" * 40)
    for n, mae in results.items():
        status = "✓" if mae < 0.05 else "✗"
        print(f"{n:11d} | {mae:.6f} {status}")
