import numpy as np
import torch
from gpytorch import Module


def freeze_parameters(module: Module) -> None:
    """Disable gradient computation for all module parameters."""
    for param in module.parameters(recurse=True):
        param.requires_grad_(False)


def to_tensor(value, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Convert value to tensor, handling both scalar and tensor inputs safely.

    Args:
        value: Input value (scalar, array, or tensor)
        dtype: Target dtype for the tensor

    Returns:
        Tensor with specified dtype
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype)
    return torch.tensor(value, dtype=dtype)


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
    return float(np.mean(correlations))
