# RKHS-SHAP: Shapley Values for Kernel Methods

This repository is based on a fork of [RKHS-SHAP](https://github.com/Chau999/RKHS-SHAP).

RKHS-SHAP computes Shapley values for kernel-based models (e.g. Kernel Ridge Regression, Gaussian Processes) using kernel mean/conditional embeddings. Supports both interventional and observational SHAP with exact and Nyström-approximate algorithms.

## Paper

📄 **[RKHS-SHAP: Shapley Values for Kernel Methods](https://arxiv.org/pdf/2110.09167)**

## What's working/tested

- Exact RKHS-SHAP with full coalition enumeration
- Monte Carlo coalition sampling for high-dimensional features
- Interventional (I) and Observational (O) Shapley values
- Nyström approximation for large-scale datasets
- Additivity/efficiency property validated in unit tests
- Integration with GPyTorch kernels and mean functions

## Installation

```bash
# Install with uv (Python >=3.11)
uv sync

# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests
```
