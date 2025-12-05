"""Tests for SubsetKernel implementation."""

import gpytorch
import pytest
import torch

from rkhs_shap.subset_kernel import SubsetKernel
from rkhs_shap.utils import to_tensor


def test_subset_kernel_rbf():
    """Test that SubsetKernel correctly subsets dimensions."""
    input_dim = 6
    subset_dims = [0, 3, 5]
    n_samples = 5

    base_kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
    )
    base_kernel.outputscale = to_tensor(2.0)
    base_kernel.base_kernel.lengthscale = to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    subset_kernel = SubsetKernel(base_kernel, subset_dims=subset_dims)

    torch.manual_seed(42)
    x = torch.randn(n_samples, input_dim)

    # Manually compute expected kernel matrix
    x_subset = x[:, subset_dims]
    lengthscales_subset = base_kernel.base_kernel.lengthscale.squeeze()[subset_dims]
    x_scaled = x_subset / lengthscales_subset
    squared_distances = torch.cdist(x_scaled, x_scaled, p=2).pow(2)
    kernel_expected = base_kernel.outputscale * torch.exp(-0.5 * squared_distances)

    kernel_actual = subset_kernel(x, x).to_dense()

    max_error = (kernel_expected - kernel_actual).abs().max().item()
    assert max_error < 1e-6, f"Error too large: {max_error}"
    assert kernel_actual.shape == (n_samples, n_samples)
    scale_kernel = subset_kernel.base_kernel
    assert hasattr(scale_kernel, "base_kernel")
    inner_kernel = scale_kernel.base_kernel
    assert inner_kernel.lengthscale.shape[-1] == len(subset_dims)  # type: ignore[index]


def test_subset_kernel_deep_copy():
    """Test that SubsetKernel creates a deep copy of the base kernel."""
    base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=5)
    base_kernel.lengthscale = to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    subset_kernel = SubsetKernel(base_kernel, subset_dims=[0, 2, 4])
    # Modify original kernel
    base_kernel.lengthscale = to_tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    # SubsetKernel should not be affected
    expected_lengthscales = to_tensor([1.0, 3.0, 5.0])
    actual_lengthscales = subset_kernel.base_kernel.lengthscale.squeeze()
    assert torch.allclose(
        actual_lengthscales, expected_lengthscales.to(actual_lengthscales.dtype)
    )


def test_subset_kernel_different_input_shapes():
    """Test SubsetKernel with different x1 and x2 input shapes."""
    base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=5)
    base_kernel.lengthscale = torch.ones(5)
    subset_kernel = SubsetKernel(base_kernel, subset_dims=[1, 3])
    n1, n2 = 4, 6
    x1 = torch.randn(n1, 5)
    x2 = torch.randn(n2, 5)
    kernel_matrix = subset_kernel(x1, x2).to_dense()
    assert kernel_matrix.shape == (n1, n2)


def test_subset_kernel_diagonal_computation():
    """Test SubsetKernel with diag=True."""
    base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=4)
    subset_kernel = SubsetKernel(base_kernel, subset_dims=[0, 2])
    n_samples = 5
    x = torch.randn(n_samples, 4)
    kernel_diag = subset_kernel(x, x, diag=True)

    # Diagonal should be all ones for RBF kernel (self-similarity)
    assert kernel_diag.shape == (n_samples,)
    assert torch.allclose(kernel_diag, torch.ones(n_samples), atol=1e-5)  # type: ignore[arg-type]


def test_subset_kernel_with_matern_kernel():
    """Test SubsetKernel with Matern kernel."""
    input_dim = 6
    subset_dims = [1, 2, 4]
    n_samples = 4

    # Create Matern kernel with nu=2.5 and ARD
    base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
    base_kernel.lengthscale = to_tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    subset_kernel = SubsetKernel(base_kernel, subset_dims=subset_dims)

    torch.manual_seed(100)
    x = torch.randn(n_samples, input_dim)

    # Compute kernel matrix using SubsetKernel
    kernel_actual = subset_kernel(x, x).to_dense()

    # Compute expected kernel matrix manually
    x_subset = x[:, subset_dims]
    lengthscales_subset = base_kernel.lengthscale.squeeze()[subset_dims]

    # Create a fresh Matern kernel with correct dimensionality
    expected_kernel = gpytorch.kernels.MaternKernel(
        nu=2.5, ard_num_dims=len(subset_dims)
    )
    expected_kernel.lengthscale = lengthscales_subset
    kernel_expected = expected_kernel(x_subset, x_subset).to_dense()

    # Check that SubsetKernel produces the same result as manual computation
    max_error = (kernel_expected - kernel_actual).abs().max().item()
    assert max_error < 1e-6, f"Error too large: {max_error}"
    assert kernel_actual.shape == (n_samples, n_samples)

    # Check that lengthscales were correctly subsetted
    expected_lengthscales = base_kernel.lengthscale.squeeze()[subset_dims]
    actual_lengthscales = subset_kernel.base_kernel.lengthscale
    assert torch.allclose(actual_lengthscales, expected_lengthscales)

    # Check that diagonal is all ones (self-similarity)
    kernel_diag = torch.diag(kernel_actual)
    assert torch.allclose(kernel_diag, torch.ones(n_samples), atol=1e-5)

    # Check that kernel is positive semi-definite
    eigenvalues = torch.linalg.eigvalsh(kernel_actual)
    assert torch.all(eigenvalues >= -1e-6), (
        "Kernel matrix should be positive semi-definite"
    )


def test_subset_kernel_with_active_dims():
    """Test SubsetKernel with a base kernel that uses active_dims."""
    base_active_dims = (1, 3, 5, 7)
    subset_dims = [0, 2]
    base_kernel = gpytorch.kernels.RBFKernel(
        ard_num_dims=len(base_active_dims), active_dims=base_active_dims
    )
    base_kernel.lengthscale = to_tensor([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(NotImplementedError):
        SubsetKernel(base_kernel, subset_dims=subset_dims)
