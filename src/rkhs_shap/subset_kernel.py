from typing import Union, Sequence, Any
import copy

import torch
from torch import Tensor
from gpytorch.kernels import Kernel
from gpytorch.lazy import LazyEvaluatedKernelTensor


class SubsetKernel(Kernel):
    """
    Wrapper kernel that restricts evaluation to a subset of input dimensions.

    This kernel creates a deep copy of the base kernel, freezes all its parameters,
    and only evaluates on the specified active dimensions. For kernels with ARD
    (Automatic Relevance Determination) lengthscales, it automatically subsets
    the lengthscale parameters to match the selected dimensions.

    Args:
        base_kernel: The kernel to wrap (will be deep-copied)
        subset_dims: Indices of dimensions to use (required)

    Example:
        >>> base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=6)
        >>> base_kernel.lengthscale = torch.tensor([1., 2., 3., 4., 5., 6.])
        >>> subset_kernel = SubsetKernel(base_kernel, subset_dims=[0, 2, 5])
        >>> x = torch.randn(10, 6)
        >>> K = subset_kernel(x, x)  # Only uses columns [0, 2, 5]
    """

    subset_dims: Tensor

    def __init__(
        self,
        base_kernel: Kernel,
        subset_dims: Union[Sequence[int], Tensor],
    ) -> None:
        super().__init__()

        self.base_kernel = copy.deepcopy(base_kernel)

        subset_dims = torch.as_tensor(subset_dims, dtype=torch.int)
        self.register_buffer("subset_dims", subset_dims)

        self._subset_kernel_params(self.base_kernel)
        self._freeze_parameters()

    def _freeze_parameters(self) -> None:
        """Disable gradient computation for all kernel parameters."""
        for param in self.base_kernel.parameters():
            param.requires_grad_(False)

    def _subset_kernel_params(self, kernel: Kernel) -> None:
        """
        Recursively subset ARD lengthscale parameters in the kernel tree.

        For kernels with ARD lengthscales, this updates both the dimensionality
        (ard_num_dims) and the lengthscale values to match the active dimensions.
        Also recursively processes nested kernels (e.g., ScaleKernel, AdditiveKernel).
        """
        if not self._should_subset_params(kernel):
            self._recurse_nested_kernels(kernel)
            return

        lengthscale = kernel.lengthscale
        lengthscale_subset = lengthscale[..., self.subset_dims].clone()

        if hasattr(kernel, "ard_num_dims"):
            kernel.ard_num_dims = len(self.subset_dims)

        if hasattr(kernel, "raw_lengthscale"):
            raw_lengthscale_subset = kernel.raw_lengthscale_constraint.inverse_transform(
                lengthscale_subset
            )
            kernel.raw_lengthscale.data = raw_lengthscale_subset.data

        self._recurse_nested_kernels(kernel)

    def _should_subset_params(self, kernel: Kernel) -> bool:
        """Check if kernel has ARD lengthscales that need subsetting."""
        if not hasattr(kernel, "lengthscale") or kernel.lengthscale is None:
            return False

        lengthscale = kernel.lengthscale
        return lengthscale.numel() > 1 and lengthscale.shape[-1] > len(self.subset_dims)

    def _recurse_nested_kernels(self, kernel: Kernel) -> None:
        """Recursively process nested kernel structures."""
        if hasattr(kernel, "base_kernel"):
            self._subset_kernel_params(kernel.base_kernel)

        if hasattr(kernel, "kernels"):
            for nested_kernel in kernel.kernels:
                self._subset_kernel_params(nested_kernel)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        **params: Any,
    ) -> Union[Tensor, LazyEvaluatedKernelTensor]:
        """
        Evaluate kernel on subsetted input dimensions.

        Args:
            x1: First input tensor of shape (..., n, d)
            x2: Second input tensor of shape (..., m, d)
            diag: If True, only compute diagonal elements
            **params: Additional kernel parameters

        Returns:
            Kernel matrix of shape (..., n, m) or (..., n) if diag=True
        """
        x1 = x1[..., self.subset_dims]
        x2 = x2[..., self.subset_dims]

        return self.base_kernel(x1, x2, diag=diag, **params)
