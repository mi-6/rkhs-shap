from copy import deepcopy
from typing import Sequence, Union

import numpy as np
import torch
from gpytorch.kernels import Kernel
from gpytorch.lazy import LazyEvaluatedKernelTensor
from torch import Tensor


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
        subset_dims: Sequence[int] | np.ndarray,
    ) -> None:
        super().__init__()
        if base_kernel.active_dims is not None:
            raise NotImplementedError(
                "SubsetKernel does not support base kernels with active_dims set."
            )

        self.base_kernel = deepcopy(base_kernel)
        self.register_buffer(
            "subset_dims", torch.as_tensor(subset_dims, dtype=torch.int)
        )
        self._subset_kernel_params(self.base_kernel)

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

        if kernel.ard_num_dims is not None:
            kernel.ard_num_dims = len(self.subset_dims)

        if kernel.has_lengthscale:
            kernel.raw_lengthscale.data = kernel.raw_lengthscale.data[
                ..., self.subset_dims
            ]

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
        last_dim_is_batch: bool = False,
        **params,
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

        return self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
