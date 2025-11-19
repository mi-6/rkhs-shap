import torch
from gpytorch import Module


def freeze_parameters(module: Module) -> None:
    """Disable gradient computation for all module parameters."""
    for param in module.parameters(recurse=True):
        param.requires_grad_(False)


def to_tensor(value, dtype: torch.dtype = torch.float32) -> torch.Tensor:
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
