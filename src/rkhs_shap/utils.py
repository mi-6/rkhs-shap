from gpytorch import Module


def freeze_parameters(module: Module) -> None:
    """Disable gradient computation for all module parameters."""
    for param in module.parameters(recurse=True):
        param.requires_grad_(False)
