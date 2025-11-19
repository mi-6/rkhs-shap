import gpytorch
import numpy as np
import torch

from rkhs_shap.utils import (
    calculate_additivity_mae,
    calculate_correlation,
    freeze_parameters,
    to_tensor,
)


class TestToTensor:
    def test_from_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor(arr)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_from_tensor(self):
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        tensor = to_tensor(original)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, original.float())

    def test_from_scalar(self):
        tensor = to_tensor(5.0)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.item() == 5.0


class TestCalculateAdditivityMae:
    def test_perfect_additivity(self):
        shap_values = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.0]])
        baseline = 10.0
        model_preds = torch.tensor([16.0, 14.0])
        mae = calculate_additivity_mae(shap_values, model_preds, baseline)
        assert mae == 0.0

    def test_imperfect_additivity(self):
        shap_values = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.0]])
        baseline = 10.0
        model_preds = torch.tensor([17.0, 15.0])
        mae = calculate_additivity_mae(shap_values, model_preds, baseline)
        assert mae == 1.0


class TestCalculateCorrelation:
    def test_perfect_correlation(self):
        values1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        values2 = values1.copy()
        corr = calculate_correlation(values1, values2)
        assert np.isclose(corr, 1.0)

    def test_negative_correlation(self):
        values1 = np.array([[1.0, 2.0, 3.0]])
        values2 = np.array([[3.0, 2.0, 1.0]])
        corr = calculate_correlation(values1, values2)
        assert np.isclose(corr, -1.0)

    def test_zero_correlation(self):
        values1 = np.array([[1.0, 2.0, 3.0]])
        values2 = np.array([[2.0, 2.0, 2.0]])
        corr = calculate_correlation(values1, values2)
        assert np.isnan(corr)


class TestFreezeParameters:
    def test_freeze_linear_layer(self):
        layer = torch.nn.Linear(10, 5)
        assert all(p.requires_grad for p in layer.parameters())

        freeze_parameters(layer)
        assert all(not p.requires_grad for p in layer.parameters())

    def test_freeze_gpytorch_kernel(self):
        kernel = gpytorch.kernels.RBFKernel()
        kernel.lengthscale = 1.0
        assert kernel.raw_lengthscale.requires_grad

        freeze_parameters(kernel)
        assert not kernel.raw_lengthscale.requires_grad
