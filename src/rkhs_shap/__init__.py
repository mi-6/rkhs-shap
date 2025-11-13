"""RKHS-SHAP: Shapley Values for Kernel Methods"""

from rkhs_shap.rkhs_shap_exact import RKHSSHAP
from rkhs_shap.rkhs_shap_approx import RKHSSHAP_Approx
from rkhs_shap.kernel_ridge_regression import KernelRidgeRegressor
from rkhs_shap.kernel_approx import Nystroem_gpytorch
from rkhs_shap.shapley_regulariser import ShapleyRegulariser

__version__ = "0.1.0"

__all__ = [
    "RKHSSHAP",
    "RKHSSHAP_Approx",
    "KernelRidgeRegressor",
    "Nystroem_gpytorch",
    "ShapleyRegulariser",
]
