import numpy as np
import torch
from gpytorch.kernels import Kernel
from sklearn.cluster import KMeans
from torch import Tensor


class Nystroem:
    """Nyström approximation for GPyTorch kernels.

    Uses KMeans to select landmark points, then approximates kernel matrices
    using the Nyström method: K ≈ K_nm @ K_mm^{-1} @ K_mn where m are landmarks.
    """

    def __init__(self, kernel: Kernel, n_components: int) -> None:
        """Initialize Nyström approximation.

        Args:
            kernel: GPyTorch kernel (already fitted with appropriate lengthscale)
            n_components: Number of landmark points for approximation
        """
        self.kernel = kernel
        self.n_components = n_components
        self.landmarks = None

    def fit(self, X: np.ndarray) -> None:
        """Fit landmark points using KMeans clustering.

        Args:
            X: Training data of shape (n, m)
        """
        km = KMeans(n_clusters=self.n_components, random_state=0)
        km.fit(X)
        self.landmarks = torch.tensor(km.cluster_centers_, dtype=torch.float32)

    def transform(self, X: np.ndarray) -> Tensor:
        """Transform data using Nyström approximation.

        Args:
            X: Data to transform of shape (n, m)

        Returns:
            Transformed features of shape (n, n_components)
        """
        if self.landmarks is None:
            raise ValueError("Must call fit() before transform()")

        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Compute K_mm^{-1/2} @ K_mn
        ZT = (
            self.kernel(self.landmarks)
            .add_jitter()
            .cholesky()
            .solve(self.kernel(self.landmarks, X_tensor).to_dense())
        )

        return ZT.T

    def compute_kernel(self, X: np.ndarray) -> Tensor:
        """Compute approximated kernel matrix.

        Args:
            X: Data of shape (n, m)

        Returns:
            Approximated kernel matrix K ≈ Z @ Z.T of shape (n, n)
        """
        Z = self.transform(X)
        return Z @ Z.T
