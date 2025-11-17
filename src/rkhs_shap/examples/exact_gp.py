from typing import Optional

import gpytorch
import numpy as np
import torch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Optional[gpytorch.likelihoods.GaussianLikelihood] = None,
    ) -> None:
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean()

        # TODO: try with scale kernel
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            return self.likelihood(self(x))

    def predict_mean_numpy(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x)
        posterior = self.predict(x_tensor)
        return posterior.mean.numpy()

    def fit(self, training_iter: int = 50, lr: float = 0.1) -> None:
        train_x = self.train_inputs[0]
        train_y = self.train_targets
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self(train_x)
            loss = mll(output, train_y) * -1
            loss.backward()

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    "iter %d/%d - loss: %.3f {} noise: %.3f"
                    % (
                        i + 1,
                        training_iter,
                        loss.detach().item(),
                        self.likelihood.noise.detach(),
                    )
                )
                # print(f"Lengthscale: {self.covar_module.lengthscale.detach()}")
            optimizer.step()

        self.eval()
        self.likelihood.eval()

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.covar_module.lengthscale


if __name__ == "__main__":
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    X, y = shap.datasets.california(n_points=500)

    X_train, X_test, train_y, test_y = train_test_split(
        X.values, y, test_size=0.2, random_state=42
    )

    input_scaler = MinMaxScaler()
    X_train = input_scaler.fit_transform(X_train)
    X_test = input_scaler.transform(X_test)

    output_scaler = StandardScaler()
    train_y = output_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
    test_y = output_scaler.transform(test_y.reshape(-1, 1)).flatten()

    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)
    test_x = torch.tensor(X_test, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    model = ExactGPModel(train_x, train_y)
    model.fit(training_iter=50, lr=0.1)
    test_pred = model.predict(test_x)

    ss_res = torch.sum((test_y - test_pred.mean) ** 2)
    ss_tot = torch.sum((test_y - torch.mean(test_y)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    print("\nModel Evaluation:")
    print(f"R² test-score: {r2_score.item():.4f}")
