import gpytorch
import numpy as np
import torch

from rkhs_shap.utils import to_tensor


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None,
        covar_module: gpytorch.kernels.Kernel | None = None,
    ) -> None:
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if covar_module is None:
            covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])

        super().__init__(train_x, train_y, likelihood)

        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x: torch.Tensor) -> torch.distributions.Distribution:
        self.eval()
        self.likelihood.eval()
        with torch.inference_mode():
            return self.likelihood(self(x))

    def predict_mean_numpy(self, x: np.ndarray) -> np.ndarray:
        x_tensor = to_tensor(x)
        posterior = self.predict(x_tensor)
        return posterior.mean.numpy()

    def fit(self, training_iter: int = 50, lr: float = 0.1) -> None:
        assert self.train_inputs is not None
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
    train_y_array = np.asarray(train_y).reshape(-1, 1)
    train_y = output_scaler.fit_transform(train_y_array).flatten()
    test_y_array = np.asarray(test_y).reshape(-1, 1)
    test_y = output_scaler.transform(test_y_array).flatten()

    train_x = torch.tensor(X_train)
    train_y = torch.tensor(train_y)
    test_x = torch.tensor(X_test)
    test_y = torch.tensor(test_y)

    print("Testing RBF Kernel:")
    model = ExactGPModel(train_x, train_y)
    model.fit(training_iter=50, lr=0.1)
    test_pred = model.predict(test_x)

    ss_res = torch.sum((test_y - test_pred.mean) ** 2)
    ss_tot = torch.sum((test_y - torch.mean(test_y)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)

    print("\nModel Evaluation:")
    print(f"R² test-score: {r2_score.item():.4f}")

    print("\n" + "=" * 50)
    print("Testing Matern Kernel:")
    matern_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1])
    model_matern = ExactGPModel(train_x, train_y, covar_module=matern_kernel)
    model_matern.fit(training_iter=50, lr=0.1)
    test_pred_matern = model_matern.predict(test_x)

    ss_res_matern = torch.sum((test_y - test_pred_matern.mean) ** 2)
    ss_tot_matern = torch.sum((test_y - torch.mean(test_y)) ** 2)
    r2_score_matern = 1 - (ss_res_matern / ss_tot_matern)

    print("\nModel Evaluation:")
    print(f"R² test-score: {r2_score_matern.item():.4f}")
