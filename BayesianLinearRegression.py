"""
There is no problem with the code.
 I think the bug is about PyCharm
 (it cannot recognise the wrapped pytorch classes in the code editor window).
 But I can run the code successfully.
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# For reproducibility
pyro.set_rng_seed(0)

class BayesianLinearRegression:
    def __init__(self):
        self.generate_data()
        self.define_model_and_guide()
        self.run_inference()
        self.posterior_predictive()
        self.plot_results()

    def generate_data(self):
        """Step 1: Simulate some linear data with noise"""
        self.true_slope = 2.0
        self.true_intercept = 1.0
        self.n_samples = 100
        self.X = torch.linspace(0, 10, self.n_samples)
        self.Y = self.true_intercept + self.true_slope * self.X + torch.randn(self.n_samples)

    def model(self, X, Y=None):
        """Step 2: Define the probabilistic model"""
        slope = pyro.sample("slope", dist.Normal(0., 10.))
        intercept = pyro.sample("intercept", dist.Normal(0., 10.))
        sigma = pyro.sample("sigma", dist.HalfNormal(1.0))

        mean = intercept + slope * X
        with pyro.plate("data", len(X)):
            pyro.sample("obs", dist.Normal(mean, sigma), obs=Y)

    def guide(self, X, Y=None):
        """Step 3: Define the variational guide (approximate posterior)"""
        slope_loc = pyro.param("slope_loc", torch.tensor(0.0))
        slope_scale = pyro.param("slope_scale", torch.tensor(1.0), constraint=dist.constraints.positive)

        intercept_loc = pyro.param("intercept_loc", torch.tensor(0.0))
        intercept_scale = pyro.param("intercept_scale", torch.tensor(1.0), constraint=dist.constraints.positive)

        sigma_loc = pyro.param("sigma_loc", torch.tensor(1.0), constraint=dist.constraints.positive)

        pyro.sample("slope", dist.Normal(slope_loc, slope_scale))
        pyro.sample("intercept", dist.Normal(intercept_loc, intercept_scale))
        pyro.sample("sigma", dist.HalfNormal(sigma_loc))

    def define_model_and_guide(self):
        """Setup optimizer and inference object"""
        self.optimizer = Adam({"lr": 0.01})
        self.svi = SVI(self.model, self.guide, self.optimizer, loss=Trace_ELBO())

    def run_inference(self, num_steps=1000):
        """Step 4: Run variational inference"""
        losses = []
        for step in range(num_steps):
            loss = self.svi.step(self.X, self.Y)
            losses.append(loss)
            if step % 100 == 0:
                print(f"[Step {step}] ELBO Loss: {loss:.2f}")
        self.losses = losses

    def posterior_predictive(self, num_samples=1000):
        """Step 5: Draw samples from the posterior"""
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
        self.posterior_samples = predictive(self.X, self.Y)

    def plot_results(self):
        """Step 6: Visualize posterior distributions and regression lines"""
        slope_samples = self.posterior_samples["slope"]
        intercept_samples = self.posterior_samples["intercept"]
        sigma_samples = self.posterior_samples["sigma"]

        # Print estimated mean values
        print("\nPosterior Estimates:")
        print(f"Mean Slope: {slope_samples.mean().item():.3f}")
        print(f"Mean Intercept: {intercept_samples.mean().item():.3f}")
        print(f"Mean Sigma: {sigma_samples.mean().item():.3f}")

        # Plot posterior distributions
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        sns.kdeplot(slope_samples, ax=axs[0], shade=True)
        axs[0].set_title("Posterior of Slope")

        sns.kdeplot(intercept_samples, ax=axs[1], shade=True)
        axs[1].set_title("Posterior of Intercept")

        sns.kdeplot(sigma_samples, ax=axs[2], shade=True)
        axs[2].set_title("Posterior of Sigma")

        plt.tight_layout()
        plt.show()

        # Plot regression lines sampled from the posterior
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X.numpy(), self.Y.numpy(), label="Data", color="black", s=10)

        for i in range(100):
            y_pred = intercept_samples[i] + slope_samples[i] * self.X
            plt.plot(self.X.numpy(), y_pred.detach().numpy(), color="blue", alpha=0.1)

        plt.title("Posterior Predictive Regression Lines")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


# Run the Bayesian Linear Regression
blr = BayesianLinearRegression()
