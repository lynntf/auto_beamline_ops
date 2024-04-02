"""
Ref: https://github.com/saugatkandel/AI-ML_Control_System/blob/66ed73afa80d2746bae8126d0cbe3c0ea570f141/work_directory/34-ID/jupyter/botorch_test/turbo_1.ipynb#L34
"""
import logging

import botorch
import gpytorch
import numpy as np
import torch
import matplotlib.pyplot as plt

from autobl.steering.configs import *

class ExperimentGuide:

    def __init__(self, config: ExperimentGuideConfig, *args, **kwargs):
        self.config = config

    def build(self, *args, **kwargs):
        pass

    def suggest(self):
        pass

    def update(self, *args, **kwargs):
        pass

    def get_bounds(self):
        lb = self.config.lower_bounds
        if lb is None:
            lb = torch.tensor([-np.inf] * self.config.dim_measurement_space)
        ub = self.config.upper_bounds
        if ub is None:
            ub = torch.tensor([np.inf] * self.config.dim_measurement_space)
        if not isinstance(lb, torch.Tensor):
            lb = torch.from_numpy(lb)
        if not isinstance(ub, torch.Tensor):
            ub = torch.from_numpy(ub)
        return torch.stack([lb, ub])


class GPExperimentGuide(ExperimentGuide):

    def __init__(self, config: GPExperimentGuideConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        assert isinstance(self.config, GPExperimentGuideConfig)
        self.model = None
        self.fitting_func = None
        self.acquisition_function = None
        self.optimizer = None
        self.data_x = torch.tensor([])
        self.data_y = torch.tensor([])

    def build(self, x_train=None, y_train=None):
        """
        Build model, fit hyperparameters, and initialize other variables.

        :param x_train: Optional[Tensor]. Features of the data for training the GP model and finding hyperparameters
                        (e.g., kernel paraemters).
        :param y_train: Optional[Tensor]. Observations of the data for training the GP model and finding hyperparameters
                        (e.g., kernel paraemters).
        """
        self.build_model(x_train, y_train)
        self.build_acquisition_function()
        self.build_optimizer()

    def record_data(self, x, y):
        self.data_x = torch.concatenate([self.data_x, x])
        self.data_y = torch.concatenate([self.data_y, y])

    def build_model(self, x_train, y_train):
        self.train_model(x_train, y_train)
        self.record_data(x_train, y_train)

    def build_acquisition_function(self):
        if issubclass(self.config.acquisition_function_class, botorch.acquisition.AnalyticAcquisitionFunction):
            assert self.config.num_candidates == 1, ('Since an analytical acquisition function is used, '
                                                     'num_candidates must be 1.')
        self.acquisition_function = self.config.acquisition_function_class(
            self.model,
            **self.config.acquisition_function_params
        )

    def build_optimizer(self):
        self.optimizer = self.config.optimizer_class(
            bounds=self.get_bounds(),
            num_candidates=self.config.num_candidates,
            **self.config.optimizer_params
        )

    def suggest(self):
        candidate, acq_val = self.optimizer.maximize(self.acquisition_function)
        return candidate

    def update(self, x_data, y_data):
        """
        Update the model using newly measured data.

        :param x_data: Tensor. Features of new data.
        :param y_data: Tensor. Observations of new data.
        """
        self.record_data(x_data, y_data)
        self.model = self.model.condition_on_observations(x_data, y_data)
        # condition_on_observations does not make in-place changes to the model object but creates a new object, so
        # we need to reset the model object in the acquisition function.
        self.build_acquisition_function()

    def train_model(self, x_data, y_data):
        # Create model and compute covariance matrix.
        self.model = self.config.model_class(x_data, y_data, **self.config.model_params)
        # Fit hyperparameters.
        logging.info('Kernel lengthscale before optimization (normalized & standardized): {}'.format(
            self.model.covar_module.lengthscale)
        )
        self.fitting_func = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        botorch.fit.fit_gpytorch_mll(self.fitting_func)
        logging.info('Kernel lengthscale after optimization (normalized & standardized): {}'.format(
            self.model.covar_module.lengthscale)
        )

    def plot_posterior(self, x):
        """
        Plot the posterior mean and standard deviation of the GP model. Only works with 1-dimension feature space.

        :param x: torch.Tensor[float, ...]. The points to plot.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        if x.ndim == 1:
            x = x[:, None]
        posterior = self.model.posterior(x)
        mu = posterior.mean.reshape(-1).cpu().detach().numpy()
        sigma = posterior.variance.clamp_min(1e-12).sqrt().reshape(-1).cpu().detach().numpy()
        acq = self.acquisition_function(x.view(-1, 1, 1)).reshape(-1).cpu().detach().numpy()

        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        x = np.squeeze(x)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].plot(x, mu)
        ax[0].fill_between(x, mu - sigma, mu + sigma, alpha=0.5)
        ax[0].scatter(self.data_x.reshape(-1), self.data_y.reshape(-1))
        ax[0].set_title('Posterior mean and $+/-\sigma$ interval')
        ax[1].plot(x, acq)
        ax[1].set_title('Acquisition function')
        plt.show()
