"""
Ref: https://github.com/saugatkandel/AI-ML_Control_System/blob/66ed73afa80d2746bae8126d0cbe3c0ea570f141/work_directory/34-ID/jupyter/botorch_test/turbo_1.ipynb#L34
"""
import logging

import botorch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
import gpytorch
import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt

from autobl.steering.configs import *
from autobl.steering.acquisition import PosteriorStandardDeviationDerivedAcquisition
from autobl.steering.model import ProjectedSpaceSingleTaskGP
from autobl.util import *


class StoppingCriterion:

    def __init__(self, configs: StoppingCriterionConfig, guide):
        self.configs = configs
        self.guide = guide

    def check(self):
        if self.configs is None:
            return False
        if self.guide.n_update_calls < self.configs.n_updates_to_begin:
            return False
        if (self.guide.n_update_calls - self.configs.n_updates_to_begin) % self.configs.n_check_interval != 0:
            return False
        if self.configs.method == 'max_uncertainty':
            return self.check_max_uncertainty()

    def check_max_uncertainty(self):
        t = self.configs.params['threshold']
        x = torch.linspace(0, 1, self.guide.data_x.shape[0] * 5).view(-1, 1)
        mu, sigma = self.guide.get_posterior_mean_and_std(x, transform=False, untransform=True)
        sigma = sigma.squeeze()
        if self.guide.acqf_weight_func is not None:
            w = torch.clip(self.guide.acqf_weight_func(x).squeeze(), 0, 1)
        else:
            w = 1.0
        max_sigma = (sigma * w).max()
        if max_sigma < t:
            logging.info('Stopping criterion triggered: max sigma {} < {}.'.format(max_sigma, t))
            return True
        else:
            return False


class ExperimentGuide:

    def __init__(self, config, *args, **kwargs):
        self.config = config

    def build(self, *args, **kwargs):
        pass

    def suggest(self):
        pass

    def update(self, *args, **kwargs):
        pass


class GPExperimentGuide(ExperimentGuide):

    def __init__(self, config: GPExperimentGuideConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model = None
        self.fitting_func = None
        self.acquisition_function = None
        self.optimizer = None
        self.data_x = torch.tensor([])
        self.data_y = torch.tensor([])
        self.input_transform = None
        self.outcome_transform = None
        self.n_suggest_calls = 0
        self.n_update_calls = 0
        self.stopping_criterion = StoppingCriterion(self.config.stopping_criterion_configs, self)

    def build(self, x_train=None, y_train=None):
        """
        Build model, fit hyperparameters, and initialize other variables.

        :param x_train: Optional[Tensor]. Features of the data for training the GP model and finding hyperparameters
                        (e.g., kernel paraemters).
        :param y_train: Optional[Tensor]. Observations of the data for training the GP model and finding hyperparameters
                        (e.g., kernel paraemters).
        """
        self.build_counters()
        self.build_transform()
        self.build_model(x_train, y_train)
        self.build_acquisition_function()
        self.build_optimizer()

    def build_counters(self):
        self.n_suggest_calls = 0
        self.n_update_calls = 0

    def build_transform(self):
        self.input_transform = Normalize(d=self.config.dim_measurement_space)
        self.outcome_transform = Standardize(m=self.config.dim_measurement_space)

    def record_data(self, x, y):
        self.data_x = torch.concatenate([self.data_x, x])
        self.data_y = torch.concatenate([self.data_y, y])

    def build_model(self, x_train, y_train):
        x_train, y_train = self.transform_data(x_train, y_train, train=True)
        self.train_model(x_train, y_train)
        self.record_data(x_train, y_train)

    def build_acquisition_function(self):
        if issubclass(self.config.acquisition_function_class, botorch.acquisition.AnalyticAcquisitionFunction):
            assert self.config.num_candidates == 1, ('Since an analytical acquisition function is used, '
                                                     'num_candidates must be 1.')
        additional_params = {}
        if issubclass(self.config.acquisition_function_class, PosteriorStandardDeviationDerivedAcquisition):
            additional_params['input_transform'] = self.input_transform

        self.acquisition_function = self.config.acquisition_function_class(
            self.model,
            **additional_params,
            **self.config.acquisition_function_params,
            posterior_transform=botorch.acquisition.objective.UnstandardizePosteriorTransform(
                Y_mean=self.outcome_transform.means[0], Y_std=self.outcome_transform.stdvs[0])
        )

    def build_optimizer(self):
        self.optimizer = self.config.optimizer_class(
            bounds=self.get_bounds(),
            num_candidates=self.config.num_candidates,
            **self.config.optimizer_params
        )

    def get_bounds(self):
        lb = self.config.lower_bounds
        if lb is None:
            lb = torch.tensor([-np.inf] * self.config.dim_measurement_space)
        ub = self.config.upper_bounds
        if ub is None:
            ub = torch.tensor([np.inf] * self.config.dim_measurement_space)
        if not isinstance(lb, torch.Tensor):
            lb = torch.tensor(lb)
        lb, _ = self.transform_data(lb)
        if not isinstance(ub, torch.Tensor):
            ub = torch.tensor(ub)
        ub, _ = self.transform_data(ub)
        return torch.stack([lb, ub])

    def scale_by_normalizer_bounds(self, x, dim=0):
        """
        Scale data by 1 / span_of_normalizer_bounds.

        :param x: Any. The input data.
        :param dim: int. Use the `dim`-th dimension of the normalizer bounds to calculate the scaling factor.
                    If x has a shape of [n, ..., d] where d equals to the number of dimensions of the bounds,
                    the scaling factors are calculated separately for each dimension and the `dim` argument is
                    disregarded.
        :return:
        """
        if isinstance(x, torch.Tensor) and x.ndim >= 2:
            return x / (self.input_transform.bounds[1] - self.input_transform.bounds[0])
        else:
            s = self.input_transform.bounds[1][dim] - self.input_transform.bounds[0][dim]
            return x / s

    def unscale_by_normalizer_bounds(self, x, dim=0):
        """
        Scale data by span_of_normalizer_bounds.

        :param x: Any. The input data.
        :param dim: int. Use the `dim`-th dimension of the normalizer bounds to calculate the scaling factor.
                    If x has a shape of [n, ..., d] where d equals to the number of dimensions of the bounds,
                    the scaling factors are calculated separately for each dimension and the `dim` argument is
                    disregarded.
        :return:
        """
        if isinstance(x, torch.Tensor) and x.ndim >= 2:
            return x * (self.input_transform.bounds[1] - self.input_transform.bounds[0])
        else:
            s = self.input_transform.bounds[1][dim] - self.input_transform.bounds[0][dim]
            return x * s

    def transform_data(self, x=None, y=None, train=False):
        if x is not None:
            x = x.double()
        if y is not None:
            y = y.double()
        if x is not None and self.input_transform is not None:
            do_squeeze = False
            if x.ndim == 1:
                x = x[:, None]
                do_squeeze = True
            if train:
                self.input_transform.train()
            else:
                self.input_transform.eval()
            x = self.input_transform(x)
            if do_squeeze:
                x = x[:, 0]
        if y is not None and self.outcome_transform is not None:
            if train:
                self.outcome_transform.train()
            else:
                self.outcome_transform.eval()
            y, _ = self.outcome_transform(y)
        return x, y

    def untransform_data(self, x=None, y=None):
        if x is not None and self.input_transform is not None:
            self.input_transform.training = False
            x = self.input_transform.untransform(x)
        if y is not None and self.outcome_transform is not None:
            self.outcome_transform.training = False
            y, _ = self.outcome_transform.untransform(y)
        return x, y

    def untransform_posterior(self, posterior):
        posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior

    def suggest(self):
        candidate, acq_val = self.optimizer.maximize(self.acquisition_function)
        candidate, _ = self.untransform_data(candidate)
        self.n_suggest_calls += 1
        return candidate

    def update(self, x_data, y_data):
        """
        Update the model using newly measured data.

        :param x_data: Tensor. Features of new data.
        :param y_data: Tensor. Observations of new data.
        """
        x_data, y_data = self.transform_data(x_data, y_data, train=False)
        self.record_data(x_data, y_data)
        additional_params = {}
        if self.config.noise_variance is not None:
            additional_params['noise'] = torch.full_like(y_data, self.config.noise_variance)
        new_model = self.model.condition_on_observations(x_data, y_data, **additional_params)
        # In-place update all attribute in self.model so that all references get updated.
        self.model.__dict__ = new_model.__dict__
        if hasattr(self.acquisition_function, 'update_hyperparams_following_schedule'):
            self.acquisition_function.update_hyperparams_following_schedule()
        self.n_update_calls += 1

    def create_model_object(self, x_data, y_data):
        # Create model and compute covariance matrix.
        assert not ('train_Yvar' in self.config.model_params.keys() and self.config.noise_variance is not None)
        additional_params = {}
        if self.config.noise_variance is not None:
            additional_params['train_Yvar'] = torch.full_like(y_data, self.config.noise_variance)
        self.model = self.config.model_class(x_data, y_data, **self.config.model_params, **additional_params)

    def train_model(self, x_data, y_data):
        self.create_model_object(x_data, y_data)

        # Fit hyperparameters.
        logging.info('Kernel lengthscale before optimization (normalized & standardized): {}'.format(
            to_numpy(self.model.covar_module.lengthscale))
        )
        self.fitting_func = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        botorch.fit.fit_gpytorch_mll(self.fitting_func)
        logging.info('Kernel lengthscale after optimization (normalized & standardized): {}'.format(
            to_numpy(self.model.covar_module.lengthscale))
        )

        # Override kernel lengthscale if applicable.
        if self.config.override_kernel_lengthscale is not None:
            self.model.covar_module.lengthscale = self.scale_by_normalizer_bounds(
                self.config.override_kernel_lengthscale
            )
            logging.info('Kernel lengthscale overriden to: {} ({} after normalization)'.format(
                self.config.override_kernel_lengthscale,
                self.scale_by_normalizer_bounds(self.config.override_kernel_lengthscale)))

    def get_posterior_mean_and_std(self, x, transform=True, untransform=True):
        if transform:
            x_transformed, _ = self.transform_data(x, None, train=False)
        else:
            x_transformed = x
        posterior = self.model.posterior(x_transformed)
        if untransform:
            posterior = self.untransform_posterior(posterior)
        mu = posterior.mean
        sigma = posterior.variance.clamp_min(1e-12).sqrt()
        return mu, sigma

    def plot_posterior(self, x, ax=None):
        """
        Plot the posterior mean and standard deviation of the GP model. Only works with 1-dimension feature space.

        :param x: torch.Tensor[float, ...]. The points to plot.
        """
        if not isinstance(x, torch.Tensor):
            x = to_tensor(x)
        if x.ndim == 1:
            x = x[:, None]
        mu, sigma = self.get_posterior_mean_and_std(x)
        mu = mu.reshape(-1).cpu().detach().numpy()
        sigma = sigma.reshape(-1).cpu().detach().numpy()
        x_transformed, _ = self.transform_data(x, None, train=False)
        acq = self.acquisition_function(x_transformed.view(-1, 1, 1)).reshape(-1).cpu().detach().numpy()

        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        x = np.squeeze(x)
        external_ax = True
        if ax is None:
            external_ax = False
            fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        if not isinstance(ax, (list, tuple, np.ndarray)):
            ax = [ax]
        ax[0].plot(x, mu, label='Posterior mean', linewidth=0.5)
        ax[0].fill_between(x, mu - sigma, mu + sigma, alpha=0.5)
        data_x, data_y = self.untransform_data(self.data_x, self.data_y)
        ax[0].scatter(to_numpy(data_x.reshape(-1)), to_numpy(data_y.reshape(-1)), label='Measured data', s=4)
        ax[0].set_title('Posterior mean and $+/-\sigma$ interval')
        if len(ax) > 1:
            ax[1].plot(x, acq)
            ax[1].set_title('Acquisition function')
        if external_ax:
            return ax
        else:
            plt.show()


class XANESExperimentGuide(GPExperimentGuide):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acqf_weight_func = None
        self.feature_projection_func = None

    def build_model(self, x_train, y_train):
        x_train, y_train = self.transform_data(x_train, y_train, train=True)
        if issubclass(self.config.model_class, ProjectedSpaceSingleTaskGP):
            self.build_feature_projection_function(x_train, y_train)
        self.train_model(x_train, y_train)
        self.record_data(x_train, y_train)

    def create_model_object(self, x_data, y_data):
        # Create model and compute covariance matrix.
        assert not ('train_Yvar' in self.config.model_params.keys() and self.config.noise_variance is not None)
        additional_params = {}
        if self.config.noise_variance is not None:
            additional_params['train_Yvar'] = torch.full_like(y_data, self.config.noise_variance)
        if issubclass(self.config.model_class, ProjectedSpaceSingleTaskGP):
            additional_params['projection_function'] = self.feature_projection_func
        self.model = self.config.model_class(x_data, y_data, **self.config.model_params, **additional_params)

    def update(self, x_data, y_data):
        super().update(x_data, y_data)
        if self.config.n_updates_create_acqf_weight_func is not None and \
                self.n_update_calls == self.config.n_updates_create_acqf_weight_func:
            logging.info('Building acquisition function mask with floor value {}.'.format(self.config.acqf_weight_func_floor_value))
            self.build_acqf_weight_function(floor_value=self.config.acqf_weight_func_floor_value)
            if hasattr(self.acquisition_function, 'set_weight_func'):
                self.acquisition_function.set_weight_func(self.acqf_weight_func)

    def build_acquisition_function(self):
        super().build_acquisition_function()
        if hasattr(self.acquisition_function, 'set_weight_func'):
            self.acquisition_function.set_weight_func(self.acqf_weight_func)

    def build_acqf_weight_function(self, floor_value=0.1):
        """
        Create and set a mask function to the acquisition function, such that it lowers the values in pre-edge
        regions.
        """
        if not hasattr(self.acquisition_function, 'set_weight_func'):
            logging.warning('Acquisition function does not have attribute "set_weight_func".')
            self.acqf_weight_func = None
            return
        x = torch.linspace(0, 1, self.data_x.shape[0] * 10).reshape(-1, 1)
        mu, _ = self.get_posterior_mean_and_std(x, transform=False)
        mu = mu.squeeze()
        # convert 3 eV to pixel
        gaussian_grad_sigma = 3.0 / (self.input_transform.bounds[1][0] - self.input_transform.bounds[0][0]) * len(x)
        gaussian_grad_sigma = float(gaussian_grad_sigma)
        mu = scipy.ndimage.gaussian_filter(to_numpy(mu), sigma=gaussian_grad_sigma)
        mu_grad = scipy.signal.convolve(np.pad(mu, [1, 1], mode='edge'), [0.5, 0, -0.5], mode='valid')

        # convert 3 eV to pixel
        min_peak_width = float(3.0 / (self.input_transform.bounds[1][0] - self.input_transform.bounds[0][0]) * len(x))
        peak_locs, peak_properties = scipy.signal.find_peaks(mu_grad, height=0.05, width=min_peak_width)
        max_peak_ind = np.argmax(peak_properties['peak_heights'])

        peak_loc_normalized = float(peak_locs[max_peak_ind]) / len(x)
        peak_width_normalized = peak_properties['widths'][max_peak_ind] / len(x)

        def weight_func(x):
            m = sigmoid(x, r=20. / peak_width_normalized, d=peak_loc_normalized - 1.6 * peak_width_normalized)
            m = m + gaussian(x,
                             a=self.config.acqf_weight_func_post_edge_gain,
                             mu=peak_loc_normalized + self.config.acqf_weight_func_post_edge_offset * peak_width_normalized,
                             sigma=peak_width_normalized * self.config.acqf_weight_func_post_edge_width,
                             c=0.0)
            m = m * (1 - floor_value) + floor_value
            return m

        self.acqf_weight_func = weight_func
        return

    def build_feature_projection_function(self, x_train, y_train):
        if len(x_train) < 4:
            raise ValueError('To build a proper projection function, initial data should have at least 4 points.')
        x_dat = to_numpy(x_train.squeeze())
        y_dat = to_numpy(y_train.squeeze())

        x_dense = np.linspace(0, 1, len(x_train) * 10)
        y_dense = scipy.interpolate.CubicSpline(x_dat, y_dat)(x_dense)
        grad_y = (y_dense[2:] - y_dense[:-2]) / (x_dense[2:] - x_dense[:-2])

        peak_locs, peak_properties = scipy.signal.find_peaks(grad_y, height=grad_y.max() * 0.5, width=1)
        edge_ind = np.argmax(peak_properties['peak_heights'])
        peak_loc = float(peak_locs[edge_ind]) / len(x_dense) + (x_dense[1] - x_dense[0])
        peak_width = peak_properties['widths'][edge_ind] / len(x_dense)

        def sparseness_function(x):
            # A basin function that looks like this: ```\___/```
            lower_bound = self.config.project_func_sparseness_lower_bound
            s = (sigmoid(x, r=1 / peak_width, d=peak_loc - peak_width * 5) +
                 sigmoid(x, r=-1 / peak_width, d=peak_loc + peak_width * 50))
            s = (s - 1.0) * (1.0 - lower_bound) + lower_bound
            return s

        sparseness = sparseness_function(x_dense)

        cdf = np.cumsum(sparseness)
        cdf = cdf - cdf[0]
        cdf = cdf / cdf[-1]
        cdf = to_numpy(cdf)
        mapper = lambda test_pts: np.interp(test_pts, x_dense, cdf)

        def projection_func(x):
            x_n = to_numpy(x)
            x_n = mapper(x_n)
            x = torch.tensor(x_n, device=x.device)
            return x

        self.feature_projection_func = projection_func
