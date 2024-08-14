"""
File: network_layers.py
Author: Anna-Lena Popkes, Hiske Overweg
Description: This file contains all different network layer types
"""



from torch.distributions import HalfCauchy
import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.special import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


import torch.distributions as dists

# %%

# Run on GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Bayesian Layer parameters
bayesian_weight_rho_scale = -4.
bayesian_bias_rho_scale = -4.
bayesian_scale = 1
PI = 1
SIGMA1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA2 = torch.cuda.FloatTensor([math.exp(-6)])
# hyper-parameters of horseshoe prior
global_cauchy_scale = 1.
weight_cauchy_scale = 1.
beta_rho_scale = -7.
log_tau_mean = 1
log_tau_rho_scale = -6.
bias_rho_scale = -7.
log_v_mean = 1
log_v_rho_scale = -6.


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass


class ReparametrizedGaussian(Distribution):
    """
    Diagonal ReparametrizedGaussian distribution with parameters mu (mean) and rho. The standard
    deviation is parametrized as sigma = log(1 + exp(rho))

    A sample from the distribution can be obtained by sampling from a unit Gaussian,
    shifting the samples by the mean and scaling by the standard deviation:
    w = mu + log(1 + exp(rho)) * epsilon
    """

    def __init__(self, mu, rho):
        self.mean = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.point_estimate = self.mean

    @property
    def std_dev(self):
        return torch.log1p(torch.exp(self.rho))

    # def sample(self, n_samples=1):
    #  epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=(n_samples, *self.mean.size()))
    # return self.mean+ self.std_dev * epsilon
    def sample(self):
        epsilon = torch.randn_like(self.rho)  # "randn" samples from standard normal distr
        return self.mean + torch.log(1 + torch.exp(self.rho)) * epsilon

    def logprob(self, target):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.std_dev)
                - ((target - self.mean) ** 2) / (2 * self.std_dev ** 2)).sum()

    def entropy(self):
        """
        Computes the entropy of the Diagonal Gaussian distribution.
        Details on the computation can be found in the 'diagonal_gaussian_entropy' notes in the repo
        """
        if self.mean.dim() > 1:
            n_inputs, n_outputs = self.mean.shape
        else:
            n_inputs = len(self.mean)
            n_outputs = 1

        part1 = (n_inputs * n_outputs) / 2 * (torch.log(torch.tensor([2 * math.pi])) + 1)
        part2 = torch.sum(torch.log(self.std_dev))

        return part1 + part2


class InverseGamma(Distribution):
    """ Inverse Gamma distribution """

    def __init__(self, shape, rate):
        """
        Class constructor, sets parameters of the distribution.

        Args:
            shape: torch tensor of floats, shape parameters of the distribution
            rate: torch tensor of floats, rate parameters of the distribution
        """
        self.shape = shape
        self.rate = rate

    def exp_inverse(self):
        """
        Calculates the expectation E[1/x], where x follows
        the inverse gamma distribution
        """
        return self.shape / self.rate

    def exp_log(self):
        """
        Calculates the expectation E[log(x)], where x follows
        the inverse gamma distribution
        """
        exp_log = torch.log(self.rate) - torch.digamma(self.shape)
        return exp_log

    def entropy(self):
        """
        Calculates the entropy of the inverse gamma distribution
        """
        entropy = self.shape + torch.log(self.rate) + torch.lgamma(self.shape) \
                  - (1 + self.shape) * torch.digamma(self.shape)
        return torch.sum(entropy)

    def logprob(self, target):
        """
        Computes the value of the predictive log likelihood at the target value

        Args:
            target: Torch tensor of floats, point(s) to evaluate the logprob

        Returns:
            loglike: float, the log likelihood
        """
        part1 = (self.rate ** self.shape) / gamma(self.shape)
        part2 = target ** (-self.shape - 1)
        part3 = torch.exp(-self.rate / target)

        return torch.log(part1 * part2 * part3)

    def update(self, shape, rate):
        """
        Updates shape and rate of the distribution

        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        """
        self.shape = shape
        self.rate = rate


class HorseshoeLayer(nn.Module):
    """
    Single linear layer of a horseshoe prior for regression
    """

    def __init__(self, in_features, out_features):
        """
        Args:
            in_features: int, number of input features
            out_features: int, number of output features
            parameters: instance of class HorseshoeHyperparameters
            device: cuda device instance
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Scale to initialize weights, according to Yingzhen's work

        scale = 1. * np.sqrt(6. / (in_features + out_features))

        # Initialization of parameters of prior distribution
        # weight parameters
        self.prior_tau_shape = torch.Tensor([0.5])

        # local shrinkage parameters
        self.prior_lambda_shape = torch.Tensor([0.5])
        self.prior_lambda_rate = torch.Tensor([1 / weight_cauchy_scale ** 2])

        # global shrinkage parameters
        self.prior_v_shape = torch.Tensor([0.5])
        self.prior_theta_shape = torch.Tensor([0.5])
        self.prior_theta_rate = torch.Tensor([1 / global_cauchy_scale ** 2])

        # Initialization of parameters of variational distribution
        # weight parameters
        self.beta_mean = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-scale, scale))
        self.beta_rho = nn.Parameter(torch.ones([out_features, in_features]) * beta_rho_scale)
        self.beta = ReparametrizedGaussian(self.beta_mean, self.beta_rho)

        # local shrinkage parameters
        self.lambda_shape = self.prior_lambda_shape * torch.ones(in_features)
        self.lambda_rate = self.prior_lambda_rate * torch.ones(in_features)
        self.lambda_ = InverseGamma(self.lambda_shape, self.lambda_rate)

        # Sample from half-Cauchy to initialize the mean of log_tau
        # We initialize the parameters using a half-Cauchy because this
        # is the prior distribution over tau

        distr = HalfCauchy(1 / np.sqrt(self.prior_lambda_rate))
        sample = distr.sample(torch.Size([in_features])).squeeze()
        self.log_tau_mean = nn.Parameter(torch.log(sample))

        self.log_tau_rho = nn.Parameter(torch.ones(in_features) * log_tau_rho_scale)
        self.log_tau = ReparametrizedGaussian(self.log_tau_mean, self.log_tau_rho)

        # bias parameters
        self.bias_mean = nn.Parameter(torch.zeros([1, out_features], ))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * bias_rho_scale)
        self.bias = ReparametrizedGaussian(self.bias_mean, self.bias_rho)

        # global shrinkage parameters
        self.theta_shape = self.prior_theta_shape
        self.theta_rate = self.prior_theta_rate
        self.theta = InverseGamma(self.theta_shape, self.theta_rate)

        # Sample from half-Cauchy to initialize the mean of log_v
        # We initialize the parameters using a half-Cauchy because this
        # is the prior distribution ovev

        distr = HalfCauchy(1 / np.sqrt(self.prior_theta_rate))
        sample = distr.sample()
        self.log_v_mean = nn.Parameter(torch.log(sample))

        self.log_v_rho = nn.Parameter(torch.ones([1, 1]) * log_v_rho_scale)
        self.log_v = ReparametrizedGaussian(self.log_v_mean, self.log_v_rho)

        '''
        #regular Horseshoe prior
        self.log_c_mean = nn.Parameter(torch.ones([1]))
        self.log_c_rho = nn.Parameter(torch.ones([1, 1]) * log_v_rho_scale)
        self.log_c = ReparametrizedGaussian(self.log_c_mean, self.log_c_rho)

        '''
        self.log_variational_posterior = 0
        self.log_prior = 0

    def log_prior(self):
        """
        Computes the expectation of the log of the prior p under the variational posterior q
        """

        def exp_log_inverse_gamma(shape, exp_rate, exp_log_rate, exp_log_x, exp_x_inverse):
            """
            Calculates the expectation of the log of an inverse gamma distribution p under
            the posterior distribution q
            E_q[log p(x | shape, rate)]


            Args:
            shape: float, the shape parameter of the gamma distribution
            exp_rate: torch tensor, the expectation of the rate parameter under q
            exp_log_rate: torch tensor, the expectation of the log of the rate parameter under q
            exp_log_x: torch tensor, the expectation of the log of the random variable under q
            exp_x_inverse: torch tensor, the expectation of the inverse of the random variable under q

            Returns:
            exp_log: torch tensor, E_q[log p(x | shape, rate)]
            """
            exp_log = - torch.lgamma(shape) + shape * exp_log_rate - (shape + 1) * exp_log_x \
                      - exp_rate * exp_x_inverse

            # We need to sum over all components since this is a vectorized implementation.
            # That is, we compute the sum over the individual expected values. For example,
            # in the horseshoe BLR model we have one local shrinkage parameter for each weight
            # and therefore one expected value for each of these shrinkage parameters.
            return torch.sum(exp_log)

        def exp_log_gaussian(mean, std):
            """
            Calculates the expectation of the log of a Gaussian distribution p under the posterior distribution q
            E_q[log p(x)] - see note log_prior_gaussian.pdf

            Args:
            mean: torch tensor, the mean of the posterior distribution
            std: torch tensor, the standard deviation of the posterior distribution

            Returns:
            exp_gaus: torch tensor, E_q[p(x)]


            Comment about how this function is vectorized:
            Every component beta_i follows a univariate Gaussian distribution, and therefore has
            a scalar mean and a scalar variance. We can combine all components of beta into a
            diagonal Gaussian distribution, which has a mean vector of the same length as the
            beta vector, and a standard deviation vector of the same length. By summing over the
            mean vector and over the standard deviations, we therefore sum over all components of beta.
            """
            dim = mean.shape[0] * mean.shape[1]
            exp_gaus = - 0.5 * dim * (torch.log(torch.tensor(2 * math.pi))) - 0.5 * (
                        torch.sum(mean ** 2) + torch.sum(std ** 2))
            return exp_gaus

        # Calculate E_q[ln p(\tau | \lambda)] + E[ln p(\lambda)]
        # E_q[ln p(\tau | \lambda)] for the weights
        shape = self.prior_tau_shape
        exp_lambda_inverse = self.lambda_.exp_inverse()
        exp_log_lambda = self.lambda_.exp_log()
        exp_log_tau = self.log_tau.mean
        exp_tau_inverse = torch.exp(-self.log_tau.mean + 0.5 * self.log_tau.std_dev ** 2)
        log_inv_gammas_weight = exp_log_inverse_gamma(shape, exp_lambda_inverse, -exp_log_lambda,
                                                      exp_log_tau, exp_tau_inverse)

        # E_q[ln p(\lambda)] for the weights
        shape = self.prior_lambda_shape
        rate = self.prior_lambda_rate
        log_inv_gammas_weight += exp_log_inverse_gamma(shape, rate, np.log(rate),
                                                       exp_log_lambda, exp_lambda_inverse)

        '''
        #Regularized Horseshoe priors
        exp_log_c = self.log_c.mean
        exp_c_inverse = torch.exp(-self.log_c.mean + 0.5 * self.log_c.std_dev ** 2)
        shape = self.prior_c_rate
        rate = self.prior_c_rate
        log_inv_gammas_c = exp_log_inverse_gamma(shape, rate, np.log(rate),
                                                       exp_log_c, exp_c_inverse)
        '''

        # E_q[ln p(v | \theta)] for the global shrinkage parameter
        shape = self.prior_v_shape
        exp_theta_inverse = self.theta.exp_inverse()
        exp_log_theta = self.theta.exp_log()
        exp_log_v = self.log_v.mean
        exp_v_inverse = torch.exp(-self.log_v.mean + 0.5 * self.log_v.std_dev ** 2)
        log_inv_gammas_global = exp_log_inverse_gamma(shape, exp_theta_inverse, -exp_log_theta,
                                                      exp_log_v, exp_v_inverse)

        # E_q[ln p(\theta)] for the global shrinkage parameter
        shape = self.prior_theta_shape
        rate = self.prior_theta_rate
        log_inv_gammas_global += exp_log_inverse_gamma(shape, rate, np.log(rate),
                                                       exp_log_theta, exp_theta_inverse)

        # Add all expectations
        log_inv_gammas = log_inv_gammas_weight + log_inv_gammas_global
        '''
        #Regularized Horseshoe priors
        log_inv_gammas = log_inv_gammas_weight + log_inv_gammas_global + log_inv_gammas_c
        '''

        # E_q[N(beta)]
        log_gaussian = exp_log_gaussian(self.beta.mean, self.beta.std_dev) \
                       + exp_log_gaussian(self.bias.mean, self.bias.std_dev)

        return log_gaussian + log_inv_gammas

    def log_variational_posterior(self):
        """
        Computes the log of the variational posterior by computing the entropy.

        The entropy is defined as -integral[q(theta) log(q(theta))]. The log of the
        variational posterior is given by integral[q(theta) log(q(theta))].
        Therefore, we compute the entropy and return -entropy.

        Tau and v follow log-Normal distributions. The entropy of a log normal
        is the entropy of the normal distribution + the mean.
        """
        entropy = self.beta.entropy() \
                  + self.log_tau.entropy() + torch.sum(self.log_tau.mean) \
                  + self.lambda_.entropy() + self.bias.entropy() \
                  + self.log_v.entropy() + torch.sum(self.log_v.mean) \
                  + self.theta.entropy()
        # + self.log_c.entropy() + torch.sum(self.log_c.mean)  #Regularized Horseshoe prior

        if sum(torch.isnan(entropy)).item() != 0:
            raise Exception("entropy/log_variational_posterior computation ran into nan!")
            print('self.beta.entropy(): ', self.beta.entropy())
            print('beta mean: ', self.beta.mean)
            print('beta std: ', self.beta.std_dev)

        return -entropy

    def forward(self, input, take_sample=True):
        """
        Performs a forward pass through the layer, that is, computes
        the layer output for a given input batch.

        Args:
            input_: torch Tensor, input data to forward through the net
            sample: bool, whether to samples weights and bias
            n_samples: int, number of samples to draw from the weight and bias distribution
        """
        beta = self.beta.sample()
        log_tau = self.log_tau.sample()
        log_v = self.log_v.sample()

        '''
        #Regularized horseshoe
        log_c = self.log_c.sample()
        log_tau_hat = (log_c * log_tau ** 2) / (log_c + log_tau ** 2 * log_v ** 2)
        log_tau_hat = torch.sqrt(log_tau_hat)

        weight = beta * log_tau_hat * log_v
        '''

        weight = beta * log_tau * log_v

        bias = self.bias.sample()

        return F.linear(input, weight, bias)

    def analytic_update(self):
        """
        Calculates analytic updates of lambda_ and theta

        Lambda and theta follow inverse Gamma distributions and can be updated
        analytically. The update equations are given in the paper in equation 9
        of the appendix: bayesiandeeplearning.org/2017/papers/42.pdf
        """
        new_shape = torch.Tensor([1])
        # new lambda rate is given by E[1/tau_i] + 1/b_0^2
        new_lambda_rate = torch.exp(-self.log_tau.mean + 0.5 * (self.log_tau.std_dev ** 2)) \
                          + self.prior_lambda_rate

        # new theta rate is given by E[1/v] + 1/b_g^2
        new_theta_rate = torch.exp(-self.log_v.mean + 0.5 * (self.log_v.std_dev ** 2)) \
                         + self.prior_theta_rate

        self.lambda_.update(new_shape, new_lambda_rate)
        self.theta.update(new_shape, new_theta_rate)

class RegHorseshoeLayer(nn.Module):
    """
    Single linear layer of a horseshoe prior for regression
    """
    def __init__(self, in_features, out_features):
        """
        Args:
            in_features: int, number of input features
            out_features: int, number of output features
            parameters: instance of class HorseshoeHyperparameters
            device: cuda device instance
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Scale to initialize weights, according to Yingzhen's work

        scale = 1. * np.sqrt(6. / (in_features + out_features))


        # Initialization of parameters of prior distribution
        # weight parameters
        self.prior_tau_shape = torch.Tensor([0.5])

        # local shrinkage parameters
        self.prior_lambda_shape = torch.Tensor([0.5])
        self.prior_lambda_rate = torch.Tensor([1 / weight_cauchy_scale**2])

        # global shrinkage parameters
        self.prior_v_shape = torch.Tensor([0.5])
        self.prior_theta_shape = torch.Tensor([0.5])
        self.prior_theta_rate = torch.Tensor([1 / global_cauchy_scale**2])

        # Initialization of parameters of variational distribution
        # weight parameters
        self.beta_mean = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-scale, scale))
        self.beta_rho = nn.Parameter(torch.ones([out_features, in_features]) * beta_rho_scale)
        self.beta = ReparametrizedGaussian(self.beta_mean, self.beta_rho)

        # local shrinkage parameters
        self.lambda_shape = self.prior_lambda_shape * torch.ones(in_features)
        self.lambda_rate = self.prior_lambda_rate * torch.ones(in_features)
        self.lambda_ = InverseGamma(self.lambda_shape, self.lambda_rate)

        # Sample from half-Cauchy to initialize the mean of log_tau
        # We initialize the parameters using a half-Cauchy because this
        # is the prior distribution over tau

        distr = HalfCauchy(1 / np.sqrt(self.prior_lambda_rate))
        sample = distr.sample(torch.Size([in_features])).squeeze()
        self.log_tau_mean = nn.Parameter(torch.log(sample))

        self.log_tau_rho = nn.Parameter(torch.ones(in_features) * log_tau_rho_scale)
        self.log_tau = ReparametrizedGaussian(self.log_tau_mean, self.log_tau_rho)

        # bias parameters
        self.bias_mean = nn.Parameter(torch.zeros([1, out_features], ))
        self.bias_rho = nn.Parameter(torch.ones([1, out_features]) * bias_rho_scale)
        self.bias = ReparametrizedGaussian(self.bias_mean, self.bias_rho)

        # global shrinkage parameters
        self.theta_shape = self.prior_theta_shape
        self.theta_rate = self.prior_theta_rate
        self.theta = InverseGamma(self.theta_shape, self.theta_rate)

        # Sample from half-Cauchy to initialize the mean of log_v
        # We initialize the parameters using a half-Cauchy because this
        # is the prior distribution ovev

        distr = HalfCauchy(1 / np.sqrt(self.prior_theta_rate))
        sample = distr.sample()
        self.log_v_mean = nn.Parameter(torch.log(sample))

        self.log_v_rho = nn.Parameter(torch.ones([1, 1])  * log_v_rho_scale)
        self.log_v = ReparametrizedGaussian(self.log_v_mean, self.log_v_rho)


        #regular Horseshoe prior
        self.log_c_mean = nn.Parameter(torch.ones([1]))
        self.log_c_rho = nn.Parameter(torch.ones([1, 1]) * log_v_rho_scale)
        self.log_c = ReparametrizedGaussian(self.log_c_mean, self.log_c_rho)
        

        self.log_prior = 0
        self.log_variational_posterior = 0

    def log_prior(self):
        """
        Computes the expectation of the log of the prior p under the variational posterior q
        """
        def exp_log_inverse_gamma(shape, exp_rate, exp_log_rate, exp_log_x, exp_x_inverse):
            """
            Calculates the expectation of the log of an inverse gamma distribution p under
            the posterior distribution q
            E_q[log p(x | shape, rate)]


            Args:
            shape: float, the shape parameter of the gamma distribution
            exp_rate: torch tensor, the expectation of the rate parameter under q
            exp_log_rate: torch tensor, the expectation of the log of the rate parameter under q
            exp_log_x: torch tensor, the expectation of the log of the random variable under q
            exp_x_inverse: torch tensor, the expectation of the inverse of the random variable under q

            Returns:
            exp_log: torch tensor, E_q[log p(x | shape, rate)]
            """
            exp_log = - torch.lgamma(shape) + shape * exp_log_rate - (shape + 1) * exp_log_x\
                      -exp_rate * exp_x_inverse

            # We need to sum over all components since this is a vectorized implementation.
            # That is, we compute the sum over the individual expected values. For example,
            # in the horseshoe BLR model we have one local shrinkage parameter for each weight
            # and therefore one expected value for each of these shrinkage parameters.
            return torch.sum(exp_log)

        def exp_log_gaussian(mean, std):
            """
            Calculates the expectation of the log of a Gaussian distribution p under the posterior distribution q
            E_q[log p(x)] - see note log_prior_gaussian.pdf

            Args:
            mean: torch tensor, the mean of the posterior distribution
            std: torch tensor, the standard deviation of the posterior distribution

            Returns:
            exp_gaus: torch tensor, E_q[p(x)]


            Comment about how this function is vectorized:
            Every component beta_i follows a univariate Gaussian distribution, and therefore has
            a scalar mean and a scalar variance. We can combine all components of beta into a
            diagonal Gaussian distribution, which has a mean vector of the same length as the
            beta vector, and a standard deviation vector of the same length. By summing over the
            mean vector and over the standard deviations, we therefore sum over all components of beta.
            """
            dim = mean.shape[0] * mean.shape[1]
            exp_gaus = - 0.5 * dim * (torch.log(torch.tensor(2 * math.pi))) - 0.5 * (torch.sum(mean **2) + torch.sum(std**2))
            return exp_gaus

        # Calculate E_q[ln p(\tau | \lambda)] + E[ln p(\lambda)]
        # E_q[ln p(\tau | \lambda)] for the weights
        shape = self.prior_tau_shape
        exp_lambda_inverse = self.lambda_.exp_inverse()
        exp_log_lambda = self.lambda_.exp_log()
        exp_log_tau = self.log_tau.mean
        exp_tau_inverse = torch.exp(-self.log_tau.mean + 0.5 * self.log_tau.std_dev **2)
        log_inv_gammas_weight = exp_log_inverse_gamma(shape, exp_lambda_inverse, -exp_log_lambda,
                                exp_log_tau, exp_tau_inverse)

        # E_q[ln p(\lambda)] for the weights
        shape = self.prior_lambda_shape
        rate = self.prior_lambda_rate
        log_inv_gammas_weight += exp_log_inverse_gamma(shape, rate, np.log(rate),
                                                       exp_log_lambda, exp_lambda_inverse)


        #Regularized Horseshoe priors
        exp_log_c = self.log_c.mean
        exp_c_inverse = torch.exp(-self.log_c.mean + 0.5 * self.log_c.std_dev ** 2)
        shape = self.prior_c_rate
        rate = self.prior_c_rate
        log_inv_gammas_c = exp_log_inverse_gamma(shape, rate, np.log(rate),
                                                       exp_log_c, exp_c_inverse)


        # E_q[ln p(v | \theta)] for the global shrinkage parameter
        shape = self.prior_v_shape
        exp_theta_inverse = self.theta.exp_inverse()
        exp_log_theta = self.theta.exp_log()
        exp_log_v = self.log_v.mean
        exp_v_inverse = torch.exp(-self.log_v.mean + 0.5 * self.log_v.std_dev **2)
        log_inv_gammas_global = exp_log_inverse_gamma(shape, exp_theta_inverse, -exp_log_theta,
                                exp_log_v, exp_v_inverse)

        # E_q[ln p(\theta)] for the global shrinkage parameter
        shape = self.prior_theta_shape
        rate = self.prior_theta_rate
        log_inv_gammas_global += exp_log_inverse_gamma(shape, rate, np.log(rate),
                                                       exp_log_theta, exp_theta_inverse)

        # Add all expectations
      #  log_inv_gammas = log_inv_gammas_weight + log_inv_gammas_global

        #Regularized Horseshoe priors
        log_inv_gammas = log_inv_gammas_weight + log_inv_gammas_global + log_inv_gammas_c


        # E_q[N(beta)]
        log_gaussian = exp_log_gaussian(self.beta.mean, self.beta.std_dev)\
                       + exp_log_gaussian(self.bias.mean, self.bias.std_dev)

        return log_gaussian + log_inv_gammas

    def log_variational_posterior(self):
        """
        Computes the log of the variational posterior by computing the entropy.

        The entropy is defined as -integral[q(theta) log(q(theta))]. The log of the
        variational posterior is given by integral[q(theta) log(q(theta))].
        Therefore, we compute the entropy and return -entropy.

        Tau and v follow log-Normal distributions. The entropy of a log normal
        is the entropy of the normal distribution + the mean.
        """
        entropy = self.beta.entropy()\
                + self.log_tau.entropy() + torch.sum(self.log_tau.mean)\
                + self.lambda_.entropy() + self.bias.entropy()\
                + self.log_v.entropy() + torch.sum(self.log_v.mean)\
                + self.theta.entropy()\
                + self.log_c.entropy() + torch.sum(self.log_c.mean)  #Regularized Horseshoe prior



        if sum(torch.isnan(entropy)).item() != 0:
            raise Exception("entropy/log_variational_posterior computation ran into nan!")
            print('self.beta.entropy(): ', self.beta.entropy())
            print('beta mean: ', self.beta.mean)
            print('beta std: ', self.beta.std_dev)

        return -entropy


    def forward(self, input, sample=True):
        """
        Performs a forward pass through the layer, that is, computes
        the layer output for a given input batch.

        Args:
            input_: torch Tensor, input data to forward through the net
            sample: bool, whether to samples weights and bias
            n_samples: int, number of samples to draw from the weight and bias distribution
        """
        beta = self.beta.sample()
        log_tau = self.log_tau.sample()
        log_v = self.log_v.sample()


        #Regularized horseshoe
        log_c = self.log_c.sample()
        log_tau_hat = (log_c * log_tau ** 2) / (log_c + log_tau ** 2 * log_v ** 2)
        log_tau_hat = torch.sqrt(log_tau_hat)

        weight = beta * log_tau_hat * log_v


       # weight = beta * log_tau * log_v

        bias = self.bias.sample()

        return F.linear(input, weight, bias)


    def analytic_update(self):
        """
        Calculates analytic updates of lambda_ and theta

        Lambda and theta follow inverse Gamma distributions and can be updated
        analytically. The update equations are given in the paper in equation 9
        of the appendix: bayesiandeeplearning.org/2017/papers/42.pdf
        """
        new_shape = torch.Tensor([1])
        # new lambda rate is given by E[1/tau_i] + 1/b_0^2
        new_lambda_rate = torch.exp(-self.log_tau.mean + 0.5 * (self.log_tau.std_dev**2)) \
                          + self.prior_lambda_rate

        # new theta rate is given by E[1/v] + 1/b_g^2
        new_theta_rate = torch.exp(-self.log_v.mean + 0.5 * (self.log_v.std_dev**2)) \
                         + self.prior_theta_rate

        self.lambda_.update(new_shape, new_lambda_rate)
        self.theta.update(new_shape, new_theta_rate)



class ReparameterizedGaussian:
    def __init__(self, mus, rhos):
        self.mus = mus
        self.rhos = rhos
        self.shape = self.mus.size()

    def _get_sigmas(self):
        return torch.log(1 + torch.exp(self.rhos))

    def sample(self):
        epsilon = dists.Normal(0, 1).sample(self.shape).to(DEVICE)

        sample = self.mus + self._get_sigmas() * epsilon
        return sample

    def get_variational_posterior(self, sample):
        pdf_likelihood = dists.Normal(self.mus, self._get_sigmas()).log_prob(sample)
        return pdf_likelihood.sum()


class ScaleMixturePrior:
    def __init__(self, sigma1, sigma2, pi):
        self.gaussian1 = dists.Normal(0, sigma1)
        self.gaussian2 = dists.Normal(0, sigma2)
        self.pi = pi

        if sigma1 < sigma2:
            print('Warning: sigma1 < sigma2 in the ScaleMixturePrior!')

    def get_prior(self, weights):
        likelihood1 = torch.exp(self.gaussian1.log_prob(weights))
        likelihood2 = torch.exp(self.gaussian2.log_prob(weights))

        prior_matrix = self.pi * likelihood1 + (1.0 - self.pi) * likelihood2
        return torch.log(prior_matrix).sum()


class BayesianLayer(nn.Module):
    def __init__(self, in_size, out_size,
                 mu_range=(-0.25, 0.25), rho_range=(-5, -4)):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size

        # Weights
        self.weight_mus = nn.Parameter(torch.Tensor(out_size, in_size).uniform_(*mu_range))
        self.weight_rhos = nn.Parameter(torch.Tensor(out_size, in_size).uniform_(*rho_range))
        self.weight_prior = ScaleMixturePrior(SIGMA1, SIGMA2, PI)
        self.weight = ReparameterizedGaussian(self.weight_mus, self.weight_rhos)

        # Biases
        self.bias_mus = nn.Parameter(torch.Tensor(out_size).uniform_(*mu_range))
        self.bias_rhos = nn.Parameter(torch.Tensor(out_size).uniform_(*rho_range))
        self.bias_prior = ScaleMixturePrior(SIGMA1, SIGMA2, PI)
        self.bias = ReparameterizedGaussian(self.bias_mus, self.bias_rhos)

        self.log_variational_posterior = 0
        self.log_prior = 0

    def forward(self, x, take_sample=True):
        if take_sample or self.training:
            # Take weights/biases as samples from gaussians
            weights = self.weight.sample()
            biases = self.bias.sample()

            self.log_variational_posterior = \
                self.weight.get_variational_posterior(weights) + \
                self.bias.get_variational_posterior(biases)

            self.log_prior = \
                self.weight_prior.get_prior(weights) + \
                self.bias_prior.get_prior(biases)

        else:
            weights = self.weight.mus
            biases = self.bias.mus

            self.log_variational_posterior = 0
            self.log_prior = 0

        return F.linear(x, weights, bias=biases)






class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA1, SIGMA2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA1, SIGMA2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, take_sample=False, calculate_log_probs=False):
        if self.training or take_sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


