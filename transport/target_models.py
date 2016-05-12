#!/usr/bin/env python
"""
Target Density Functions

"""

# Import Modules
import numpy as np

# Code Implementation
class LogisticRegression(object):
    """ Logistic Regression Target Density Object

    Logistic Function is expit

    Args:
      A (D by N ndarray) - covariates
      b (D ndarray) - binary observations in {-1, +1}
      mu_theta (N ndarray) - prior mean for theta (default 0)
      sigma2_theta (N ndarray) - prior (diagonal) variance for theta (default 1)

    Methods:
      log_joint_prob - func of theta (N ndarray) returns log_prob (double)
      grad_log_joint_prob - func of theta (N ndarray) returns grad (N ndarray)
      posterior_sample - func of number_samples (int) returns samples
    """
    def __init__(self, A, b, mu_theta=None, sigma2_theta=None):
        raise NotImplementedError()

    def log_joint_prob(self, theta):
        raise NotImplementedError()

    def grad_log_joint_prob(self, theta):
        raise NotImplementedError()

    def posterior_sample(self, number_samples):
        """ Approximately sample from the posterior using MCMC + Stan """

        raise NotImplementedError()


class LinearRegression(object):
    """ Linear Regression Target Density Object

    Args:
      A (D by N ndarray) - covariates
      b (D ndarray) - observations
      mu_theta (N ndarray) - prior mean for theta (default 0)
      sigma2_theta (N ndarray) - prior (diagonal) variance for theta (default 1)

    Methods:
      log_joint_prob - func of theta (N ndarray) returns log_prob (double)
      grad_log_joint_prob - func of theta (N ndarray) returns grad (N ndarray)
      posterior_sample - func of number_samples (int) returns samples
    """
    def __init__(self, A, b, mu_theta=None, sigma2_theta=None):
        raise NotImplementedError()

    def log_joint_prob(self, theta):
        raise NotImplementedError()

    def grad_log_joint_prob(self, theta):
        raise NotImplementedError()

    def posterior_sample(self, number_samples):
        raise NotImplementedError()

def logistic_regression_mcmc(model, number_samples):

    D, N = np.shape(model.A)
    y = (model.b+1)/2 # Convert to {0,1}

    # Stan Code
    logistic_code = """
    data {
        int<lower=0> N;
        int<lower=0> D;
        matrix[D,N] x;
        int<lower=0, upper=1> y[D];
        vector[N] mu;
        matrix[N,N] Sigma;
    }
    parameters {
        vector[N] theta;
    }
    model {
        theta ~ multi_normal(mu, Sigma);
        y ~ bernoulli_logit(x*theta);
    }
    """

    logistic_data = {
            "N": N,
            "D": D,
            "x": model.A,
            "y": y,
            "mu": model.mu_theta,
            "Sigma": np.diag(model.sigma2_theta),
            }
    fit = pystan.stan(model_code=logistic_code, data=logistic_data,
            iter=1000+number_samples, chains=4)

    theta = fit.extract("theta")["theta"]
    samples = theta[1000:,:]
    return samples





# Code To Execute
if __name__ == '__main__':
    print "target_models.py"





#EOF
