#!/usr/bin/env python
"""
Target Density Functions

"""

# Import Modules
import numpy as np
import pystan
from scipy.special import expit

# Code Implementation
class LogisticRegression(object):
    """ Logistic Regression Target Density Object

    Logistic Function is expit

    Args:
      A (D by N ndarray) - covariates
      b (D ndarray) - binary observations in {-1, +1}
      mu_theta (N ndarray) - prior mean for theta (default 0)
      sigma2_theta (N ndarray) - prior (diag) variance for theta (default 10)

    Methods:
      log_joint_prob - func of theta (N ndarray) returns log_prob (double)
      grad_log_joint_prob - func of theta (N ndarray) returns grad (N ndarray)
      posterior_sample - func of number_samples (int) returns samples
    """
    def __init__(self, A, b, mu_theta=None, sigma2_theta=None):
        D, N = np.shape(A)
        if mu_theta is None:
            mu_theta = np.zeros(N)
        if sigma2_theta is None:
            sigma2_theta = np.ones(N)*10.0

        self.N = N
        self.D = D
        self.A = A
        self.b = b
        self.mu_theta = mu_theta
        self.sigma2_theta = sigma2_theta
        return

    def log_joint_prob(self, theta):
        """ Return the log joint probability of the parameter
        Args:
          theta (N ndarray) - coefficient parameter
        Returns:
          log_joint_prob (double) - log prior + log likelihood
        """
        prior_resid = theta-self.mu_theta
        log_joint_prob = -0.5 * (prior_resid/self.sigma2_theta).dot(prior_resid)
        obs_resid = self.A.dot(theta) * self.b
        log_joint_prob += np.sum(np.log(expit(obs_resid)))
        return log_joint_prob

    def grad_log_joint_prob(self, theta):
        """ Return the log joint probability of the parameter
        Args:
          theta (N ndarray) - coefficient parameter
        Returns:
          grad_log_joint_prob (N ndarray) - log prior + log likelihood
        """
        prior_resid = theta-self.mu_theta
        grad_log_joint_prob = - prior_resid / self.sigma2_theta
        obs_resid = self.A.dot(theta) * self.b
        grad_log_joint_prob += (expit(-obs_resid) * self.b).dot(self.A)
        return grad_log_joint_prob

    def posterior_sample(self, number_samples):
        """ Approximately sample from the posterior using MCMC + Stan
        Args:
          number_samples (int)
        Returns:
          samples (number_samples by N ndarray) - posterior samples
        """
        samples = logistic_regression_mcmc(self, number_samples)
        return samples


class LinearRegression(object):
    """ Linear Regression Target Density Object

    Args:
      A (D by N ndarray) - covariates
      b (D ndarray) - observations
      sigma2_obs (double) - variance of observations (default 1.0)
      mu_theta (N ndarray) - prior mean for theta (default 0)
      sigma2_theta (N ndarray) - prior (diag) variance for theta (default 10)

    Methods:
      log_joint_prob - func of theta (N ndarray) returns log_prob (double)
      grad_log_joint_prob - func of theta (N ndarray) returns grad (N ndarray)
      posterior_sample - func of number_samples (int) returns samples
    """
    def __init__(self, A, b, sigma2_obs=1.0, mu_theta=None, sigma2_theta=None):
        D, N = np.shape(A)
        if mu_theta is None:
            mu_theta = np.zeros(N)
        if sigma2_theta is None:
            sigma2_theta = np.ones(N)*10.0

        self.N = N
        self.D = D
        self.A = A
        self.b = b
        self.sigma2_obs = sigma2_obs
        self.mu_theta = mu_theta
        self.sigma2_theta = sigma2_theta
        return

    def log_joint_prob(self, theta):
        """ Return the log joint probability of the parameter
        Args:
          theta (N ndarray) - coefficient parameter
        Returns:
          log_joint_prob (double) - log prior + log likelihood
        """
        prior_resid = theta-self.mu_theta
        log_joint_prob = -0.5 * (prior_resid/self.sigma2_theta).dot(prior_resid)
        obs_resid = self.A.dot(theta) - self.b
        log_joint_prob += -0.5 * obs_resid.T.dot(obs_resid) / self.sigma2_obs
        return log_joint_prob

    def grad_log_joint_prob(self, theta):
        """ Return the log joint probability of the parameter
        Args:
          theta (N ndarray) - coefficient parameter
        Returns:
          grad_log_joint_prob (N ndarray) - log prior + log likelihood
        """
        prior_resid = theta-self.mu_theta
        grad_log_joint_prob = - prior_resid / self.sigma2_theta
        obs_resid = self.A.dot(theta) - self.b
        grad_log_joint_prob += - obs_resid.dot(self.A) / self.sigma2_obs
        return grad_log_joint_prob

    def posterior_sample(self, number_samples):
        """ Sample from the posterior
        Args:
          number_samples (int)
        Returns:
          samples (number_samples by N ndarray) - posterior samples of theta
        """
        posterior_Sigma = np.linalg.inv(
                np.diag(self.sigma2_theta**-1) + self.A.T.dot(self.A))
        posterior_mu = posterior_Sigma.dot(
                self.mu_theta/self.sigma2_theta +self.A.T.dot(self.b))

        samples = np.array([
            np.random.multivariate_normal(posterior_mu, posterior_Sigma)
            for _ in xrange(0, number_samples)])

        return samples


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
            "y": y.astype(int),
            "mu": model.mu_theta,
            "Sigma": np.diag(model.sigma2_theta),
            }
    frac_samples = np.ceil(number_samples/4.0)
    fit = pystan.stan(model_code=logistic_code, data=logistic_data,
            iter=1000+frac_samples, chains=4)

    theta = fit.extract("theta")["theta"]
    samples = theta[0:number_samples,:]
    return samples

def generate_linear_regression(theta_true, obs_variance=1.0, D=100):
    """ Generate Data for linear regression model
    Args:
      theta_true (N ndarray) - true coefficient vector
      obs_variance (double) - observation noise level (default 1.0)
      D (int) - number of observations (default 100)
    Returns:
      A (D by N ndarray) - covariates/features
      b (D ndarray) - response/targets
    """
    A = np.random.randn(D, np.size(theta_true))*2.0
    A[:,0] = np.ones(D)
    b = A.dot(theta_true) + np.random.randn(D)*np.sqrt(obs_variance)
    return A, b

def generate_logistic_regression(theta_true, obs_variance=0.0, D=100):
    """ Generate Data for logistic regression model
    Args:
      theta_true (N ndarray) - true coefficient vector
      obs_variance (double) - observation noise level (default 0.0)
      D (int) - number of observations (default 100)
    Returns:
      A (D by N ndarray) - covariates/features
      b (D ndarray) - response/targets
    """
    A = np.random.randn(D, np.size(theta_true))*2.0
    A[:,0] = np.ones(D)
    eta = A.dot(theta_true) + np.random.randn(D)*np.sqrt(obs_variance)
    p = expit(eta)
    b = (np.random.rand(D) < p) * 2.0 - 1.0 # Map to +/-1
    return A, b



# Code To Execute
if __name__ == '__main__':
    print "target_models.py"





#EOF
