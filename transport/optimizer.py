#!/usr/bin/env python
"""
Main Optimization Algorithm

"""

# Import Modules
import numpy as np
from transport.sgd import Options, sgd
from transport.transport_map import TransportMap

# Code Implementation

class TransportMapOptimizer(object):
    """ Driver for optimizing/learning a Transport Map

    Args:
      target_model (Model) - target density object
            methods: log_joint_prob and grad_log_joint_prob methods
      lambduh (double) - regularization on gamma (default = 0)
      transport_map (TransportMap) - transport map object
            methods: map, jacobian, grad_map, grad_jacobian, set_gamma
        or
      **kwargs - keyword=argument pairs for TransportMap

    Attributes:
      N (int) - dimension of density
      P_s (N ndarray) - dimension of gamma parameters in tranport map
      P (int) - total dimension of all gamma parameters

    Methods:
      oracle - func of gamma, batch_size, returns noisy KL objective + gradient
      oracle_svrg - func of gamma, gamma_tilde, batch_size, return gradient_diff
      learn_map - func of gamma_0, Options, runs stochastic gradient descent
      sample - return samples from Transport Mapped Reference
      random_gamma, set_gamma, get_gamma
    """
    def __init__(self, target_model, transport_map=None, lambduh=0.0, **kwargs):
        if transport_map is None:
            transport_map = TransportMap(**kwargs)
        self.target_model = target_model
        self.transport_map = transport_map
        self.lambduh = lambduh

        self.N = self.target_model.N
        if self.N != transport_map.N:
            raise ValueError("Dimension N in target_model and tranport_map must match")

        self.P_s = np.array([np.size(gamma_n)
            for gamma_n in self.transport_map.gammas])
        self.P = np.sum(self.P_s)

        return

    def oracle(self, gamma, batch_size):
        """ Return noisy objective and gradient
        Args:
          gamma (P ndarray) - current stacked tranport_map parameter
          batch_size (int) - batch_size
        Returns:
          noisy_objective (double) - noisy estimate of objective
          noisy_gradient (P ndarray) - noisy estimate of gradient
        """

        self.set_gamma(gamma)

        noisy_objective_sum = 0.0
        noisy_gradient_sum = np.zeros(self.P)

        for j in xrange(0, batch_size):
            sample_j = self.transport_map.reference_measure(self.N)
            objective_j, gradient_j = self._calculate_oracle_one(sample_j)

            noisy_objective_sum += objective_j
            noisy_gradient_sum += gradient_j

        noisy_objective = noisy_objective_sum / batch_size
        noisy_gradient = np.hstack(noisy_gradient_sum) / batch_size

        # Lambduh L2 Regularization
        noisy_objective += 0.5*self.lambduh * gamma.dot(gamma)
        noisy_gradient += self.lambduh * gamma

        return noisy_objective, noisy_gradient

    def _calculate_oracle_one(self, sample):
        """ Calculate Noisy KL and Gradient for sample
        Args:
          sample (N ndarray) - sample from reference measure

        Returns:
          objective (double) - objective based on sample
          gradient (P ndarray) - gradient based on sample
        """
        self.transport_map.fit_sample(sample)

        # Chain Rule Goodness
        mapped_sample = self.transport_map.mapping()
        jacobian_sample = self.transport_map.jacobian()

        objective = -1.0 * self.target_model.log_joint_prob(mapped_sample)
        objective += -1.0 * np.sum(np.log(np.abs(jacobian_sample)))

        gradient = [np.zeros(self.P_s[n]) for n in xrange(0,self.N)]
        grad_model = self.target_model.grad_log_joint_prob(mapped_sample)
        for n in xrange(0, self.N):
            gradient[n] += -1.0 * grad_model[n] * (
                    self.transport_map.grad_mapping(n))
            gradient[n] += -1.0*(jacobian_sample[n] ** -1) * (
                    self.transport_map.grad_jacobian(n))
        gradient = np.hstack(gradient)

        return objective, gradient

    def oracle_svrg(self, gamma, gamma_tilde, batch_size):
        """ Return noisy objective and gradient
        Args:
          gamma (P ndarray) - current stacked tranport_map parameter
          gamma_tilde (P ndarray) - svrg stacked tranport_map parameter
          batch_size (int) - batch_size
        Returns:
          noisy_objective (double) - noisy estimate of objective
          noisy_gradient_diff (P ndarray)-noisy estimate of gradient difference
        """
        noisy_objective_sum = 0.0
        noisy_gradient_sum = np.zeros(self.P)

        samples = [self.transport_map.reference_measure(self.N)
                for j in xrange(0, batch_size)]

        self.set_gamma(gamma_tilde)
        for j, sample_j in enumerate(samples):
            _, gradient_tilde_j = self._calculate_oracle_one(sample_j)
            noisy_gradient_sum -= gradient_tilde_j

        self.set_gamma(gamma)
        for j, sample_j in enumerate(samples):
            objective_j, gradient_j = self._calculate_oracle_one(sample_j)
            noisy_objective_sum += objective_j
            noisy_gradient_sum += gradient_j

        noisy_objective = noisy_objective_sum / batch_size
        noisy_gradient_diff = np.hstack(noisy_gradient_sum) / batch_size

        # Lambduh L2 Regularization
        noisy_objective += 0.5*self.lambduh * gamma.dot(gamma)
        noisy_gradient_diff += self.lambduh * (gamma - gamma_tilde)

        return noisy_objective, noisy_gradient_diff

    def random_gamma(self):
        """ Randomly set gamma in transport_map """
        self.transport_map.random_gammas()
        return self.get_gamma()


    def set_gamma(self, gamma):
        """ Set gamma in transport_map
        Args:
          gamma (P ndarray) - current stacked transport_map parameter
        """
        start = 0
        for n in xrange(0, self.N):
            self.transport_map.set_gamma(n, gamma[start:start+self.P_s[n]])
            start += self.P_s[n]
        return

    def get_gamma(self):
        """ Return gamma from transport_map """
        return np.hstack(self.transport_map.gammas)

    def learn_map(self, gamma_0=None, options=None, **kwargs):
        """ Run SGD to learn gamma
        Args:
          gamma_0 (P ndarray) - initial parameter estimate
          options (sgd.Options) - Options for SGD
            or
          **kwargs for sgd.Options

        Returns:
          gamma_star (P ndarray) - optimal/final parameter estimate
          gamma_s (T by P ndarray) - gamma estimates per iteration
          f_s (T ndarray) - noisy objective estimates per iteration
        """
        if gamma_0 is not None:
            self.set_gamma(gamma_0)
        if options is None:
            options=Options(**kwargs)

        gamma_star, gamma_s, f_s, eta_s = sgd(self, self.get_gamma(), options)
        self.set_gamma(gamma_star)
        return gamma_star, gamma_s, f_s, eta_s

    def sample(self, number_samples):
        """ Wrapper for transport_map.sample(number_samples) """
        return self.transport_map.sample(number_samples)


# Code To Execute
if __name__ == '__main__':
    print "optimizer.py"





#EOF
