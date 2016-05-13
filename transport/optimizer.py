#!/usr/bin/env python
"""
Main Optimization Algorithm

"""

# Import Modules
import numpy as np
from transport.sgd import Options, sgd

# Code Implementation

class TransportMapOptimizer(object):
    """ Driver for optimizing/learning a Transport Map

    Args:
      target_model (Model) - target density object
            methods: log_joint_prob and grad_log_joint_prob methods
      transport_map (TransportMap) - transport map object
            methods: map, jacobian, grad_map, grad_jacobian, set_gamma
      lambduh (double) - regularization on gamma (default = 0)

    Attributes:
      N (int) - dimension of density
      P_s (N ndarray) - dimension of gamma parameters in tranport map
      P (int) - total dimension of all gamma parameters

    Methods:
      oracle - func of gamma, batch_size, returns noisy KL objective + gradient
      learn_map - func of gamma_0, Options, runs stochastic gradient descent
    """
    def __init__(self, target_model, transport_map, lambduh=0.0):
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
        noisy_gradient_sums = [np.zeros(self.P_s[n]) for n in xrange(0,self.N)]

        for j in xrange(0, batch_size):
            sample_j = self.transport_map.reference_measure(self.N)
            self.transport_map.fit_sample(sample_j)

            # Chain Rule Goodness
            mapped_sample_j = self.transport_map.mapping()
            jacobian_sample_j = self.transport_map.jacobian()

            noisy_objective_sum += -1.0 * self.target_model.log_joint_prob(
                    mapped_sample_j)
            noisy_objective_sum += -1.0 * np.sum(np.log(
                np.abs(jacobian_sample_j)))

            grad_model = self.target_model.grad_log_joint_prob(mapped_sample_j)
            for n in xrange(0, self.N):
                noisy_gradient_sums[n] += -1.0 * grad_model[n] * (
                        self.transport_map.grad_mapping(n))
                noisy_gradient_sums[n] += -1.0*(jacobian_sample_j[n] ** -1) * (
                        self.transport_map.grad_jacobian(n))

        noisy_objective = noisy_objective_sum / batch_size
        noisy_gradient = np.hstack(noisy_gradient_sums) / batch_size

        return noisy_objective, noisy_gradient

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
        """ Run SGD to learn gamma """
        if gamma_0 is not None:
            self.set_gamma(gamma_0)
        if options is None:
            options=Options(**kwargs)

        gamma_star, gamma_s, f_s = sgd(self, self.get_gamma(), options)
        self.set_gamma(gamma_star)
        return gamma_star, gamma_s, f_s

    def sample(self, number_samples):
        """ Wrapper for transport_map.sample(number_samples) """
        return self.transport_map.sample(number_samples)


# Code To Execute
if __name__ == '__main__':
    print "optimizer.py"





#EOF
