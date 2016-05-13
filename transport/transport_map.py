#!/usr/bin/env python
"""
Transport Map

"""

# Import Modules
import numpy as np
import itertools
from basis import Hermite

# Code Implementation
class TransportMap(object):
    """ Transport Map Object
    Args:
      N (int) - dimension of input/output space
      K (int) - maximum degree of polynomial
      basis (object) - has eval, grad, hess functions of x, order (default Hermite)
      reference_measure (func) - draw samples  (default np.random.randn)
      perm_matrix (N ndarray) - permutation matrix (default identity)

    Attributes:
      H_z (N by K+1 ndarray) - univariate basis evaluated (dimension, order)
      dH_z (N by K+1 ndarray) = gradient of basis evaluated (dimension, order)
      gammas (N list of P_n ndarrays) - coefficients in the transport map
      J_s (N list of P_n by n ndarrays) - multi-index map for basis products

    Methods:
      fit_sample - func of z (N ndarray)
      map - func of n, returns T_n (double)
      jacobian - func of n, returns dT_n/dx_n (double)
      grad_map - func of n, returns grad_gamma T_n (P_n ndarray)
      grad_jacobian - func of n, returns grad_gamma dT_n/dx_n (P_n ndarray)
      update_gamma - func of n, delta_gamma (P_n ndarray)
      set_gamma - func of n, new_gamma (P_n ndarray)
      random_gamma - randomly initialize gammas
      sample - func number_samples, returns samples from transport density

    """
    def __init__(self, N, K, basis=None, reference_measure=None,
            perm_matrix=None):
        if basis is None:
            basis = Hermite()
        if reference_measure is None:
            reference_measure = np.random.randn

        self.N = N
        self.K = K
        self.basis = basis
        self.reference_measure = reference_measure

        self.H_z = np.zeros((self.N, self.K+1))
        self.dH_z = np.zeros((self.N, self.K+1))
        self.J_s = [build_multi_index(n+1, self.K) for n in xrange(0, self.N)]
        self.gammas = [np.zeros(len(J_n)) for J_n in self.J_s]

    def fit_sample(self, z):
        """ Precompute basis values for sample z
        Args:
          z (N ndarray) - sample from reference measure
        """
        for n in xrange(0, self.N):
            for order in xrange(0, self.K+1):
                self.H_z[n, order] = self.basis.fun(z[n], order)
                self.dH_z[n, order] = self.basis.grad(z[n], order)

    def _check_n(self, n):
        """ Check if n is valid """
        if type(n) is not int:
            raise TypeError("n must be an int")
        if n >= self.N:
            raise ValueError("n must be less than self.N")
        return

    def mapping(self, n=None, z=None):
        """ Return the n-th element of T for the fitted sample
        Args:
          n (int) - dimension of map (if None will return all dimensions)
          z (N ndarray) - sample (default: use sample from fit_sample)
        Returns:
          T_n (double) - mapping value at dimension n (or an N ndarray of all values)
        """
        if z is not None:
            self.fit_sample(z)
        if n is None:
            return np.array([self.mapping(n) for n in xrange(0, self.N)])
        T_n = 0.0
        for i, index in enumerate(self.J_s[n]):
            prod = self.gammas[n][i]
            for j in xrange(0, n+1):
                prod *= self.H_z[j,index[j]]
            T_n += prod
        return T_n

    def jacobian(self, n=None, z=None):
        """ Return the n-th element of dT/dx_n for the fitted sample
        Args:
          n (int) - dimension of map (if None will return all dimensions)
          z (N ndarray) - sample (default: use sample from fit_sample)
        Returns:
          dT_n (double) - derivative at dimension n (or an N ndarray of all values)
        """
        if z is not None:
            self.fit_sample(z)
        if n is None:
            return np.array([self.jacobian(n) for n in xrange(0, self.N)])
        dT_n = 0.0
        for i, index in enumerate(self.J_s[n]):
            prod = self.dH_z[n, index[n]] * self.gammas[n][i]
            for j in xrange(0, n):
                prod *= self.H_z[j, index[j]]
            dT_n += prod
        return dT_n

    def grad_mapping(self, n, z=None):
        """ Return the gradient of n-th element of T for the fitted sample
        Args:
          n (int) - dimension of map
          z (N ndarray) - sample (default: use sample from fit_sample)
        Returns:
          grad_T_n (P_n ndarray) - gradient of mapping value at dimension n
        """
        if z is not None:
            self.fit_sample(z)
        grad_T_n = np.zeros(len(self.gammas[n]))
        for i, index in enumerate(self.J_s[n]):
            prod = 1.0
            for j in xrange(0, n+1):
                prod *= self.H_z[j,index[j]]
            grad_T_n[i] += prod
        return grad_T_n


    def grad_jacobian(self, n, z=None):
        """ Return the gradient of n-th element of dT/dx_n for the fitted sample
        Args:
          n (int) - dimension of map
          z (N ndarray) - sample (default: use sample from fit_sample)
        Returns:
          grad_dT_n (P_n ndarray) - gradient of derivative value at dimension n
        """
        if z is not None:
            self.fit_sample(z)
        grad_dT_n = np.zeros(len(self.gammas[n]))
        for i, index in enumerate(self.J_s[n]):
            prod = self.dH_z[n, index[n]]
            for j in xrange(0, n):
                prod *= self.H_z[j, index[j]]
            grad_dT_n[i] += prod
        return grad_dT_n

    def update_gamma(self, n, delta_gamma):
        """ Update gamma_n by adding delta_gamma """
        self._check_n(n)
        if np.size(delta_gamma) !=  np.size(self.gammas[n]):
            raise ValueError("delta_gamma is wrong size")
        self.gammas[n] += delta_gamma
        return

    def set_gamma(self, n, new_gamma):
        """ Set gamma_n to new_gamma """
        self._check_n(n)
        if np.size(new_gamma) !=  np.size(self.gammas[n]):
            raise ValueError("new_gamma is wrong size")
        self.gammas[n] = new_gamma
        return


    def random_gammas(self):
        """ Randomly initialize gammas """
        for n in xrange(0, self.N):
            self.gammas[n] = np.random.randn(len(self.gammas[n]))
        return

    def sample(self, number_samples):
        """ Return samples from the transport density """
        samples = np.zeros((number_samples, self.N))
        for i in xrange(0, number_samples):
            z = self.reference_measure(self.N)
            samples[i,:] = self.mapping(z=z)

        return samples



def build_multi_index(number_variables, max_degree):
    """ Build the multi-index J for all product of univariate polynomials
    over x_1, ..., x_number_variables with total degree at most max_degree

    This function is exponential in number_variables (should be rarely called)

    e.g. max_degree = 2, number_variables = 3
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
     [0, 1, 1], [2, 0, 0], [0, 2, 0], [0, 0 ,2]]

    Args:
      number_variables (int) - number of variables
      max_degree (int) - maximum total degree of polynomial

    Returns:
      J (P by number_variables ndarray) - index with degree
    """
    J = []
    degree_range = [xrange(0, max_degree+1)]*number_variables
    for index in itertools.product(*degree_range):
        if np.sum(index) <= max_degree:
            J.append(np.array(index))
    return J


# Code To Execute
if __name__ == '__main__':
    print "transport_map.py"





#EOF
