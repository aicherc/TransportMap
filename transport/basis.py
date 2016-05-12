#!/usr/bin/env python
"""
Basis Functions for transport_map.py

"""

# Import Modules
import numpy as np
from scipy.special import eval_hermitenorm

# Code Implementation
class Hermite(object):
    """ Hermite Polynomial univariate basis functions

    H_k = (-1)^k e^{x^2/2} d_x^k e^{-x^2/2}

    Methods:
      fun (x, order) - returns H_order(x)
      grad(x, order) - returns d_x H_order(x)
    """
    def __init__(self):
        return

    def fun(self, x, order):
        """ Return the hermite polynomial evaluated at x
        Args:
          x (double) - evaulation point
          order (int) - order of polynomial
        Returns:
          H_x (double)
        """
        return eval_hermitenorm(order, x)

    def grad(self, x, order):
        """ Return the first derivate of hermite polynomial evaluated at x
        Args:
          x (double) - evaulation point
          order (int) - order of polynomial
        Returns:
          H_x (double)
        """
        return order * eval_hermitenorm(order-1, x)


# Code To Execute
if __name__ == '__main__':
    print "basis.py"





#EOF
