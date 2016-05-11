#!/usr/bin/env python
"""
Stochastic Gradient Descent Algorithm Tester

"""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from transport.sgd import Options, sgd

# Code Implementation
class NoisyF(object):
    def __init__(self, x_star, lambduh):
        self.x_star = x_star
        self.lambduh = lambduh

    def oracle(self, x, batch_size):
        noise = np.random.randn(np.size(x)) * 1.0/batch_size
        resid = x - self.x_star + noise
        fx = 0.5 * self.lambduh * resid.dot(resid)
        gx = self.lambduh * resid
        return fx, gx


# Code To Execute
if __name__ == '__main__':
    print "sgd_test.py"

    f_1 = NoisyF(x_star=np.array([1,2,3]), lambduh = 2.0)
    options = Options(eta_0 = 1.0, stepsize_update="ada")

    x_0 = np.array([0,0,0])

    x_star, x_s, f_s = sgd(f_1, x_0, options)
    plt.plot(x_s[:,0], x_s[:,1])





#EOF
