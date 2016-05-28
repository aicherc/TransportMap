#!/usr/bin/env python
"""
Hermite Polynomial Plot

"""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transport.basis import Hermite


# Code Implementation

# Code To Execute
if __name__ == '__main__':
    print "hermite_polynomial_plot.py"

    hermite = Hermite()
    x = np.linspace(-3, 3)
    def plot_order(order):
        y = hermite.fun(x, order)
        plt.plot(x, y, label="Order {0}".format(order))

    for order in xrange(0, 5):
        plot_order(order)

    plt.legend(loc="best")

    plt.savefig('figs/hermite_poly.png')
    plt.savefig('figs/hermite_poly.pdf')





#EOF
