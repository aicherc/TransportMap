#!/usr/bin/env python
"""
Transport Map Tester

"""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from transport.transport_map import TransportMap

# Code Implementation

# Code To Execute
if __name__ == '__main__':
    print "sgd_test.py"

    N = 2
    K = 3
    my_map = TransportMap(N, K)

    z0 = np.zeros(N)
    z1 = np.ones(N)
    z2 = np.array([0,1])
    z3 = np.array([1,0])

    my_map.update_gamma(0, np.array([1, 1, 1, 0]))
    my_map.update_gamma(1, np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))

    my_map.fit_sample(z0)
    print my_map.mapping()
    print my_map.jacobian()
    my_map.fit_sample(z1)
    print my_map.mapping()
    print my_map.jacobian()
    my_map.fit_sample(z2)
    print my_map.mapping()
    print my_map.jacobian()
    my_map.fit_sample(z3)
    print my_map.mapping()
    print my_map.jacobian()




#EOF
