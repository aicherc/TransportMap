#!/usr/bin/env python
"""
Transport Map Optimizer Test Script

"""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from transport.sgd import Options
from transport.target_models import LaplaceDensity
from transport.transport_map import TransportMap
from transport.optimizer import TransportMapOptimizer
from transport._utils import test_transport_maps, test_stepsize_updates

np.random.seed(1234)

# Code To Execute
maximum_degree = 1
theta_true = np.array([-1.0, 2.0, -3.0])
A = np.eye(np.size(theta_true))
sgd_options = Options(T=5000, eta_0=1, stepsize_update="ada")

target_model = LaplaceDensity(A=A,
        b=theta_true)

transport_maps = [TransportMap(N=np.size(theta_true), K=k)
        for k in xrange(1,maximum_degree+1)]

## -> AdaGrad is the way to go
#transport_map = transport_maps[0]
#transport_optimizer = TransportMapOptimizer(target_model=target_model,
#        lambduh=0.001,
#        transport_map=transport_map)
#
#
#stepsize_updates = ['fixed', 'decay', 'ada', 'svrg', 'bb']
#gamma_star, gamma_s, f_s, eta_s, samples = test_stepsize_updates(
#        transport_optimizer,
#        stepsize_updates=stepsize_updates,
#        sgd_options=sgd_options)


gamma_star, gamma_s, f_s, eta_s, samples = test_transport_maps(target_model,
        transport_maps, sgd_options)




#EOF
