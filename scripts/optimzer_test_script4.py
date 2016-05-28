#!/usr/bin/env python
"""
Transport Map Optimizer Test Script

"""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from transport.sgd import Options
from transport.target_models import LogisticRegression, expit
from transport.transport_map import TransportMap
from transport.optimizer import TransportMapOptimizer
from transport._utils import test_transport_maps, test_stepsize_updates

np.random.seed(1234)

# Code To Execute
sgd_options = Options(T=1000, eta_0=0.01, stepsize_update="ada")
maximum_degree = 1

number_observations = 100
theta_true = np.array([-1.0, 2.0, -3.0])
number_samples = 2000

A = np.zeros((number_observations, len(theta_true)))
A[:,0] = 1.0
A[:,1:] = (np.random.randn(number_observations, len(theta_true)-1)*0.1 +
        np.outer(np.random.randn(number_observations),
            np.ones(len(theta_true)-1)))
b = (np.random.rand(number_observations) < expit(A.dot(theta_true)))*2-1

target_model = LogisticRegression(A=A, b=b, mu_theta=theta_true)

transport_maps = [TransportMap(N=np.size(theta_true), K=k)
        for k in xrange(1,maximum_degree+1)]

# -> AdaGrad is the way to go
transport_map = transport_maps[0]
transport_optimizer = TransportMapOptimizer(target_model=target_model,
        lambduh=0.001,
        transport_map=transport_map)


#stepsize_updates = ['fixed', 'decay', 'ada', 'svrg', 'bb']
stepsize_updates = ['ada', 'svrg']
gamma_star, gamma_s, f_s, eta_s, samples = test_stepsize_updates(
        transport_optimizer,
        stepsize_updates=stepsize_updates,
        sgd_options=sgd_options)


#gamma_star, gamma_s, f_s, eta_s, samples = test_transport_maps(target_model,
#        transport_maps, sgd_options)




#EOF
