#!/usr/bin/env python
"""
Transport Map Optimizer Test Script

"""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transport.target_models import LinearRegression, generate_linear_regression
from transport.transport_map import TransportMap
from transport.optimizer import TransportMapOptimizer

np.random.seed(1234)

# Code To Execute
maximum_degree = 1
number_observations = 100
theta_true = np.array([-1.0, 2.0, -3.0])
number_samples = 1000

A = np.zeros((number_observations, len(theta_true)))
A[:,0] = 1.0
A[:,1:] = (np.random.randn(number_observations, len(theta_true)-1)*0.5 +
        np.outer(np.random.randn(number_observations),
            np.ones(len(theta_true)-1)))
b = A.dot(theta_true) + np.random.randn(number_observations)

target_model = LinearRegression(A=A, b=b, mu_theta=theta_true)

sample_results = {}
sample_results['exact'] = target_model.posterior_sample(1000)

for degree in xrange(0,maximum_degree):
    print "Order %u of %u " % (degree+1, maximum_degree)
    transport_map = TransportMap(N=np.size(theta_true), K=degree+1)
    transport_map.random_gammas()
    my_driver = TransportMapOptimizer(
            target_model=target_model,
            transport_map=transport_map)
    _,_,f = my_driver.learn_map(batch_size=10, eta_0=0.1, T=10000,
            stepsize_update="ada", alpha=0.01)
    sample_results['order_'+str(degree+1)] = my_driver.sample(1000)



#EOF
