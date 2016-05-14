#!/usr/bin/env python
"""
Transport Map Optimizer Test Script

"""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from transport.target_models import (
        LogisticRegression,
        generate_logistic_regression,
        )
from transport.transport_map import TransportMap
from transport.optimizer import TransportMapOptimizer

np.random.seed(1234)

# Code To Execute
maximum_degree = 5
number_observations = 100
theta_true = np.array([-1.0, 2.0, -3.0])

A,b = generate_logistic_regression(theta_true=theta_true, D=number_observations)
target_model = LogisticRegression(A=A, b=b, mu_theta=theta_true)

transport_map = TransportMap(N=np.size(theta_true), K=maximum_degree)
transport_map.random_gammas()

my_driver = TransportMapOptimizer(
        target_model=target_model,
        transport_map=transport_map)

#my_driver.learn_map(batch_size=10, T=1000, eta_0=0.00001)



#EOF
