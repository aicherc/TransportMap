#!/usr/bin/env python
"""
Transport Map Optimizer Test Script

"""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from transport.sgd import Options
from transport.target_models import GumbelDensity
from transport.transport_map import TransportMap
from transport.optimizer import TransportMapOptimizer
from transport._utils import test_transport_maps, test_stepsize_updates
from transport._plot_util import kde_plot

np.random.seed(12345)

# Code To Execute
sgd_options = Options(T=1000, eta_0=1, stepsize_update="ada", batch_size=10)

maximum_degree = 3
mu_true = np.array([-2.0])
beta_true = np.array([1.0])

number_samples = 5000

target_model = GumbelDensity(mu_true, beta_true)

exact_samples = target_model.posterior_sample(number_samples)

def truncated_gaussian(shape):
    x = np.random.randn(shape)
    x[x > 3.0] = 3.0
    x[x < -3.0] = -3.0
    return x

transport_maps = [TransportMap(N=np.size(mu_true), K=k,
    reference_measure=truncated_gaussian)
        for k in xrange(1,maximum_degree+1)]

# -> AdaGrad is the way to go
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
        transport_maps, sgd_options, lambduh=0.0001,
        number_samples=number_samples)

kde_plot(exact_samples, 0)
for d, sample in enumerate(samples):
    kde_plot(sample, 0, label=str(d+1))
plt.legend(loc="best")


#EOF
