#!/usr/bin/env python
"""
Transport Map Optimizer Test Script

"""

# Import Modules
import numpy as np
from scipy.stats import ks_2samp
from transport.sgd import Options
from transport.transport_map import TransportMap
from transport.optimizer import TransportMapOptimizer

# Code Implementation
def test_marginal_ks(samples1, samples2, axis=0):
    """ Run KS Test on two samples along marginal axis
    Args:
      samples1 (number_samples by D ndarray)
      samples2 (number_samples by D ndarray)
      axis (int) axis to use (default 0)
    Returns:
      ks_statistic (float) test statistic
      pvalue (float) two-sided
    """
    return ks_2samp(samples1[:,axis], samples2[:,axis])

def test_stepsize_updates(transport_optimizer, stepsize_updates,
        sgd_options=None, gamma_0=None, number_samples=1000):
    """ Test different stepsize_updates
    Args:
      transport_optimizer (TransportMapOptimizer)
      stepsize_updates (list of strings) - stepsize update
      sgd_options (sgd.Options) - options for learn_map()
      gamma_0 (N ndarray) - initial guess
      number_samples (int) - default 1000

    Returns:
      lists of learn_map output
      gamma_star (list of N ndarrays)
      gamma_s (list of T by N ndarrays)
      f_s (list of T ndarray)
      eta_s (list of T ndarray)
    """
    if sgd_options is None:
        sgd_options = Options()
    if gamma_0 is None:
        gamma_0 = np.copy(transport_optimizer.random_gamma())

    G_0 = sgd_options.G+0
    eta_0 = sgd_options.eta_0+0

    gamma_star = [None]*len(stepsize_updates)
    gamma_s = [None]*len(stepsize_updates)
    f_s = [None]*len(stepsize_updates)
    eta_s= [None]*len(stepsize_updates)
    samples= [None]*len(stepsize_updates)

    for ii, stepsize_update in enumerate(stepsize_updates):
        sgd_options.stepsize_update = stepsize_update
        sgd_options.G = G_0+0
        sgd_options.eta_0 = eta_0+0
        if stepsize_update == "ada":
            sgd_options.eta_0 = eta_0*100

        out = transport_optimizer.learn_map(gamma_0, sgd_options)
        gamma_star[ii] = out[0]
        gamma_s[ii] = out[1]
        f_s[ii] = out[2]
        eta_s[ii] = out[3]
        samples[ii] = transport_optimizer.sample(number_samples)

    return gamma_star, gamma_s, f_s, eta_s, samples

def test_transport_maps(target_model, transport_maps, sgd_options=None,
        number_samples=1000, lambduh = 0.0):
    """ Test different transport_maps
    Args:
      target_model (Model) - target density object
      transport_maps (list of TransportMap) - transport map objects
      sgd_options (sgd.Options) - options for learn_map()
      number_samples (int) - default 1000
      lambduh (double) - regularization on gamma (default = 0)

    Returns:
      gamma_star (list of N ndarrays)
      gamma_s (list of T by N ndarrays)
      f_s (list of T ndarray)
      eta_s (list of T ndarray)
      samples (list of number_samples
    """
    if sgd_options is None:
        sgd_options = Options()

    gamma_star = [None]*len(transport_maps)
    gamma_s = [None]*len(transport_maps)
    f_s = [None]*len(transport_maps)
    eta_s= [None]*len(transport_maps)
    samples= [None]*len(transport_maps)

    for ii, transport_map in enumerate(transport_maps):
        transport_optimizer = TransportMapOptimizer(
                target_model = target_model,
                transport_map = transport_map,
                lambduh = lambduh)
        transport_optimizer.random_gamma()
        sgd_options.G = 0.0

        out = transport_optimizer.learn_map(options=sgd_options)
        gamma_star[ii] = out[0]
        gamma_s[ii] = out[1]
        f_s[ii] = out[2]
        eta_s[ii] = out[3]
        samples[ii] = transport_optimizer.sample(number_samples)

    return gamma_star, gamma_s, f_s, eta_s, samples


# Code To Execute
if __name__ == '__main__':
    print "_utils.py"





#EOF
