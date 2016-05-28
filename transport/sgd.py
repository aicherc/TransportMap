#!/usr/bin/env python
"""
Stochastic Gradient Descent Algorithm

"""

# Import Modules
import numpy as np

# Code Implementation

class Options(object):
    """ Options for SGD
    Args:
      T (int) - number of sgd iterations (default 100)
      eta_0 (double or N ndarray) - stepsize for gradient steps (default 1.0)
      batch_size (int) - number of samples for stochastic gradient (default 1)
      stepsize_update (string) - (default `fixed`)
        `fixed` - fixed stepsize eta_0
        `decay` - decaying stepsize eta_0/(1+eta_0*k*strong_convexity)
        `ada` - AdaGrad stepsize
        `svrg` - stochastic variance reduced gradient (SVRG)
        `bb` - Barzilai-Borwein stepsize
      G (N ndarray) - vector for gradients (default initialized to zeros)
        AdaGrad -> squared gradients
        SVRG -> averaged gradient
        BB -> decaying sum of gradients
      update_freq (int) - freq to update avg gradient (default 100)
        Used by SVRG and BB
      beta_decay (double) - decay of avg gradients in BB (default 0.5)
      alpha (double) - fraction of last states to average over (default 0.1)

    Returns:
      options (Options) - options object
    """
    def __init__(self, **kwargs):
        options = {
                "T": 100,
                "batch_size": 1,
                "eta_0": 1.0,
                "stepsize_update": "fixed",
                "G": 0.0,
                "update_freq": 100,
                "beta_decay": 0.5,
                "alpha": 0.1,
                }
        for key, value in kwargs.iteritems():
            if key in options.keys():
                options[key] = value
            else:
                print "Ignoring unrecognized kwarg: %s" % key

        for key, value in options.iteritems():
            setattr(self, key, value)

        self._check_options()
        return

    def _check_options(self):
        if type(self.T) is not int:
            raise TypeError("T must be an int")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if type(self.batch_size) is not int:
            raise TypeError("batch_size must be an int")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if type(self.update_freq) is not int:
            raise TypeError("update_freq must be an int")
        if self.update_freq <= 0:
            raise ValueError("update_freq must be positive")
        if self.eta_0 <= 0:
            raise ValueError("eta_0 must be positive")
        if (self.beta_decay <= 0) or (self.beta_decay >= 1.0):
            raise ValueError("beta_decay must be in (0,1)")
        if (self.alpha <= 0) or (self.alpha > 1):
            raise ValueError("alpha must be in (0,1]")
        return


def sgd(f, x_0, options):
    """ Stochastic Gradient Descent
    Args:
      f (objective) - has oracle function (objective + gradient)
      x_0 (N ndarray) - initial point
      options (Options) - other options (see Options)
    Returns:
      x_star (N ndarray) - final solution
      x_s (T by N ndarray) - list of solution points
      f_s (T ndarray) - list of (noisy) objective values
      eta_s (T ndarray) - list of stepsizes
    """
    N = np.size(x_0)
    x_s = np.zeros((options.T, N))
    f_s = np.zeros((options.T))
    eta_s = np.zeros((options.T))

    # Initialize x and stepsize
    x = x_0
    eta = options.eta_0

    # Main Loop
    for t in xrange(0, options.T):
        if ((t+1) % 1000) == 0:
            print "Iteration %u of %u" % (t, options.T)

        # Calculate Gradient
        samples = f.sample(options.batch_size)

        # Update stepsize
        if options.stepsize_update == "fixed":
            fx, gx = f.oracle(x, options.batch_size)
            eta = options.eta_0

        elif options.stepsize_update == "decay":
            fx, gx = f.oracle(x, options.batch_size)
            eta = options.eta_0 / (1.0 + options.eta_0 * f.lambduh * t)

        elif options.stepsize_update == "ada":
            fx, gx = f.oracle(x, options.batch_size)
            options.G += (gx ** 2)
            eta = options.eta_0 / np.sqrt(options.G)

        elif options.stepsize_update == "bb":
            if t % options.update_freq == 0:
                if (t == 0) or (t == options.update_freq):
                    # Do nothing the first two epochs
                    eta = options.eta_0
                    G_prev = options.G
                    x_prev = np.mean(x_s[t-options.update_freq:t], axis=0)
                else:
                    x_new = np.mean(x_s[t-options.update_freq:t], axis=0)
                    inner_prod = np.dot(x-x_prev, options.G-G_prev)
                    if np.abs(inner_prod) > 1e-10:
                        eta = (np.dot(x-x_prev, x-x_prev) /
                                (inner_prod*options.update_freq))
                    G_prev = options.G
                    x_prev = x_new
            fx, gx = f.oracle(x, options.batch_size)
            options.G = ( options.beta_decay*gx +
                         (1.0-options.beta_decay)*options.G )

        elif options.stepsize_update == "svrg":
            if t % options.update_freq == 0:
                x_tilde = x
                _, G_tilde = f.oracle(x_tilde,
                        batch_size=options.batch_size * options.update_freq)
                options.G = G_tilde/options.update_freq
            fx, g_diff = f.oracle_svrg(x, x_tilde, options.batch_size)
            gx = g_diff + options.G
            eta = options.eta_0
        else:
            raise ValueError("Unrecognized stepsize update")

        # Update x
        x = x - eta * gx

        # Record History
        x_s[t] = x
        f_s[t] = fx
        eta_s[t] = np.linalg.norm(eta)/np.size(eta)

    # Calculate final solution
    T_frac = int(options.alpha * options.T)
    x_star = np.mean(x_s[-T_frac:,], axis=0)
    return x_star, x_s, f_s, eta_s


# Code To Execute
if __name__ == '__main__':
    print "sgd.py"





#EOF
