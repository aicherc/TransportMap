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
      stepsize_update (string) - `fixed`, `decay`, `ada` (default `fixed`)
      G (N ndarray) - vector for AdaGrad squared gradients (default 0.0)
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
                "alpha": 0.1,
                }
        for key, value in kwargs.iteritems():
            if key in options.keys():
                options[key] = value
            else:
                print "Ignoring unrecognized kwarg: %s" % key

        self.T = options["T"]
        self.batch_size = options["batch_size"]
        self.eta_0 = options["eta_0"]
        self.stepsize_update = options["stepsize_update"]
        self.G = options["G"]
        self.alpha = options["alpha"]

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
        if self.eta_0 <= 0:
            raise ValueError("eta_0 must be positive")
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
    """
    N = np.size(x_0)
    x_s = np.zeros((options.T, N))
    f_s = np.zeros((options.T))

    # Initialize x and stepsize
    x = x_0
    eta = options.eta_0

    # Main Loop
    for t in xrange(0, options.T):
        if ((t+1) % 1000) == 0:
            print "Iteration %u of %u" % (t, options.T)

        # Calculate Gradient
        fx, gx = f.oracle(x, options.batch_size)

        # Update stepsize
        if options.stepsize_update == "fixed":
            eta = options.eta_0
        elif options.stepsize_update == "decay":
            eta = options.eta_0 / (1.0 + options.eta_0 * f.lambduh * t)
        elif options.stepsize_update == "ada":
            options.G += (gx ** 2)
            eta = options.eta_0 / np.sqrt(options.G)
        else:
            raise ValueError("Unrecognized stepsize update")

        # Update x
        x = x - eta * gx

        # Record History
        x_s[t] = x
        f_s[t] = fx

    # Calculate final solution
    T_frac = int(options.alpha * options.T)
    x_star = np.mean(x_s[-T_frac:,], axis=0)
    return x_star, x_s, f_s


# Code To Execute
if __name__ == '__main__':
    print "sgd.py"





#EOF
