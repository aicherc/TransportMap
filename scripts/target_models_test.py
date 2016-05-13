#!/usr/bin/env python
"""
Stochastic Gradient Descent Algorithm Tester

"""

# Import Modules
import numpy as np
import matplotlib.pyplot as plt
from transport.target_models import LinearRegression, generate_linear_regression
from transport.target_models import LogisticRegression, generate_logistic_regression

# Code Implementation

# Code To Execute
if __name__ == '__main__':
    print "target_model_test.py"
    theta_true = np.array([-2, 2, -5])

    A, b = generate_linear_regression(theta_true)
    my_model = LinearRegression(A=A, b=b, sigma2_obs=1.0, mu_theta=theta_true)
#    samples = my_model.posterior_sample(1000)

    A2, b2 = generate_logistic_regression(theta_true)
    my_model2 = LogisticRegression(A=A2, b=b2, mu_theta=theta_true)





#EOF
