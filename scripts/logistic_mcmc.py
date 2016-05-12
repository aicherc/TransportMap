import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pystan


# Stan Code
logistic_code = """
data {
    int<lower=0> N;
    int<lower=0> D;
    matrix[D,N] x;
    int<lower=0, upper=1> y[D];
    vector[N] mu;
    matrix[N,N] Sigma;
}
parameters {
    vector[N] theta;
}
model {
    theta ~ multi_normal(mu, Sigma);
    y ~ bernoulli_logit(x*theta);
}
"""

theta = np.array([1, -1, 1, -1])
N = np.size(theta)
D = 100
x = np.random.randn(D, N)
y = np.random.rand(D) < x.dot(theta)
y = [int(y_i) for y_i in y]
mu_theta = theta
sigma2_theta = np.ones(N)

logistic_data = {
        "N": N,
        "D": D,
        "x": x,
        "y": y,
        "mu": mu_theta,
        "Sigma": np.diag(sigma2_theta),
        }
fit = pystan.stan(model_code=logistic_code, data=logistic_data,
        iter=1000, chains=4)

theta = fit.extract("theta")["theta"]
sns.jointplot(theta[:,0], theta[:,1], kind="kde")

