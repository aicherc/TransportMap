# Optimal Transport for Sampling

Christopher Aicher

EE 546 Project - Spring 2016


## Background
See Marzouk, Youseef, et al. _An introduction to sampling via measure transport_. 2016

## Code Layout
* optimizer.py - main optimization algorithm
* sgd.py - Stochastic Gradient Descent (Adagrad + Decaying)
* transport_map.py - transport map
* basis.py - lower-triangular basis functions (Hermite)
* target_models.py - Bayesian Linear + Logistic Regression (approximation)
