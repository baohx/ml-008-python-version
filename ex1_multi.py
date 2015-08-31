# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:22:14 2015

@author: ba0hx
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data2.txt', delimiter = ',')

X = data[:, 0:2]

y = data[:, 2]

y = y.reshape((-1, 1))

m = y.shape[0]

def featureNormalize(X):
    X_norm = X
    mu = np.mean(X_norm, axis = 0)
    mu = mu.reshape((1, -1))
    X_norm = X_norm - mu
    sigma = np.std(X_norm, axis = 0)
    sigma = sigma.reshape((1, -1))
    X_norm = X_norm / sigma    
    return X_norm, mu, sigma

X, mu, sigma = featureNormalize(X)

X = np.hstack( (np.ones( (m, 1) ), X) )

alpha = 0.01

num_iters = 400

theta = np.zeros((3, 1))

def computeCostMulti(X, y, theta):
    m = len(y)
    J = np.sum((np.dot(X, theta)-y)**2 / m / 2.0)
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - alpha * np.dot(X.T, np.dot(X, theta) - y) / m
        J_history[iter] = computeCostMulti(X, y, theta)
    return theta, J_history

theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

plt.figure()

plt.plot(J_history)

plt.show()

print theta[0, 0], theta[1, 0], theta[2, 0]

price = np.dot(np.array([[1, 
                          (1650 - mu[0, 0] ) / sigma[0, 0], 
                          (3 - mu[0, 1])/ sigma[0, 1]]]), 
               theta)[0, 0]

print price

data = np.loadtxt('ex1data2.txt', delimiter = ',')

X = data[:, 0:2]

y = data[:, 2]

y = y.reshape((-1, 1))

m = y.shape[0]

X = np.hstack( (np.ones( (m, 1) ), X) )

def normalEqn(X, y):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return theta

theta = normalEqn(X, y)

print theta[0, 0], theta[1, 0], theta[2, 0]

price = np.dot(np.array([[1, 1650, 3]]), theta)[0, 0]

print price