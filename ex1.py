# -*- coding: utf-8 -*-
"""
Created on Sun Mar 08 14:28:18 2015

@author: ba0hx
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

A = np.eye(5)

print A

data = np.loadtxt('ex1data1.txt', delimiter = ',')

X = data[:, 0]

X = X.reshape((-1, 1))

y= data[:, 1]

y = y.reshape((-1, 1))

m = y.shape[0]

plt.figure()

plt.plot(X, y, 'xr')

plt.xlabel('Population of City in 10,000s')

plt.ylabel('Profit in $10,000s')

X = np.hstack( (np.ones( (m, 1) ), X) )

theta = np.zeros( (2, 1) )

iterations = 1500

alpha = 0.01

def computeCost(X, y, theta):
    m = len(y)
    J = np.sum((np.dot(X, theta)-y)**2 / m / 2.0)
    return J

J = computeCost(X, y, theta)

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - alpha * np.dot(X.T, np.dot(X, theta) - y) / m
        J_history[iter] = computeCost(X, y, theta)
    return theta, J_history

theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

plt.plot(X[:, 1], np.dot(X, theta))

plt.legend(['Training data', 'Linear regression'])

plt.show()

predict1 = np.dot(np.array([[1, 3.5]]), theta)[0, 0] * 10000

predict2 = np.dot(np.array([[1, 7]]), theta)[0, 0] * 10000

theta0_vals = np.linspace(-10, 10, 100)

theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

for i, theta0_val in enumerate(theta0_vals):
    for j, theta1_val in enumerate(theta1_vals):
	  t = np.array([theta0_val, theta1_val]).reshape(-1,1)    
	  J_vals[i, j] = computeCost(X, y, t)

fig = plt.figure()

ax = fig.gca(projection='3d') 

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

ax.plot_surface(theta0_vals, theta1_vals, J_vals.T)

plt.show()

contour(np.linspace(-10, 10, 100), np.linspace(-1, 4, 100), J_vals.T, logspace(-2, 3, 20))

show()