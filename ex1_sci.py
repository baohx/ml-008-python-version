# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:48:13 2015

@author: ba0hx
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
#import seaborn; seaborn.set()
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

def computeCost(X, y, theta):
    m = len(y)
    J = np.sum((np.dot(X, theta)-y)**2 / m / 2.0)
    return J

model = LinearRegression(normalize=True)
data = np.loadtxt('ex1data1.txt', delimiter = ',')
x = data[:, 0]
X = x[:, np.newaxis]
y= data[:, 1]
model.fit(X, y)
X_fit = np.linspace(5, 25, 100)[:, np.newaxis]
y_fit = model.predict(X_fit)
print(model.coef_)
print(model.intercept_)
print(model.residues_)
plt.figure()
plt.plot(X.squeeze(), y, 'o')
plt.plot(X_fit.squeeze(), y_fit)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()

predict11 = model.predict(np.array([[3.5]]))[0]*10000
predict22 = model.predict(np.array([[7.0]]))[0]*10000

theta0_vals = np.linspace(-10, 10, 100)

theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

for i, theta0_val in enumerate(theta0_vals):
    for j, theta1_val in enumerate(theta1_vals):
        t = np.array([theta0_val, theta1_val]).reshape(-1,1)
        J_vals[i, j] = computeCost(np.hstack( (np.ones( (X.shape[0], 1) ), X) ), y[:,np.newaxis], t)


fig = plt.figure()

ax = fig.gca(projection='3d') 

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

ax.plot_surface(theta0_vals, theta1_vals, J_vals.T)

plt.show()

contour(np.linspace(-10, 10, 100), np.linspace(-1, 4, 100), J_vals.T, logspace(-2, 3, 20))

show()