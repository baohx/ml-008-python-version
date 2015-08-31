# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:57:41 2015

@author: ba0hx
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:03:02 2015

@author: ba0hx
"""

import numpy as np
from pylab import *
import scipy.optimize as opt

data = np.loadtxt('ex2data2.txt', delimiter = ',')

COLUMN_NUM = 2

X = data[:, 0:COLUMN_NUM]

y = data[:, COLUMN_NUM]

y = y.reshape((-1, 1))

def plotData(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    figure()
    scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    legend(['y = 1', 'y = 0'])

plotData(X, y)

def mapFeature(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0],1))
    for i in xrange(1, degree + 1):
        for j in xrange(0, i + 1):
            out = np.append(out, np.power(X1, i-j) * np.power(X2, j), axis=1)
    return out

X = mapFeature(X[:,0].reshape((-1,1)), X[:,1].reshape((-1,1)))

n = X.shape[1]

initial_theta = np.zeros((n, 1))

lamb = 1.0

def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z))

def costFunctionReg(theta, X, y, lamb):
    n = X.shape[1]
    theta = theta.reshape((n,1)) #roll
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    return sum(-(y*np.log(h))-(1-y)*np.log(1-h)) / m + lamb / 2 / m * sum(np.power(theta[1:, :], 2))

def gradFunctionReg(theta, X, y, lamb):
    n = X.shape[1]
    theta = theta.reshape((n,1)) #roll
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    error = h - y
    grad = np.dot(X.T, error) / m + np.append([[0]], lamb / m * theta[1:, :], axis=0)
    return grad.ravel() #unroll

cost = costFunctionReg(initial_theta, X, y, lamb)

grad = gradFunctionReg(initial_theta, X, y, lamb)

#注意：无论initial_theta形如(n,)还是(n,1)，
#传入costFunction和gradFunction的theta值都是(n,)的样子，
#同时gradFunction返回的也必须是(n,)的形式
#类似于Matlab中的Roll和Unroll的操作
theta_optimize = opt.fmin_l_bfgs_b(costFunctionReg, initial_theta.ravel(), 
                                   fprime=gradFunctionReg, args=(X, y, lamb), maxiter = 400)

def plotDecisionBoundary(theta, X, y):
    plotData(X[:,1:3], y)
    if X.shape[1] <= 3:
        plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,1])+2])
        plot_y = (-1.0/theta[2,0])*(theta[1,0]*plot_x + theta[0,0])
        plot(plot_x, plot_y)
        legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        axis([30, 100, 30, 100])
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in xrange(0, len(u)):
            for j in xrange(0, len(v)):
                z[i,j] = np.dot(mapFeature(u[i].reshape((-1, 1)), v[j].reshape((-1, 1))), theta)
        z = z.T
        contour(u, v, z, 1)


theta = theta_optimize[0].reshape((n,1))

plotDecisionBoundary(theta, X, y)

def predict(theta, X):
    m = X.shape[0]
    p = np.zeros((m, 1))
    pos = np.where(sigmoid(np.dot(X, theta)) >= 0.5);
    p[pos, 0] = 1
    return p

p = predict(theta, X)

print np.mean(np.array((p == y).ravel(), dtype=np.float)) * 100
