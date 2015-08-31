# -*- coding: utf-8 -*-
"""
Created on Mon May 04 13:47:25 2015

@author: ba0hx
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import seaborn; seaborn.set()
from sklearn.linear_model import LogisticRegression
def plotData(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    figure()
    scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    legend(['y = 1', 'y = 0'])
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

data = np.loadtxt('ex2data2.txt', delimiter = ',')

COLUMN_NUM = 2

X = data[:, 0:COLUMN_NUM]

y = data[:, COLUMN_NUM]

def mapFeature(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0],1))
    for i in xrange(1, degree + 1):
        for j in xrange(0, i + 1):
            out = np.append(out, np.power(X1, i-j) * np.power(X2, j), axis=1)
    return out

X = mapFeature(X[:,0].reshape((-1,1)), X[:,1].reshape((-1,1)))

model = LogisticRegression(penalty='l2',C=0.034)

model.fit(X,y)

print(model.coef_)
print(model.intercept_)
plotDecisionBoundary((model.intercept_+model.coef_[0]).reshape((-1,1)), X, y[:,np.newaxis])