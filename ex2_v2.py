# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:03:02 2015

@author: ba0hx
"""

import numpy as np
from pylab import *
import scipy.optimize as opt

data = np.loadtxt('ex2data1.txt', delimiter = ',')

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
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    legend(['Admitted', 'Not admitted'])

plotData(X, y)

m = X.shape[0]

n = X.shape[1]

X = np.hstack( (np.ones( (m, 1) ), X) )

initial_theta = np.zeros((n + 1, 1))

def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z))

def costFunction(theta, X, y):
    n = X.shape[1]
    theta = theta.reshape((n,1)) #roll
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    return sum(-(y*np.log(h))-(1-y)*np.log(1-h)) / m

def gradFunction(theta,X,y):
    n = X.shape[1]
    theta = theta.reshape((n,1)) #roll
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    error = h - y
    grad = np.dot(X.T, error) / m    
    return grad.ravel() #unroll

#注意：无论initial_theta形如(n,)还是(n,1)，
#传入costFunction和gradFunction的theta值都是(n,)的样子，
#同时gradFunction返回的也必须是(n,)的形式
#类似于Matlab中的Roll和Unroll的操作
theta_optimize = opt.fmin_l_bfgs_b(costFunction, initial_theta.ravel(), 
                                   fprime=gradFunction, args=(X, y))

def plotDecisionBoundary(theta, X, y):
    plotData(X[:,1:3], y)
    plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,1])+2])
    plot_y = (-1.0/theta[2,0])*(theta[1,0]*plot_x + theta[0,0])
    plot(plot_x, plot_y)
    legend([ 'Decision Boundary', 'Admitted', 'Not admitted'])
    #legend(['Decision Boundary'])
    axis([30, 100, 30, 100])

theta = theta_optimize[0].reshape((n+1,1))

plotDecisionBoundary(theta, X, y)

prob = sigmoid(np.dot([[1,45,85]], theta))

def predict(theta, X):
    m = X.shape[0]
    p = np.zeros((m, 1))
    pos = np.where(sigmoid(np.dot(X, theta)) >= 0.5);
    p[pos, 0] = 1
    return p

p = predict(theta, X)

print np.mean(np.array((p == y).ravel(), dtype=np.float)) * 100
