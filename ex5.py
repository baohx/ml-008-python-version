# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:44:20 2015

@author: ba0hx

说明：结果和Matlab中不同，推测可能和小数位数有关系。
"""

import scipy.io as sio
import numpy as np
from pylab import *
import scipy.optimize as opt


data=sio.loadmat('ex5data1.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']
m = X.shape[0]

figure()
scatter(X, y, marker='x', c='r')
xlabel('Change in water level (x)')
ylabel('Water flowing out of the dam (y)')

theta = np.array([[1.0],[1.0]])

def linearRegCostFunction(theta, X, y, lamb):
    n = X.shape[1]
    theta = theta.reshape((n,1))
    m = len(y)
    J = np.sum((np.dot(X, theta)-y)**2 / m / 2.0) + lamb * np.sum((theta[1:,:])**2 / m / 2.0)
    return J

def gradRegFunction(theta, X, y, lamb):
    n = X.shape[1]
    theta = theta.reshape((n,1))
    m = len(y)
    grad = np.dot(X.T, np.dot(X, theta)-y) / m + np.append([[0]], lamb / m * theta[1:, :], axis=0)
    return grad.ravel()

J = linearRegCostFunction(theta.ravel(), np.hstack( (np.ones( (m, 1) ), X) ), y, 1.0)
grad = gradRegFunction(theta.ravel(), np.hstack( (np.ones( (m, 1) ), X) ), y, 1.0)

lamb = 0.0

def trainLinearReg(X, y, lamb):
    initial_theta = np.zeros((X.shape[1], 1))
    return opt.fmin_l_bfgs_b(linearRegCostFunction, initial_theta.ravel(), fprime=gradRegFunction, args=(X, y, lamb), maxiter = 200)

theta =  trainLinearReg(np.hstack( (np.ones( (m, 1) ), X) ), y, lamb)[0]
figure()
scatter(X, y, marker='x', c='r')
xlabel('Change in water level (x)')
ylabel('Water flowing out of the dam (y)')
plot(X, np.dot(np.hstack( (np.ones( (m, 1) ), X) ), theta.reshape((-1, 1))))

lamb = 0.0

def learningCurve(X, y, Xval, yval, lamb):
    m = X.shape[0]
    error_train = np.zeros((m,))
    error_val = np.zeros((m,))
    for i in xrange(1, m+1):
        theta =  trainLinearReg(X[0:i,:], y[0:i,:], lamb)[0]
        error_train[i-1] = linearRegCostFunction(theta, X[0:i,:], y[0:i,:], 0.00)
        error_val[i-1] = linearRegCostFunction(theta, Xval, yval, 0.00)
    return error_train, error_val
print 'start:'
error_train, error_val = learningCurve(np.hstack( (np.ones( (m, 1) ), X) ), y, np.hstack( (np.ones( (Xval.shape[0], 1) ), Xval) ), yval, lamb)

figure()
plot(xrange(1, m+1), error_train)
plot(xrange(1, m+1), error_val)
title('Learning curve for linear regression')
legend(['Train', 'Cross Validation'])
xlabel('Number of training examples')
ylabel('Error')
axis([0, 13, 0, 150])

for i in xrange(1, m+1):
    print i, error_train[i-1], error_val[i-1]

p = 8

def polyFeatures(X, p):
    return np.power(X, np.array(range(1, p+1)).reshape((1,-1)))

def featureNormalize(X):
    mu = np.mean(X, axis = 0)
    X_norm = X - mu
    #ddof = 1得到的数字和matlab中的std相同，区别如下：
    #numpy默认：1 / n * sum((xi - mean(x)) ** 2)
    #matlab默认：1 / (n - 1) * sum((xi - mean(x)) ** 2)
    sigma = np.std(X_norm, axis = 0,ddof = 1)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.hstack( (np.ones( (m, 1) ), X_poly) )
print X_poly[0, :]

X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.hstack( (np.ones( (X_poly_test.shape[0], 1) ), X_poly_test) )

X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.hstack( (np.ones( (X_poly_val.shape[0], 1) ), X_poly_val) )

lamb = 0.0

theta = trainLinearReg(X_poly, y, lamb)[0]
figure()
scatter(X, y, marker='x', c='r')
def plotFit(min_x, max_x, mu, sigma, theta, p):
    x = np.linspace(np.min(X)-15,np.max(X)+25,(np.max(X)-np.min(X)+10)/0.05).reshape((-1,1))
    X_poly = polyFeatures(x, p)
    X_poly = X_poly-mu
    X_poly = X_poly/sigma
    X_poly = np.hstack( (np.ones( (x.shape[0], 1) ), X_poly) )
    plot(x, np.dot(X_poly,theta))
plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
xlabel('Change in water level (x)')
ylabel('Water flowing out of the dam (y)')
title('Polynomial Regression Fit (lambda = '+str(lamb)+')')

figure()
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lamb)
plot(range(1,m+1), error_train, range(1,m+1), error_val)
title('Polynomial Regression Learning Curve (lambda = '+str(lamb)+')')
xlabel('Number of training examples')
ylabel('Error')
axis([0,13,0,100])
legend(['Train', 'Cross Validation'])

for i in xrange(1, m+1):
    print i, error_train[i-1], error_val[i-1]

def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0.0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]).reshape((-1,1))
    error_train = np.zeros((lambda_vec.shape[0],))
    error_val = np.zeros((lambda_vec.shape[0],))
    for i in xrange(1,lambda_vec.shape[0]+1):
        lamb = lambda_vec[i-1,0]
        theta = trainLinearReg(X, y, lamb)[0]
        error_train[i-1] = linearRegCostFunction(theta,X,y,0.00)
        error_val[i-1] = linearRegCostFunction(theta,Xval, yval, 0.00)
    return lambda_vec, error_train, error_val
lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)
figure()
plot(lambda_vec, error_train, lambda_vec, error_val)
legend(['Train', 'Cross Validation'])
xlabel('lambda')
ylabel('Error')

for i in xrange(1,lambda_vec.shape[0]+1):
    print lambda_vec[i-1], error_train[i-1], error_val[i-1]