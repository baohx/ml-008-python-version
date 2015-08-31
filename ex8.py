# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:38:56 2015

@author: ba0hx
"""

import scipy.io as sio
import numpy as np
from pylab import *
data=sio.loadmat('ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = np.array(data['yval'], dtype = np.float64)

figure()
scatter(X[:, 0], X[:, 1], marker='x', c='b')
axis([0,30,0,30])
xlabel('Latency (ms)')
ylabel('Throughput (mb/s)')
show()
def estimateGaussian(X):
    m,n= X.shape
    mu = np.mean(X, axis = 0)
    sigma2 = (np.sum(np.power(X - mu, 2), axis = 0) / m).reshape((n,1))
    mu = mu.reshape((n,1))
    return mu,sigma2
mu,sigma2 = estimateGaussian(X)
def multivariateGaussian(X, mu, Sigma2):
    k= len(mu)
    if Sigma2.shape[1] == 1 or Sigma2.shape[0] == 1:
        Sigma2 = np.diag(Sigma2.ravel())
    X = X - mu.ravel()
    return np.power(2 * np.pi, -k/2.0) * np.power(np.linalg.det(Sigma2),-0.5) * np.exp(-0.5*np.sum(np.dot(X, np.linalg.pinv(Sigma2)) * X, axis = 1)).reshape((-1,1))
p = multivariateGaussian(X, mu, sigma2)
def visualizeFit(X, mu, sigma2):
    n = arange(0,35.5,0.5)
    [X1,X2] =meshgrid(n,n)
    Z = multivariateGaussian(np.hstack((X1.ravel(1).reshape((-1,1)),X2.ravel(1).reshape((-1,1)))),mu,sigma2)
    #Matlab的reshape先列后行，numpy的reshape先行后列    
    #reshape(1:6,2,3); arange(1,7).reshape(-1,2).transpose()
    Z = Z.reshape((-1,X1.shape[0])).T
    scatter(X[:, 0], X[:, 1], marker='x', c='b')
    contour(X1, X2, Z,  np.power(10,arange(-20.0,0.0,3.0)))
    #, np.power(10,arange(-20,0,3))
figure()
visualizeFit(X,  mu, sigma2)
xlabel('Latency (ms)')
ylabel('Throughput (mb/s)')

pval = multivariateGaussian(Xval, mu, sigma2)

def selectThreshold(yval, pval):
    bestEpsilon = 0.0
    bestF1 = 0.0
    F1 = 0.0
    stepsize = (np.max(pval)-np.min(pval))/1000.0
    for epsilon in arange(np.min(pval),np.max(pval),stepsize):
        predictions = (pval < epsilon)
        tp = len(predictions[yval == 1][predictions[yval == 1] == 1])
        fp = len(predictions[yval == 0][predictions[yval == 0] == 1])
        fn = len(predictions[yval == 1][predictions[yval == 1] == 0])
        if tp == 0:
            prec = rec = F1 = 0.0
        else:
            prec = 1.0 * tp / (tp + fp)
            rec = 1.0 * tp / (tp + fn)
            F1 = 2.0 * prec * rec / (prec + rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1
epsilon,F1 = selectThreshold(yval, pval)
outliers = p < epsilon
scatter(X[outliers.ravel(), 0], X[outliers.ravel(), 1], marker='o', c='r')
show()
data=sio.loadmat('ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = np.array(data['yval'], dtype = np.float64)
mu,sigma2 = estimateGaussian(X)
p = multivariateGaussian(X, mu, sigma2)
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon,F1 = selectThreshold(yval, pval)
print epsilon
print F1
print np.sum(p < epsilon)