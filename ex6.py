# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 08:44:35 2015

@author: ba0hx
#对于K[:i]不能想当然认为是列向量，其实是行向量，需要进行转换
#sigma有待改进
"""


import scipy.io as sio
import numpy as np
from pylab import *

data=sio.loadmat('ex6data1.mat')
X = data['X']
y = np.array(data['y'], dtype = np.float64)

def plotData(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    figure()
    scatter(X[pos, 0], X[pos, 1], marker='+')
    scatter(X[neg, 0], X[neg, 1], marker='o', c='y')

plotData(X, y)
show()
C = 10
def linearKernel(x1, x2, sigma):
    x1 = x1.reshape((-1,1))
    x2 = x2.reshape((-1,1))
    return np.dot(x1.T, x2)

def gaussianKernel(x1, x2, sigma):
    x1 = x1.reshape((-1,1))
    x2 = x2.reshape((-1,1))
    return np.exp(-np.sum(np.power(x1-x2,2))/(2*np.power(sigma,2)))

def visualizeBoundaryLinear(X, y, model):
    w = model.w
    b = model.b
    xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    yp = - (w[0,0]*xp + b)/w[1,0]
    plotData(X, y)
    plot(xp, yp)
    show()

class Model():
    pass

def svmTrain(X, Y, C, kernelFunction, sigma, tol=1e-3, max_passes=5):
    m, n = X.shape
    Y=Y.copy()
    Y[Y==0] = -1
    alphas = np.zeros((m, 1))
    b = 0.0
    #E = np.zeros((m, 1))
    passes = 0
    eta = 0.0
    L = 0.0
    H = 0.0
    if kernelFunction.__name__ == 'linearKernel':
        K = np.dot(X, X.T)
    elif kernelFunction.__name__ .find('gaussianKernel') >= 0:
        X2 = np.sum(np.power(X, 2), axis = 1).reshape((-1,1))
        K = X2 + (X2.T - 2 *  np.dot(X, X.T))
        K = np.power(kernelFunction(np.array([1]),np.array([0]),sigma), K)
    else:
        K = np.zeros((m,m))
        for i in xrange(1, m+1):
            for j in xrange(1, m+1):
                K[i-1,j-1] = kernelFunction(X[i-1,:].T, X[j-1,:].T,sigma)
                K[j-1,i-1] = K[i-1,j-1]
    dots = 12
    while passes < max_passes:
        num_changed_alphas = 0
        for i in xrange(1, m+1):
            fXi = b + np.sum(alphas*Y*K[:,i-1].reshape((-1,1)))
            Ei = fXi - Y[i-1,0]
            if ((Y[i-1,0]*Ei < -tol and alphas[i-1,0] < C) or (Y[i-1,0]*Ei > tol and alphas[i-1,0] > 0)):
                j = np.ceil(m * np.random.rand())
                while j == i:
                    j = np.ceil(m * np.random.rand())
                fXj = b + np.sum(alphas*Y*K[:,j-1].reshape((-1,1)))
                Ej = fXj - Y[j-1,0]
                alpha_i_old = alphas[i-1,0].copy()
                alpha_j_old = alphas[j-1,0].copy()
                
                if Y[i-1,0] == Y[j-1,0]:
                    L = max(0, alphas[j-1,0] + alphas[i-1,0] - C)
                    H = min(C, alphas[j-1,0] + alphas[i-1,0])
                else:
                    L = max(0, alphas[j-1,0] - alphas[i-1,0])
                    H = min(C, C + alphas[j-1,0] - alphas[i-1,0])

                if L == H:
                    continue
                    
                eta = 2.0 * K[i-1,j-1] - K[i-1,i-1] - K[j-1,j-1]
                if eta >= 0.0:
                    continue
            
                alphas[j-1,0] = alphas[j-1,0] - (Y[j-1,0] * (Ei - Ej)) / eta
            
                alphas[j-1,0] = min(H, alphas[j-1,0])
                alphas[j-1,0] = max(L, alphas[j-1,0])
            
                if (np.abs(alphas[j-1,0] - alpha_j_old) < tol):
                    alphas[j-1,0] = alpha_j_old
                    continue
            
                alphas[i-1,0] = alphas[i-1,0] + Y[i-1,0]*Y[j-1,0]*(alpha_j_old - alphas[j-1,0])
            
                b1 = b - Ei - Y[i-1,0] * (alphas[i-1,0] - alpha_i_old) *  K[i-1,j-1] - Y[j-1,0] * (alphas[j-1,0] - alpha_j_old) *  K[i-1,j-1]
                b2 = b - Ej - Y[i-1,0] * (alphas[i-1,0] - alpha_i_old) *  K[i-1,j-1] - Y[j-1,0] * (alphas[j-1,0] - alpha_j_old) *  K[j-1,j-1]

                if (0 < alphas[i-1,0] and alphas[i-1,0] < C):
                    b = b1
                elif (0 < alphas[j-1,0] and alphas[j-1,0] < C):
                    b = b2
                else:
                    b = (b1+b2)/2

                num_changed_alphas = num_changed_alphas + 1
        if (num_changed_alphas == 0):
            passes = passes + 1
        else:
            passes = 0
        dots = dots + 1
        if dots > 78:
            dots = 0
    idx = alphas.ravel() > 0
    model = Model()    
    model.X= X[idx,:]
    model.y= Y[idx,:]
    model.kernelFunction = kernelFunction
    model.b= b
    model.alphas= alphas[idx,:]
    model.w = np.dot((alphas * Y).T,X).T
    model.sigma = sigma
    return model
    
model = svmTrain(X, y, C, linearKernel, 0.0, max_passes=20)
visualizeBoundaryLinear(X, y, model)


x1 = np.array([[1,2,1]])
x2 = np.array([[0,4,-1]])
sigma = 2.0
sim = gaussianKernel(x1, x2, sigma)

data=sio.loadmat('ex6data2.mat')
X = data['X']
y = np.array(data['y'], dtype = np.float64)
plotData(X, y)
show()
C = 10
sigma = 0.1
model= svmTrain(X, y, C, gaussianKernel, sigma)

def svmPredict(model, X):
    if X.shape[1] == 1:
        X = X.T
    m = X.shape[0]
    p = np.zeros((m, 1))
    pred = np.zeros((m, 1))
    if model.kernelFunction.__name__ == 'linearKernel':
        p = np.dot(X, model.w) + model.b
    elif model.kernelFunction.__name__ .find('gaussianKernel') >= 0:
        X1 = np.sum(np.power(X, 2), axis = 1).reshape((-1,1))
        X2 = np.sum(np.power(model.X, 2), axis = 1).reshape((-1,1)).T
        K = X1 + (X2 - 2 *  np.dot(X, model.X.T))
        K = np.power(model.kernelFunction(np.array([1]),np.array([0]),model.sigma), K)
        K = model.y.T * K
        K = model.alphas.T * K
        p = np.sum(K, axis = 1).reshape((-1,1))
    else:
        for i in xrange(1, m+1):
            prediction = 0
            for j in xrange(1, model.X.shape[0]+1):
                prediction = prediction + model.alphas[j-1,0] * model.y[j-1,0] * model.kernelFunction(X[i-1,:].T, model.X[j-1,:].T, model.sigma)
            p[i-1,0] = prediction + model.b
    pred[p >= 0] = 1
    pred[p < 0] = 0
    return pred

def visualizeBoundary(X, y, model):
    plotData(X, y)
    x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
    x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in xrange(1, X1.shape[1]+1):
        this_X = np.hstack( (X1[:,i-1].reshape((-1,1)), X2[:,i-1].reshape(-1,1)) )
        vals[:,i-1] = svmPredict(model, this_X).ravel()
    contour(X1, X2, vals)
    show()
visualizeBoundary(X, y, model)

data=sio.loadmat('ex6data3.mat')
X = data['X']
y = np.array(data['y'], dtype = np.float64)
Xval = data['Xval']
yval = np.array(data['yval'], dtype = np.float64)

plotData(X, y)
show()
def dataset3Params(X, y, Xval, yval):
    C = 10
    sigma = 0.3
    c_step = np.array([[0.01],[0.03],[0.1],[0.3],[1],[3],[10],[30]])
    sigma_step = np.array([[0.01],[0.03],[0.1],[0.3],[1],[3],[10],[30]])
    c_c = c_step.shape[0]
    sigma_c = sigma_step.shape[0]
    mean_history = np.zeros((c_c, sigma_c))
    for c_i in xrange(1, c_c+1):
        for sigma_i in xrange(1, sigma_c+1):
            sigma = sigma_step[sigma_i-1,0]
            model= svmTrain(X, y, c_step[c_i-1,0], gaussianKernel, sigma) 
            predictions = svmPredict(model, Xval)
            mean_history[c_i-1, sigma_i-1] = np.mean(np.array(predictions <> yval, dtype= np.double))
    (row,column) = where(mean_history == np.min(mean_history))
    C = c_step[row[0],0]
    sigma = sigma_step[column[0],0]
    return C, sigma
C, sigma= dataset3Params(X, y, Xval, yval)
model= svmTrain(X, y, C, gaussianKernel, sigma)

visualizeBoundary(X, y, model)