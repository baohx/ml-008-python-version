# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:39:55 2015

@author: ba0hx
"""
import scipy.io as sio
import numpy as np
from pylab import *
import scipy.optimize as opt
import matplotlib.cm as cm
import seaborn; seaborn.set()
from sklearn.linear_model import LogisticRegression

num_labels = 10

data=sio.loadmat('ex3data1.mat')

X = data['X']

y = data['y']

m = X.shape[0]

n = X.shape[1]

rand_indices = range(m)

np.random.shuffle(rand_indices)

sel = X[rand_indices[0:100], :]

def displayData(X, example_width = np.round(np.sqrt(X.shape[1]))):
    m = X.shape[0]
    n = X.shape[1]
    example_height = (n / example_width)
    display_rows = np.int(np.floor(np.sqrt(m)))
    display_cols = np.int(np.ceil(m / display_rows))
    pad = 1
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width + pad)))
    curr_ex = 1
    for j in xrange(1, display_rows + 1):
        for i in xrange(1, display_cols + 1):
            if curr_ex > m: 
                break
            max_val = np.max(np.abs(X[curr_ex - 1, :]))
            
            display_array[pad + (j - 1) * (example_height + pad) : pad + (j - 1) * (example_height + pad) + example_height, pad + (i - 1) * (example_width + pad) : pad + (i - 1) * (example_width + pad) + example_width] = \
                X[curr_ex - 1, :].reshape((example_height, example_width)).T / max_val
            curr_ex = curr_ex + 1
        if curr_ex > m:
            break

    figure()
    imshow(display_array, cmap = cm.Greys_r)
    axis('off')
    show()

displayData(sel)

lamb = 0.1

def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z))

def lrCostFunction(theta, X, y, lamb):
    n = X.shape[1]
    m = y.shape[0]
    theta = theta.reshape((n,1)) #roll
    h = sigmoid(np.dot(X, theta))
    #编程发现-(y*np.log(h))和-y*np.log(h)处理结果不一样的说明：
    #首先，参数传入y：通过y==c方式传入的是布尔数组，-True是0，而不是-1，影响计算，-(y*np.log(h))和-y*np.log(h)是不一样的
    #再次，就算通过y_of_c = copy.deepcopy(y)；y_of_c[y <> c] = 0.0；y_of_c[y == c] = 1.0方式传入y_of_c，
    #y的类型是uint8，是无符号数，-y产生非预期值，因此-(y*np.log(h))和-y*np.log(h)也是不一样的
    #这里通过采用-(y*np.log(h))解决
    return sum(-(y*np.log(h))-(1-y)*np.log(1-h)) / m + lamb / 2 / m * sum(np.power(theta[1:, :], 2))

def gradFunction(theta, X, y, lamb):
    n = X.shape[1]
    theta = theta.reshape((n,1)) #roll
    m = y.shape[0]
    h = sigmoid(np.dot(X, theta))
    error = h - y
    grad = np.dot(X.T, error) / m + np.append([[0]], lamb / m * theta[1:, :], axis=0)
    return grad.ravel() #unroll

def oneVsAll(X, y, num_labels, lamb):
    m = X.shape[0]
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n + 1))
    X = np.hstack( (np.ones( (m, 1) ), X) )  
    for c in xrange(1, num_labels + 1):
        initial_theta = np.zeros((n + 1, 1))
        theta_optimize = opt.fmin_l_bfgs_b(lrCostFunction, initial_theta.ravel(), fprime=gradFunction, args=(X, y==c, lamb))
        all_theta[c - 1,:] = theta_optimize[0].reshape((1, -1))
        
        model = LogisticRegression()
        model.fit(X,y==c)

    return all_theta

all_theta = oneVsAll(X, y, num_labels, lamb)

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    X = np.hstack( (np.ones( (m, 1) ), X) )
    pb = sigmoid(np.dot(X,all_theta.T))
    return (np.argmax(pb, axis = 1) + 1).reshape((m, 1))

pred = predictOneVsAll(all_theta, X)

print np.mean(np.array((pred == y).ravel(), dtype=np.float)) * 100
