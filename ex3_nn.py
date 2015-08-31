# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:39:55 2015

@author: ba0hx
"""
import scipy.io as sio
import numpy as np
from pylab import *
import matplotlib.cm as cm

input_layer_size  = 400
hidden_layer_size = 25
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

weightsData=sio.loadmat('ex3weights.mat')

Theta1 = weightsData['Theta1']

Theta2 = weightsData['Theta2']

def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z))

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    a1 = np.hstack( (np.ones( (m, 1) ), X) ) 
    z2 = np.dot(a1, Theta1.T)
    a2 = np.hstack( (np.ones( (m, 1) ), sigmoid(z2)) ) 
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    return (np.argmax(a3, axis = 1) + 1).reshape((m, 1))

pred = predict(Theta1, Theta2, X)

print np.mean(np.array((pred == y).ravel(), dtype=np.float)) * 100
