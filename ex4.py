# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 08:20:09 2015

@author: ba0hx
"""
import scipy.io as sio
import numpy as np
from pylab import *
import scipy.optimize as opt

input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

data=sio.loadmat('ex4data1.mat')

X = data['X']

y = data['y']

m = X.shape[0]

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

weightsData=sio.loadmat('ex4weights.mat')

Theta1 = weightsData['Theta1']

Theta2 = weightsData['Theta2']

nn_params = np.hstack( (Theta1.ravel(), Theta2.ravel()) )

def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1-sigmoid(z))

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    Theta1 = nn_params[0 : hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, (input_layer_size + 1)))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : ].reshape((num_labels, (hidden_layer_size + 1)))
    m = X.shape[0]
    J = 0    

    a1 = np.hstack( (np.ones( (m, 1) ), X) ) # 5000*401
    z2 = np.dot(a1, Theta1.T) #5000*25
    a2 = np.hstack( (np.ones( (m, 1) ), sigmoid(z2)) ) #5000*26
    z3 = np.dot(a2, Theta2.T) #5000*10
    h = sigmoid(z3) #5000*10
    for i in xrange(1, m + 1):
        yik = np.zeros((num_labels, 1)) # 10*1
        yik[y[i-1, 0] - 1, 0] = 1
        hik = h[i - 1, :].reshape(1,-1) # 1*10
        J = J - np.dot(np.log(hik), yik)[0, 0] - np.dot(np.log(1 - hik), (1 - yik))[0, 0]
    
    return J / m + lamb / 2 / m * (sum(np.power(Theta1[:,1:], 2)) + sum(np.power(Theta2[:,1:], 2)))

def gradFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    Theta1 = nn_params[0 : hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, (input_layer_size + 1)))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : ].reshape((num_labels, (hidden_layer_size + 1)))
    m = X.shape[0]
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    for i in xrange(1, m + 1):
        ai1 = np.hstack( (1, X[i-1,:]) ).reshape((1,-1)) # 1*401
        zi2 = np.dot(ai1, Theta1.T) # 1*25
        ai2 = np.hstack( (1, sigmoid(zi2).ravel()) ).reshape((1,-1)) # 1*26
        zi3 = np.dot(ai2, Theta2.T) # 1*10
        ai3 = sigmoid(zi3) # 1*10
        yik = np.zeros( (num_labels, 1) ) # 10*1
        yik[y[i-1, 0] - 1, 0] = 1
        ei3 = ai3.T - yik # 10*1
        ei2 = np.dot(Theta2.T, ei3) * np.hstack( (1, sigmoidGradient(zi2).ravel())).reshape((1,-1)).T # 26*1
        ei2 = ei2[1:, :] # 25*1
        d2 = np.dot(ei3, ai2) # 10*26
        d1 = np.dot(ei2, ai1) # 25*401 
        Theta1_grad = Theta1_grad + d1
        Theta2_grad = Theta2_grad + d2
    Theta1_grad = Theta1_grad / m + lamb / m * np.hstack( (np.zeros( (Theta1.shape[0], 1) ), Theta1[:,1:]) )
    Theta2_grad = Theta2_grad / m + lamb / m * np.hstack( (np.zeros( (Theta2.shape[0], 1) ), Theta2[:,1:]) )
    return np.hstack( (Theta1_grad.ravel(), Theta2_grad.ravel()) )

print 'first nnCostFunction:'
lamb = 0.0
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
print J

print 'second nnCostFunction:'
lamb = 1.0
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
print J

print 'sigmoidGradient:'
g = sigmoidGradient(np.array([[1, -0.5, 0, 0.5, 1]]))
print g

def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)

initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

initial_nn_params = np.hstack( (initial_Theta1.ravel(), initial_Theta2.ravel()) )

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    return np.reshape(np.sin(range(1,W.shape[0]*W.shape[1] + 1)), W.shape) / 10

def computeNumericalGradient(J, theta, input_layer_size, hidden_layer_size, num_labels, X, y, lamb):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in xrange(1, theta.size + 1):
        perturb[p-1] = e
        loss1 = J(theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
        loss2 = J(theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
        numgrad[p-1] = (loss2 - loss1) / (2*e)
        perturb[p-1] = 0
    return numgrad

def checkNNGradients(lamb):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = (1 + np.mod(range(1, m+1), num_labels).T).reshape((m, 1))
    nn_params = np.hstack( (Theta1.ravel(), Theta2.ravel()) )
    cost = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
    grad = gradFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
    numgrad = computeNumericalGradient(nnCostFunction, nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
    print 'numgrad: ', numgrad
    print 'grad: ', grad
    print 'cost: ', cost
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)    
    print 'diff:', diff

print 'first checkNNGradients:'
lamb = 0.0
checkNNGradients(lamb)

print 'second checkNNGradients:'
lamb = 3.0
checkNNGradients(lamb)

debug_J  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)

lamb = 1.0
theta_optimize = opt.fmin_l_bfgs_b(nnCostFunction, initial_nn_params, fprime=gradFunction, args=(input_layer_size, hidden_layer_size, num_labels, X, y, lamb), maxiter = 50)

nn_params = theta_optimize[0]

Theta1 = nn_params[0 : hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, (input_layer_size + 1)))
Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1) : ].reshape((num_labels, (hidden_layer_size + 1)))

displayData(Theta1[:, 1:])

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