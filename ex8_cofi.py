# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:49:06 2015

@author: ba0hx
"""

import scipy.io as sio
import numpy as np
from pylab import *
import scipy.optimize as opt

data=sio.loadmat('ex8_movies.mat')
Y = np.array(data['Y'],dtype=np.float64)
R = np.array(data['R'],dtype=np.float64)
print np.mean(Y[0,R[0,:] == 1])
figure()
imshow(Y)
show()

data=sio.loadmat('ex8_movieParams.mat')
Theta = data['Theta']
X = data['X']
num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]
def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lamb):
    X = params[0:num_movies*num_features].reshape((num_movies, num_features))
    Theta = params[num_movies*num_features:].reshape((num_users, num_features))
    M = np.power(np.dot(X,Theta.T) - Y,2)
    J = np.sum(R*M) / 2 + lamb * np.sum(np.power(Theta,2)) /2 + lamb * np.sum(np.power(X,2)) /2
    return J
def cofiGradFunc(params, Y, R, num_users, num_movies, num_features, lamb):
    X = params[0:num_movies*num_features].reshape((num_movies, num_features))
    Theta = params[num_movies*num_features:].reshape((num_users, num_features))
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    for i in xrange(1,num_movies+1):
        idx = R[i-1,:]==1
        Theta_temp = Theta[idx,:]
        Y_temp = Y[i-1,idx]
        X_grad[i-1,:] = np.dot(np.dot(X[i-1,:],Theta_temp.T)-Y_temp,Theta_temp) + lamb * X[i-1,:]
    for j in xrange(1,num_users+1):
        idx = R[:,j-1]==1
        X_temp = X[idx,:]
        Theta_temp = Theta[j-1,:]
        Y_temp = Y[idx,j-1]
        Theta_grad[j-1,:] = np.dot((np.dot(X_temp,Theta_temp.T)-Y_temp).T,X_temp) + lamb * Theta_temp
    grad = np.hstack((X_grad.ravel(),Theta_grad.ravel()))
    return grad
    
J = cofiCostFunc(np.hstack((X.ravel(),Theta.ravel())), Y, R, num_users, num_movies,num_features, 0)
print J
def computeNumericalGradient(J, theta):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in xrange(1,len(theta)+1):
        perturb[p-1] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[p-1] = (loss2 - loss1) / (2*e)
        perturb[p-1] = 0
    return numgrad
def checkCostFunction(lamb=0):
    X_t = np.random.rand(4,3)
    Theta_t = np.random.rand(5,3)
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(Y.shape[0],Y.shape[1])>0.5]=0
    R = np.zeros(Y.shape)
    R[Y <> 0] = 1
    X = np.random.rand(X_t.shape[0],X_t.shape[1])
    Theta = np.random.rand(Theta_t.shape[0],Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]
    def cofiCostFuncTemp(t):
        return cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lamb)
    numgrad = computeNumericalGradient(cofiCostFuncTemp,np.hstack((X.ravel(),Theta.ravel())))
    grad = cofiGradFunc(np.hstack((X.ravel(),Theta.ravel())),  Y, R, num_users,num_movies, num_features, lamb)
    print numgrad
    print grad
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print diff
checkCostFunction()
J = cofiCostFunc(np.hstack((X.ravel(),Theta.ravel())), Y, R, num_users, num_movies,num_features, 1.5)
print J
checkCostFunction(1.5)

def loadMovieList():
    movieList = []
    movieListFile = open('movie_ids.txt','r')
    movieListFileLines = movieListFile.readlines()
    movieListFile.close()
    for i,movieListFileLine in enumerate(movieListFileLines):
        movieListFileLine = movieListFileLine.replace('\n','')
        if movieListFileLine <> '':
            movieList.append(movieListFileLine.replace(str(i+1)+' ',''))
    return movieList

movieList = loadMovieList()

my_ratings = np.zeros((1682, 1))

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6,0] = 3
my_ratings[11,0] = 5
my_ratings[53,0] = 4
my_ratings[63,0] = 5
my_ratings[65,0] = 3
my_ratings[68,0] = 5
my_ratings[182,0] = 4
my_ratings[225,0] = 5
my_ratings[354,0] = 5
for i in xrange(1,len(my_ratings)+1):
    if my_ratings[i-1] > 0: 
        print 'Rated ' + str(my_ratings[i-1,0]) + ' for ' + movieList[i-1]

data=sio.loadmat('ex8_movies.mat')
Y = np.array(data['Y'],dtype=np.float64)
R = np.array(data['R'],dtype=np.float64)
Y = np.hstack((my_ratings,Y))
R = np.hstack((np.array(my_ratings <> 0,dtype=np.float64),R))
def normalizeRatings(Y, R):
    m,n = Y.shape
    Ymean = np.zeros((m,1))
    Ynorm = np.zeros(Y.shape)
    for i in xrange(1,m+1):
        idx = R[i-1,:] == 1
        Ymean[i-1,0] = np.mean(Y[i-1,idx])
        Ynorm[i-1,idx] = Y[i-1,idx] - Ymean[i-1,0]
    return Ynorm, Ymean
Ynorm, Ymean = normalizeRatings(Y, R)

num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

X = np.random.rand(num_movies, num_features)
Theta = np.random.rand(num_users, num_features)

initial_parameters = np.hstack((X.ravel(),Theta.ravel()))

lamb = 10.0

theta_optimize = opt.fmin_l_bfgs_b(cofiCostFunc, initial_parameters, 
                                   fprime=cofiGradFunc, args=(Y, R, num_users, num_movies, num_features, lamb), maxiter = 100)
theta = theta_optimize[0]

X = theta[0:num_movies*num_features].reshape((num_movies, num_features))
Theta = theta[num_movies*num_features:].reshape((num_users, num_features))

p = np.dot(X,Theta.T)
my_predictions = p[:,0] + Ymean.ravel()
movieList = loadMovieList()

ix = my_predictions.argsort()
for i in xrange(1,11):
    j = ix[-i]
    print 'Predicting rating '+str(my_predictions[j])+' for movie '+ movieList[j]

for i in xrange(1,len(my_ratings)+1):
    if my_ratings[i-1] > 0: 
        print 'Rated '+str(my_ratings[i-1,0])+' for '+movieList[i-1]
