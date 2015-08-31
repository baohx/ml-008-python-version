# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:48:11 2015

@author: ba0hx
"""

import scipy.io as sio
import numpy as np
from pylab import *
from PIL import Image
from matplotlib.colors import hsv_to_rgb

data=sio.loadmat('ex7data2.mat')
X = data['X']

K = 3
initial_centroids = np.array([[3,3],[6,2],[8,5]])
def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    m = X.shape[0]
    idx = np.zeros((m,1))
    for i in xrange(1, m+1):
        d = np.zeros((K,1))
        for j in xrange(1, K+1):
            d[j-1,0] = np.sum(np.power(X[i-1,:]-centroids[j-1,:],2))
        (r,c) = where(d==np.min(d))
        idx[i-1,0] = r[0]
    return idx
idx = findClosestCentroids(X, initial_centroids)
def computeCentroids(X, idx, K):
    m,n=X.shape
    centroids = np.zeros((K, n))
    for i in xrange(1,K+1):
        centroids[i-1,:] = np.mean(X[(idx==i-1).ravel()],axis=0)
    return centroids
centroids = computeCentroids(X, idx, K)
print centroids
def plotDataPoints(X, idx, K):
    V = H = np.linspace(0,1,K).reshape((-1,1))
    S = np.ones_like(V)
    HSV = np.hstack((H,S,V))
    RGB = hsv_to_rgb(HSV)
    colors = np.array([RGB[int(i[0])] for i in idx])

    scatter(X[:,0], X[:,1], s = np.pi * 5 ** 2, alpha=0.1, c = colors)

def drawLine(p1, p2):
    plot([p1[0],p2[0]], [p1[1],p2[1]])

def plotProgresskMeans(X, centroids, previous, idx, K, i):
    plotDataPoints(X, idx, K)
    plot(centroids[:,0], centroids[:,1], 'x')
    for j in xrange(1, centroids.shape[0]+1):
        drawLine(centroids[j-1, :], previous[j-1, :])

def runkMeans(X, initial_centroids,max_iters, plot_progress=False):
    if plot_progress:
        figure()
    m,n=X.shape
    K=initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    for i in xrange(1,max_iters+1):
        idx = findClosestCentroids(X, centroids)
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
        centroids = computeCentroids(X,idx, K)
    if plot_progress:
        show()
    return centroids, idx
max_iters = 10
centroids, idx = runkMeans(X, initial_centroids, max_iters, plot_progress = True)

A = asarray(Image.open('bird_small.png')).astype('float')
A = A / 255.00

img_size = A.shape

X = A.reshape((img_size[0]*img_size[1],3))
K = 16 
max_iters = 10
def kMeansInitCentroids(X, K):
    centroids = np.zeros((K, X.shape[1]))
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[0:K], :]
    return centroids
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)

idx = findClosestCentroids(X, centroids)
X_recovered = np.array([centroids[i[0]] for i in idx])
X_recovered = X_recovered.reshape((img_size[0], img_size[1], 3))

figure()
subplot(121)
imshow(A)
subplot(122)
imshow(X_recovered)
show()