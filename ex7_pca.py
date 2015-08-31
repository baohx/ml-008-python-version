# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:04:38 2015

@author: ba0hx
"""


import scipy.io as sio
import numpy as np
from pylab import *
from PIL import Image

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

data=sio.loadmat('ex7data1.mat')
X = data['X']

figure()
scatter(X[:, 0], X[:, 1], marker='o', c='b')
axis([0.5,6.5,2,8])

def featureNormalize(X):
    mu = np.mean(X, axis = 0)
    X_norm = X - mu
    #ddof = 1得到的数字和matlab中的std相同，区别如下：
    #numpy默认：1 / n * sum((xi - mean(x)) ** 2)
    #matlab默认：1 / (n - 1) * sum((xi - mean(x)) ** 2)
    sigma = np.std(X_norm, axis = 0,ddof = 1)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma

X_norm, mu, sigma = featureNormalize(X)

def pca(X):
    m,n = X.shape
    sigma = np.dot(X.T, X) / m
    #Matlab和Numpy两者得出的U和S都是一致的，但是得出的V是转置关系！！
    return np.linalg.svd(sigma)

U,S,V = pca(X_norm)

def drawLine(p1, p2):
    plot([p1[0],p2[0]], [p1[1],p2[1]])

drawLine(mu, mu + 1.5 * S[0] * U[:,0].T)
drawLine(mu, mu + 1.5 * S[1] * U[:,1].T)
show()
print U[0,0], U[1,0]

figure()
scatter(X_norm[:, 0], X_norm[:, 1], marker='o', c='b')
axis([-4,3,-4,3])

K = 1
def projectData(X, U, K):
    Z = np.zeros((X.shape[0], K))
    for i in xrange(1, X.shape[0]+1):
        x = X[i-1,:].T
        Z[i-1,:] = np.dot(x.T,U[:,0:K])
    return Z
Z = projectData(X_norm, U, K)
print Z[0,0]
def recoverData(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    for i in xrange(1, Z.shape[0]+1):
        v = Z[i-1,:].T
        X_rec[i-1,:] = np.dot(v.T, U[:, 0:K].T)
    return X_rec
X_rec  = recoverData(Z, U, K)
print X_rec[0,0], X_rec[0,1]
scatter(X_rec[:, 0], X_rec[:, 1], marker='o', c='r')
for i in xrange(1,X_norm.shape[0]+1):
    drawLine(X_norm[i-1,:], X_rec[i-1,:])
show()

data=sio.loadmat('ex7faces.mat')
X = data['X']
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

    #figure()
    imshow(display_array, cmap = cm.Greys_r)
    axis('off')
    #show()
figure()
displayData(X[0:100, :])
show()
X_norm, mu, sigma = featureNormalize(X)
U, S,V = pca(X_norm)
figure()
displayData(U[:, 0:36].T)
show()
K = 100
Z = projectData(X_norm, U, K)
print Z.shape

K = 100
X_rec  = recoverData(Z, U, K)
figure()
subplot(121)
displayData(X_norm[0:100,:])
title('Original faces')
subplot(122)
displayData(X_rec[0:100,:])
title('Recovered faces')
show()

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
def plotProgresskMeans(X, centroids, previous, idx, K, i):
    plotDataPoints(X, idx, K)
    plot(centroids[:,0], centroids[:,1], 'x')
    for j in xrange(1, centroids.shape[0]+1):
        drawLine(centroids[j-1, :], previous[j-1, :])
def computeCentroids(X, idx, K):
    m,n=X.shape
    centroids = np.zeros((K, n))
    for i in xrange(1,K+1):
        centroids[i-1,:] = np.mean(X[(idx==i-1).ravel()],axis=0)
    return centroids
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
centroids, idx = runkMeans(X, initial_centroids, max_iters)
sel = np.array(np.floor(np.random.rand(1000, 1) * X.shape[0]),dtype=np.int).ravel()

fig = plt.figure()
ax = fig.gca(projection='3d')


V = H = np.linspace(0,1,K).reshape((-1,1))
S = np.ones_like(V)
HSV = np.hstack((H,S,V))
RGB = hsv_to_rgb(HSV)
colors = np.array([RGB[int(i[0])] for i in idx[sel,:]])

s = ax.scatter(X[sel,0], X[sel,1], X[sel,2], s = np.pi * 5 ** 2, alpha=0.1,c=colors)
title('Pixel dataset plotted in 3D. Color shows centroid memberships')

plt.show()

X_norm, mu, sigma = featureNormalize(X)

U, S, V = pca(X_norm)
Z = projectData(X_norm, U, 2)

figure()
def plotDataPoints(X, idx, K):
    V = H = np.linspace(0,1,K).reshape((-1,1))
    S = np.ones_like(V)
    HSV = np.hstack((H,S,V))
    RGB = hsv_to_rgb(HSV)
    colors = np.array([RGB[int(i[0])] for i in idx])
    scatter(X[:,0], X[:,1], s = np.pi * 5 ** 2, alpha=0.1, c = colors)

plotDataPoints(Z[sel, :], idx[sel,:], K)
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
show()