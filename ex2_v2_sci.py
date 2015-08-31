# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:35:42 2015

@author: ba0hx
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import seaborn; seaborn.set()
from sklearn.linear_model import LogisticRegression
def plotData(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    figure()
    scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    legend(['Admitted', 'Not admitted'])
def plotDecisionBoundary(theta, X, y):
    plotData(X[:,1:3], y)
    plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,1])+2])
    plot_y = (-1.0/theta[2,0])*(theta[1,0]*plot_x + theta[0,0])
    plot(plot_x, plot_y)
    legend([ 'Decision Boundary', 'Admitted', 'Not admitted'])
    #legend(['Decision Boundary'])
    axis([30, 100, 30, 100])

data = np.loadtxt('ex2data1.txt', delimiter = ',')

COLUMN_NUM = 2

X = data[:, 0:COLUMN_NUM]

y = data[:, COLUMN_NUM]

#C->The bigger it is, the smaller the regulization is.
model = LogisticRegression(C=1e5)

model.fit(X,y)

print(model.coef_)
print(model.intercept_)
plotDecisionBoundary(np.array([[model.intercept_[0]],[model.coef_[0,0]],[model.coef_[0,1]]]), np.hstack( (np.ones( (X.shape[0], 1) ), X) ), y[:,np.newaxis])