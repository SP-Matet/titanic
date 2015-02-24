# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 21:59:48 2015

@author: user
"""

import numpy as np

def dumb_learn (X, Y):
    nombre_survivants = np.zeros(6)
    nombre_total = np.zeros(6)
    
    for i in range(X.shape[0]):
        nombre_survivants[X[i,0] -1 + 3*X[i,6]] = nombre_survivants[X[i,0] -1 + 3*X[i,6]] + Y[i]
        nombre_total[X[i,0] -1 + 3*X[i,6]] = nombre_total[X[i,0] -1 + 3*X[i,6]] +1
        
    survie = np.zeros(6)
    for i in range (6):
        if (2*nombre_survivants[i] >= nombre_total[i]):
            survie[i] = 1
    
    return survie
    
def dumb_predict (X, survie):
    Y = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        Y[i] = survie[X[i,0] -1 + 3*X[i,6]]
    
    return Y