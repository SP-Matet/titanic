# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:29:30 2015

@author: user
"""

# Preprocessing / Dimensionality reduction

from tools import *
from numpy import *
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def show_ordered_feat(X,Y):
    gainIG = infogain(X,Y)
    gain = chiSQ(X,Y)
    idx = argsort(gain)[::-1]
    idxIG = argsort(gainIG)[::-1]
    print 'ChiSQ order ',idx
    print 'IG order', idxIG
    
def select_feat_chi(X,Y):
    gain = chiSQ(X,Y)
    idx = argsort(gain)[::-1]     
    return idx

def select_feat_IG(X,Y):
    gain = infogain(X,Y)
    idx = argsort(gain)[::-1]     
    return idx

def show_test_idx(n_times,training_size,n_estimators,X,Y):
    idx_chi = select_feat_chi(X,Y)
    idx_IG = select_feat_IG(X,Y)
    my_idx1 = [7,1,0,4,9,8,2,3,6,5]
    my_idx2 = [7,4,0,1,6,2,9,3,8,5]
    my_idx3 = [7,4,0,2,1,9,6,3,8,5]
    print 'Chi2 :', idx_chi
    print 'IG :', idx_IG
    print 'My idx 1 :', my_idx1
    print 'My idx 2 :', my_idx2
    print 'My idx 3 :', my_idx3
    indexes = [idx_chi,idx_IG,my_idx1,my_idx2,my_idx3]
    print "Training size : " + str(training_size)
    res = test_idx_order(n_times,indexes,training_size,n_estimators,X,Y)    

    x_axis = range(1,len(indexes[0])+1)
    fig = plt.figure(0)
    plt.title('test on index')
    plt.plot(x_axis,res[0],color='r',linewidth=2.0,label='chi')
    plt.plot(x_axis,res[1],color='b',linewidth=2.0,label='IG')
    plt.plot(x_axis,res[2],color='y',linewidth=2.0,label='my1')
    plt.plot(x_axis,res[3],color='g',linewidth=2.0,label='my2')
    plt.plot(x_axis,res[4],color='k',linewidth=2.0,label='my3')
    plt.legend()
    plt.show()
    
    
def test_idx_order(n_times,indexes,training_size,k,X,Y):
    nb_idx = len(indexes)
    results = zeros((nb_idx,n_times,len(indexes[0])), dtype=float)    
    for j in range(n_times):
        idx = range(0,X.shape[0])
        random.shuffle(idx)
        X_train = X[idx[:training_size],:]
        Y_train = Y[idx[:training_size]]
        X_test = X[idx[training_size:],:]
        Y_test = Y[idx[training_size:]]
        for r in range(len(indexes)):
            index = indexes[r]
            for n_feat in range(1,len(index)+1):
                # Create the random forest object 
                forest = RandomForestClassifier(n_estimators = k)
                # Fit the training data to the Survived labels and create the decision trees
                forest = forest.fit(X_train[:,index[:n_feat]],Y_train)
                # Take the same decision trees and run it on the test data
                prediction = forest.predict(X_test[:,index[:n_feat]])
                n = 0
                for i in range(len(prediction)):
                    if( prediction[i] == Y_test[i]):
                        n += 1
                results[r][j][(n_feat-1)] = float(n)/len(prediction)
    res = zeros((nb_idx,len(indexes[0])), dtype=float)        
    for i in range(len(indexes)):
        res[i] = mean(results[i],axis=0)
    return res
    
