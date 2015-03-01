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
    my_idx1 = [12, 5,13,0,1,9,  8, 11, 14,  2,  3, 10,4,  6,  7, 15]
    my_idx2 = [2,  3, 10,4,  6,  7, 15,12, 5,13,0,1,9,  8, 11, 14]
    print 'Chi2 :', idx_chi
    print 'IG :', idx_IG
    print 'My idx 1 :', my_idx1
    print 'My idx 2 :', my_idx2
    indexes = [idx_chi,idx_IG,my_idx1,my_idx2]
    print "Training size : " + str(training_size)
    res = test_idx_order(n_times,indexes,training_size,n_estimators,X,Y)    

    x_axis = range(1,len(indexes[0])+1)
    fig = plt.figure(0)
    plt.title('test on index')
    plt.plot(x_axis,res[0],color='r',linewidth=2.0,label='chi')
    plt.plot(x_axis,res[1],color='b',linewidth=2.0,label='IG')
    plt.plot(x_axis,res[2],color='y',linewidth=2.0,label='my1')
    plt.plot(x_axis,res[3],color='g',linewidth=2.0,label='my2')
    plt.legend()
    plt.show()
    
    
def test_idx_order(n_times,indexes,training_size,k,X,Y):
    nb_idx = len(indexes)
    results = zeros((nb_idx,n_times,len(indexes[0])), dtype=float)
    n_min_samples_split = 30    
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
                forest = RandomForestClassifier(n_estimators =k,max_depth =7,
                                                #max_features = 2, 
                                                min_samples_split=n_min_samples_split)
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

def test_idxes(n_times,indexes,training_size,k,X,Y):
    nb_idx = len(indexes)
    results = zeros((nb_idx,n_times), dtype=float)
    results_on_train = zeros((nb_idx,n_times), dtype=float)
    n_min_samples_split = 40  
    for j in range(n_times):
        idx = range(0,X.shape[0])
        random.shuffle(idx)
        X_train = X[idx[:training_size],:]
        Y_train = Y[idx[:training_size]]
        X_test = X[idx[training_size:],:]
        Y_test = Y[idx[training_size:]]
        for r in range(len(indexes)):
            index = indexes[r]
            # Create the random forest object 
            forest = RandomForestClassifier(n_estimators =k,max_depth =3,\
                max_features = 2, 
                min_samples_split=n_min_samples_split)
            # Fit the training data to the Survived labels and create the decision trees
            forest = forest.fit(X_train[:,index],Y_train)
            # Take the same decision trees and run it on the test data
            prediction = forest.predict(X_test[:,index])
            pred_on_train = forest.predict(X_train[:,index])
            #○print 'Forest wieghts : ', forest.feature_importances_
            n = 0
            m = 0
            for i in range(len(prediction)):
                if( prediction[i] == Y_test[i]):
                    n += 1
            for i in range(len(pred_on_train)):
                if( pred_on_train[i] == Y_train[i]):
                    m += 1
            results[r][j] = float(n)/len(prediction)
            results_on_train[r][j] = float(m)/len(pred_on_train)
    res = zeros((nb_idx,2), dtype=float)        
    for i in range(len(indexes)):
        res[i] = [mean(results[i],axis=0) ,mean(results_on_train[i],axis=0)]
    return res
    
    
def test_idx(n_times,index,training_size,k,X,Y):
    results = zeros((n_times), dtype=float)
    results_nullAge = zeros((n_times), dtype=float)
    results_nullFare = zeros((n_times), dtype=float)
    n_min_samples_split = 10    
    for j in range(n_times):
        idx = range(0,X.shape[0])
        random.shuffle(idx)
        X_train = X[idx[:training_size],:]
        Y_train = Y[idx[:training_size]]
        X_test = X[idx[training_size:],:]
        Y_test = Y[idx[training_size:]]
        X_nullAge = X_test[where(X_test[:,6])]
        Y_nullAge = Y_test[where(X_test[:,6])]
        X_nullFare = X_test[where(X_test[:,7])]
        Y_nullFare = Y_test[where(X_test[:,7])]
        # Create the random forest object 
        forest = RandomForestClassifier(n_estimators =k,max_depth =3,max_features = 2, min_samples_split=n_min_samples_split)
        # Fit the training data to the Survived labels and create the decision trees
        forest = forest.fit(X_train[:,index],Y_train)
        # Take the same decision trees and run it on the test data
        prediction = forest.predict(X_test[:,index])
        pred_on_null_age = forest.predict(X_nullAge[:,index])
        pred_on_null_fare = forest.predict(X_nullFare[:,index])
        n = 0
        m=0
        s =0
        for i in range(len(prediction)):
            if( prediction[i] == Y_test[i]):
                n += 1
        for i in range(len(pred_on_null_age)):
            if( pred_on_null_age[i] == Y_nullAge[i]):
                m += 1        
        for i in range(len(pred_on_null_fare)):
            if( pred_on_null_fare[i] == Y_nullFare[i]):
                s += 1                
        results[j] = float(n)/len(prediction)
        results_nullAge[j] = float(m)/len(pred_on_null_age) 
        results_nullFare[j] = float(s)/len(pred_on_null_fare) 
    res = mean(results,axis=0)
    print 'On null ages : ', mean(results_nullAge,axis=0)
    print 'On null fares : ', mean(results_nullFare,axis=0)
    return res