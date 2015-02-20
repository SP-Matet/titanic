# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 13:56:16 2015

@author: user
"""
from data_loading import *
from preprocessing import *
from numpy import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 

data,X,Y = get_data('train.csv')
print data.head(1)

# split data in training and testing sets
print data.shape[0]
idx = range(0,data.shape[0])
random.shuffle(idx)
training_size = 70*data.shape[0]/100 # 70% for training set

training_data = X[idx[:training_size],:]
training_label = Y[idx[:training_size]]
test_data = X[idx[training_size:],:]
test_label = Y[idx[training_size:]]
print training_data.shape
print test_data.shape

show_ordered_feat(training_data,training_label)
feats = select_feat(4,training_data,training_label)

X = training_data[:,feats]
print X.shape
print training_data.shape

# Create the random forest object 
forest1 = RandomForestClassifier(n_estimators = 3)
# Fit the training data to the Survived labels and create the decision trees
forest1 = forest1.fit(training_data,training_label)
# Take the same decision trees and run it on the test data
prediction1 = forest1.predict(test_data)
train_prediction1 = forest1.predict(training_data)

######### Test with limited number of features
# Create the random forest object 
forest2 = RandomForestClassifier(n_estimators = 30)
# Fit the training data to the Survived labels and create the decision trees
forest2 = forest2.fit(X,training_label)
# Take the same decision trees and run it on the test data
prediction2 = forest2.predict(test_data[:,feats])
train_prediction2 = forest2.predict(X)


print'\n** with all features **'
print 'Results on training values :'
show_results(training_label,train_prediction1)

print '\nResults on test values :'
show_results(test_label,prediction1)


print'\n** with selected numbers of features **'
print 'Results on training values :'
show_results(training_label,train_prediction2)

print '\nResults on test values :'
show_results(test_label,prediction2)