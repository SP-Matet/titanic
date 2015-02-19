# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 13:56:16 2015

@author: user
"""

from data_loading import *
from numpy import *
data,X,Y = get_data('train.csv')

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



