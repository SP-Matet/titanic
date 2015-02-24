# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 13:56:16 2015

@author: user
"""
from data_loading import *
from preprocessing import *
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 

mean_ages = get_mean_ages ()
data,X,Y = get_data(mean_ages)

print data.shape[1], ' features'
print data.head(1)


# split data in training and testing sets
training_size = abs(70*data.shape[0]/100) # 70% for training set
n_times =20
n_estimators = 20
n_min_samples_split = 10

#show_test_idx(n_times,training_size,n_estimators,X,Y)

# split data into training and test sets
idx = range(0,data.shape[0])

random.shuffle(idx)

training_data = X[idx[:training_size],:]
training_label = Y[idx[:training_size]]
test_data = X[idx[training_size:],:]
test_label = Y[idx[training_size:]]

my_idx = [0,1,2,5,8,11,7] # selected features
training_data = training_data[:,my_idx]
test_data = test_data[:,my_idx]

# Create the random forest object 
forest = RandomForestClassifier(n_estimators =100, min_samples_split=n_min_samples_split)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(training_data,training_label)

# Take the same decision trees and run it on the test data
prediction = forest.predict(test_data)
train_pred = forest.predict(training_data)

print '\nResults on test values :'
show_results(test_label,prediction)
print '\n'
print 'Results on training values :'
show_results(training_label,train_pred)
print '\n'



def make_submission(X, Y, features, name, n_min_samples_split):
    data_test,X_test,id = get_test_data('test.csv', mean_ages)

    print data_test.head(1)
    print data_test.shape


    # Create the random forest object
    forest = RandomForestClassifier(n_estimators=100, min_samples_split=n_min_samples_split)

    # Fit the training data to the Survived labels and create the decision trees
    forest = forest.fit(X[:, features], Y)

    # Take the same decision trees and run it on the test data
    result = forest.predict(X_test[:,features])

    print result.shape
    # Copy the results to a pandas dataframe 
    output = pd.DataFrame( data={"PassengerId":id, "Survived":result} )
    # Use pandas to write the comma-separated output file
    output.to_csv( (name +"_model.csv"), index=False, quoting=3 )


make_submission(X,Y,my_idx, 'with_titles', n_min_samples_split)
