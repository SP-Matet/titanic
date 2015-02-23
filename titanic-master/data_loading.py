# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:28:38 2015

@author: user
"""

# Load data

import pandas as pd
import numpy as np
from tools import substrings_in_string
from tools import change_titles

def get_data ():
    data = pd.read_csv('train.csv')
    
    print "Size of the data: ", data.shape

    # Treating missing ages
    # Inspired from other sources
    titles=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
    data['Title']=data['Name'].map(lambda x: substrings_in_string(x, titles))
    data['Title'] = data.apply(change_titles, axis=1)
    
    data['AgeWasNull'] = False
    data.loc[(data.Age.isnull()),'Age'] = True
    
    # Compute mean ages
    mean_ages = np.zeros(4)
    count = np.zeros(4)
    for i in range (data.shape[0]):
        if (not np.isnan(data.Age[i])):
            if (data.Title[i] in ['Mr']):
                mean_ages[0] = mean_ages[0] + data.Age[i]
                count[0] = count[0] + 1
            elif (data.Title[i] in ['Mrs']):
                mean_ages[1] = mean_ages[1] + data.Age[i]
                count[1] = count[1] + 1
            elif data.Title[i] in ['Miss']:
                mean_ages[2] = mean_ages[2] + data.Age[i]
                count[2] = count[2] + 1
            elif data.Title[i] in ['Master']:
                mean_ages[3] = mean_ages[3] + data.Age[i]
                count[3] = count[3] + 1
            else:
                print 'Title not found : ' + data.Title[i]
    mean_ages = np.divide(mean_ages, count)
    print mean_ages
    
    # Fill in blanks
    for i in range (data.shape[0]):
        if (np.isnan(data.Age[i])):
            if (data.Title[i] in ['Mr']):
                data.Age[i] = mean_ages[0]
            elif (data.Title[i] in ['Mrs']):
                data.Age[i] = mean_ages[1]
            elif data.Title[i] in ['Miss']:
                data.Age[i] = mean_ages[2]
            elif data.Title[i] in ['Master']:
                data.Age[i] = mean_ages[3]
            else:
                print 'Title not found : ' + data.Title[i]
                
    del data['Title']
        
    # split Cabin column and fill blank rows by Z0
    data['CabinLetter'] = 'Z'
    data['CabinNumber'] = '0'
    cabs = data.Cabin.str.split()
    for i in np.where(data.Cabin.notnull()):
        cab = cabs[i].str[0]
        data['CabinLetter'][i] = cab.str[0]
        data['CabinNumber'][i] = cab.str[1::]
    
    data.loc[(data.CabinNumber == ''),'CabinNumber'] = '0' #some issues to correct    
    data.CabinNumber.astype(int)
    
    # Fill NaA values for Embarked
    # 644 'S' - 77 'Q' - 168 'C' - 2 NaN => replace NaN by S
    data.loc[(data.Embarked.isnull()),'Embarked'] = 'S'
    
#==============================================================================
# Check that all other field don't have NaN values
#     print data[data.Pclass.isnull()].shape    
#     print data[data.Sex.isnull()].shape
#     print data[data.SibSp.isnull()].shape
#     print data[data.Parch.isnull()].shape
#     print data[data.Fare.isnull()].shape    
#==============================================================================

    #Convert string into int
    data['Gender'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    data['Embarkader'] = data['Embarked'].map({'S':0, 'C' : 1 , 'Q' : 2}).astype(int)
    data['CabinRange'] = data['CabinLetter'].map({'A':0, 'C':1, 'B':2, 'E':3, 'D':4, 'G':5, 'F':6, 'T':7, 'Z':8})    
    data['MeanFare'] = np.divide(data.Fare, data.SibSp + data.Parch + 1)
    
    Y = data.Survived.values
    del data['Survived']
    del data['Ticket']
    del data['Name'] # Pas utile a priori
    del data['Cabin']
    del data['PassengerId']    
    del data['CabinLetter']
    del data['Embarked']
    del data['Sex']
    del data['Fare']
    
    X = data.values
    
    return data,X, Y