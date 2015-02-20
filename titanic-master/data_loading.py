# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:28:38 2015

@author: user
"""

# Load data

import pandas as pd
import numpy as np

def get_data (path):
    data = pd.read_csv(path)
    print "Size of the data: ", data.shape

    # setting by default the mean age    
    # add column AgeWasNull to remember null values
    age_mean = data['Age'].mean()
    print 'Mean age : ', age_mean,'\n'
    data['AgeWasNull'] = False
    data.loc[(data.Age.isnull()),['Age','AgeWasNull']] = [age_mean,True]
    #print data[data['AgeWasNull'] == True].shape # 177 rows were not filled
    

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
    
    #Fill NaA values for Embarked
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
        
    Y = data.Survived.values
    del data['Survived']
    del data['Ticket']
    del data['Name'] # Pas utile a priori
    del data['Cabin']
    del data['PassengerId']    
    del data['CabinLetter']
    del data['Embarked']
    del data['Sex']
    
    X = data.values
    
    return data,X, Y