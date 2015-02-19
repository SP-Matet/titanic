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
    age_mean = data['Age'].mean()
    print 'Mean age : ', age_mean,'\n'
    data['AgeWasNull'] = False
    data.loc[(data.Age.isnull()),['Age','AgeWasNull']] = [age_mean,True]
    print data[data['AgeWasNull'] == True].shape # 177 rows were not filled
    
    #split Cabin column and fill blank rows
    data['CabinLetter'] = 'Z'
    data['CabinNumber'] = '0'
    cabs = data.Cabin.str.split()
    #print data.dtypes
    #data.loc[!(data.Cabin.isnull()),['CabinLetter','CabinNumber']] =[data.Cabin]
    for i in np.where(data.Cabin.notnull()):
        cab = cabs[i].str[0]
        print cab.str[1::]
        data['CabinLetter'][i] = cab.str[0]
        data['CabinNumber'][i] = cab.str[1::]
    
    data.loc[(data.CabinNumber == ''),'CabinNumber'] = '0' #some issues to correct
    
    data.CabinNumber.astype(int)
    print data.head()
    
    
    Y = data.Survived.values
    del data['Survived']
    del data['Ticket']
    del data['Name'] # Pas utile a priori
    del data['Cabin']
    del data['PassengerId']    
    
    X = data.values
    X[X == 'male'] = 1
    X[X == 'female'] = -1
    X[X == 'S'] = 0
    X[X == 'C'] = 1
    X[X == 'Q'] = 2
    
    # Que faire avec es âges non renseignés ?

    return data,X, Y