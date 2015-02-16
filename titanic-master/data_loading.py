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
    
    # Que faire avec les âges non renseignés ?

    return X, Y