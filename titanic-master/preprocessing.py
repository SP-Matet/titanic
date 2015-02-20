# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:29:30 2015

@author: user
"""

# Preprocessing / Dimensionality reduction

from tools import *
from numpy import *

def show_ordered_feat(X,Y):
    gainIG = infogain(X,Y)
    gain = chiSQ(X,Y)
    idx = argsort(gain)[::-1]
    idxIG = argsort(gainIG)[::-1]
    print 'ChiSQ order ',idx
    print 'IG order', idxIG
    
def select_feat(k,X,Y):
    gain = chiSQ(X,Y)
    idx = argsort(gain)[::-1]     
    return idx[:k]