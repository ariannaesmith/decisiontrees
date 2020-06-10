# -*- coding: utf-8 -*-
"""
Homework 1
"""

import numpy
import math



def entropy(column):
    """Calculates entropy of column given as parameter."""
    # get total number of values
    # column is transformed to vector
    size = column.shape[0]
    # get number of positive values
    summation = column.sum(axis = 0)
    if size == 1 or size == summation or summation == 0 or size == 0:
        entropy = 0
    else:
        entropy = (-summation/size*math.log((summation/size), 2)) - \
        ((size-summation)/size*math.log(((size-summation)/size), 2))
    return entropy

def variance(column):
    """Calculates variance of column given as parameter."""
    # get total number of values
    k = column.shape[0]
    # get number of positive and negative values
    k1 = column.sum(axis = 0)
    k0 = k - k1
    variance = (k0/k)*(k1/k)
    return variance

def gain(attribute, submatrix, impurity):
    """Calculates information gain for a given attribute along the path."""
    classColumn = submatrix[:, (submatrix.shape[1] - 1)]
    #print("heuristic is", impurity)
    if impurity == "entropy":
        # need last column for entropy(S)
        impurityS = entropy(classColumn)
    else:
        impurityS = variance(classColumn)
    # break into submatrices of positive and negative attribute values
    positive = numpy.where(submatrix[:, attribute] == 1)
    negative = numpy.where(submatrix[:, attribute] == 0)
    attribute1 = submatrix[positive]
    attribute0 = submatrix[negative]
        
    # Number of positive and negative examples for class
    sum1 = attribute1.shape[0]
    sum0 = attribute0.shape[0]
    sizeS = classColumn.shape[0]
    
    # In case submatrix wasn't calculated properly
    if (sum1 + sum0) != sizeS:
        print("Error!")
        return
    
    # Calculate entropy for postive and negative values of attribute
    classAtt1 = attribute1[:, (attribute1.shape[1] - 1)]
    classAtt0= attribute0[:, (attribute0.shape[1] - 1)]
    if impurity == "entropy":
        impurity1 = entropy(classAtt1)
        impurity0 = entropy(classAtt0)
    else:
        impurity1 = variance(classAtt1)
        impurity0 = variance(classAtt0)
        
    # Gain function    
    attGain = impurityS - sum1/sizeS*impurity1 - sum0/sizeS*impurity0
    return attGain
    

