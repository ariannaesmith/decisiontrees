#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:50:16 2019

@author: ariannasmith
"""

import testfunction as tf
import numpy


def depth(tree, traindata, validdata):
    """Depth-based pruning function."""
    dmax = [5, 10, 15, 20, 50, 100]
    depthList = []
    changedNodes = []
    accuracyList = []
    setAccuracy = tf.accuracy(tree, validdata)
    dCount = 0
    depthCount(tree, dCount, depthList)
    
    for d in dmax:      
        for n in range(len(depthList)):   
            # If out of range
            if  depthList[n][1] >= d:
                # Add to changed nodes list for removal
                changedNodes.append(depthList[n][0])
                val = find(tree, depthList[n][0], traindata)
                # Change the value so program thinks it is a leaf node
                depthList[n][0].value = val
        dAcc = tf.accuracy(tree, validdata)
        accuracyList.append(dAcc)
        # Change value back to internal node
        for y in changedNodes:
            y.value = -1
    
    # Find maximum accuracy
    maximum = max(accuracyList)
    maxindex = accuracyList.index(maximum)       
    
    if setAccuracy < maximum:
        for n in range(len(depthList)):   
            if  depthList[n][1] >= dmax[maxindex]:
                val = find(tree, depthList[n][0], traindata)
                depthList[n][0].value = val
    
    
def depthCount(node, dCount, depthList):   
    """Returns depth of each node."""
    if node == None:
        return
    if node.value == -1:
        depthList.append([node, dCount])
        dCount += 1
    depthCount(node.left, dCount, depthList)
    depthCount(node.right, dCount, depthList)
    return depthList
     
    
    
def reducePrune(node, traindata, validdata):
    """Reduced error pruning function."""
    accuracyList = []
    internalList = []
    internalList = internalListfunc(node, internalList)

    setAccuracy = tf.accuracy(node, validdata)
    for n in internalList:
        val = find(node, n, traindata)
        n.value = val
    
        # Use temp to "delete" node
        tmpleft = n.left
        tmpright = n.right
        n.left = n.right = None
        
        # Test accuracy with deleted subtree
        delAccuracy = tf.accuracy(node, validdata)
        accuracyList.append([delAccuracy, val])
        # Add node back
        n.left = tmpleft
        n.right = tmpright
        n.value = -1 
        
    maximum = max(accuracyList)
    maxindex = accuracyList.index(maximum)
    
    # If the accuracy increases, repeat
    if maximum[0] >= setAccuracy:
        delnode = internalList[maxindex]
        newclassvalue = accuracyList[maxindex][1]
        delnode.value = newclassvalue
        delnode.left = delnode.right = None
        reducePrune(node, traindata, validdata)
           

def internalListfunc(node, internalList):
    """List of internal nodes."""
    if node == None:
        return
    if node.value == -1:
        internalList.append(node)
    internalListfunc(node.left, internalList)
    internalListfunc(node.right, internalList)
    return internalList
        
def find(root, node, matrix):
    """Find class value majority at given node."""
    # If a leaf node is reached through recursion
    finding = node.label
    if root.value == 0 or root.value == 1:
        return -1
    # For root node    
    elif root.label == node.label:
        return 1     
    else:
        # Break into submatrices where value is 0 or 1
        positive = numpy.where(matrix[:, root.label] == 1)
        negative = numpy.where(matrix[:, root.label] == 0)       
        
        sub1 = matrix[positive]
        sub0 = matrix[negative]
        
        classColumn0 = sub0[:, (matrix.shape[1] - 1)]
        classColumn1 = sub1[:, (matrix.shape[1] - 1)]
        
        val0 = round(classColumn0.sum(axis = 0) / sub0.shape[0])
        val1 = round(classColumn1.sum(axis = 0) / sub1.shape[0])
        
        # Return 0 or 1 depending on if node looking for is left or right child
        if root.left.label == finding:
            return val0
        if root.right.label == finding:
            return val1
    return max([find(root.left, node, sub0), find(root.right, node, sub1)])

