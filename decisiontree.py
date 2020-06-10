#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tree structure
"""
import numpy
import heuristics as h

class Node:
    def __init__(self, label, value = -1):
        self.left = None
        self.right = None
        # Label is the attribute index
        self.label = label
        # Values specifies whether node is internal or a leaf (0 or 1)
        self.value = value

        
def DLT(attributesList, matrix, heuristic, currentNode):
    """Builds Decision Learning Tree"""
    #print("am i here", currentNode)
    classColumn = matrix[:, (matrix.shape[1] - 1)]
    # If attributes list is empty
    if not attributesList:
        # Change node value most common value of class column
        currentNode.value = round(classColumn.sum(axis = 0) / matrix.shape[0])
    # If nodes are pure
    elif classColumn.sum(axis = 0) / matrix.shape[0] == 1:
        currentNode.value = 1
    elif classColumn.sum(axis = 0) == 0:
        currentNode.value = 0
    else:
        # Find max gain
        gainList=[]
        for a in attributesList:
            gainList.append(h.gain(a, matrix, heuristic))
        nextAtt = attributesList[gainList.index(max(gainList))]
        
        if currentNode is None:
            #print("where am i")
            currentNode = Node(nextAtt, -1)
        currentNode.label = nextAtt
        for i in [0,1]:
            # Break into submatrices where attribute is 0 or 1
            subAtt = numpy.where(matrix[:, nextAtt] == i)
            submatrix = matrix[subAtt]    
            
            if i == 0:
                attributesList.remove(nextAtt)
                currentNode.left = Node(-10)
                nextNode = currentNode.left
                DLT(attributesList, submatrix, heuristic, nextNode)
                attributesList.append(nextAtt)
            else:
                attributesList.remove(nextAtt)
                currentNode.right = Node(-10)
                nextNode = currentNode.right
                DLT(attributesList, submatrix, heuristic, nextNode)
                attributesList.append(nextAtt)
    return currentNode    
    
