#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:40:25 2019

@author: ariannasmith
"""


def accuracy(treeNode, data):
    correct = 0
    examples = 0
    for example in data:
        result = traverse(example, treeNode)
        if example[len(example) - 1] == result:
            correct += 1
        examples += 1
    accuracy = correct/examples
    return accuracy
    
        
def traverse(example, index):
    if index.value != -1:
        return index.value
    elif example[index.label] == 0:
        return traverse(example, index.left)
    elif example[index.label] == 1:
        return traverse(example, index.right)


