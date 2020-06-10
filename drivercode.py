#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ariannasmith
File to run program from command line
"""
import sys
import numpy

import testfunction as tf
import decisiontree as dlt
import pruning as p
import randomforests as rf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

if sys.argv[1] == "c300":
    if sys.argv[2] == "d100":
        fileset = ["train_c300_d100.csv", "valid_c300_d100.csv", "test_c300_d100.csv"]
    elif sys.argv[2] == "d1000":
        fileset = ["train_c300_d1000.csv", "valid_c300_d1000.csv", "test_c300_d1000.csv"]
    elif sys.argv[2] == "d5000":
        fileset = ["train_c300_d5000.csv", "valid_c300_d5000.csv", "test_c300_d5000.csv"]
elif sys.argv[1] == "c500":
    if sys.argv[2] == "d100":
        fileset = ["train_c500_d100.csv", "valid_c500_d100.csv", "test_c500_d100.csv"]
    elif sys.argv[2] == "d1000":
        fileset = ["train_c500_d1000.csv", "valid_c500_d1000.csv", "test_c500_d1000.csv"]
    elif sys.argv[2] == "d5000":
        fileset = ["train_c500_d5000.csv", "valid_c500_d5000.csv", "test_c500_d5000.csv"]
elif sys.argv[1] == "c1000":
    if sys.argv[2] == "d100":
        fileset = ["train_c1000_d100.csv", "valid_c1000_d100.csv", "test_c1000_d100.csv"]
    elif sys.argv[2] == "d1000":
        fileset = ["train_c1000_d1000.csv", "valid_c1000_d1000.csv", "test_c1000_d1000.csv"]
    elif sys.argv[2] == "d5000":
        fileset = ["train_c1000_d5000.csv", "valid_c1000_d5000.csv", "test_c1000_d5000.csv"]
elif sys.argv[1] == "c1500":
    if sys.argv[2] == "d100":
        fileset = ["train_c1500_d100.csv", "valid_c1500_d100.csv", "test_c1500_d100.csv"]
    elif sys.argv[2] == "d1000":
        fileset = ["train_c1500_d1000.csv", "valid_c1500_d1000.csv", "test_c1500_d1000.csv"]
    elif sys.argv[2] == "d5000":
        fileset = ["train_c1500_d5000.csv", "valid_c1500_d5000.csv", "test_c1500_d5000.csv"]
elif sys.argv[1] == "c1800":
    if sys.argv[2] == "d100":
        fileset = ["train_c1800_d100.csv", "valid_c1800_d100.csv", "test_c1800_d100.csv"]
    elif sys.argv[2] == "d1000":
        fileset = ["train_c1800_d1000.csv", "valid_c1800_d1000.csv", "test_c1800_d1000.csv"]
    elif sys.argv[2] == "d5000":
        fileset = ["train_c1800_d5000.csv", "valid_c1800_d5000.csv", "test_c1800_d5000.csv"]




print("Using data of ", sys.argv[1], ', ', sys.argv[2], ":", sep = '')

if sys.argv[3] == "RF":
    rf.randomForests(fileset)
else:
    impurity = sys.argv[3]
    pruneMethod = sys.argv[4]
    traindata = numpy.genfromtxt(fileset[0], delimiter = ",")
    validdata = numpy.genfromtxt(fileset[1], delimiter = ",")
    testdata = numpy.genfromtxt(fileset[2], delimiter = ",")
    attributes = list(range(0, traindata.shape[1] - 1))
    print("Building tree with", impurity)
    tree = dlt.DLT(attributes, traindata, impurity, currentNode = None)
    acc = tf.accuracy(tree, testdata)
    print("The accuracy of the tree is ", acc*100, "%.", sep = '')
    if pruneMethod != "none":
        print("Pruning with ", pruneMethod, ".", sep = '')
        if pruneMethod == "error":
            p.reducePrune(tree, traindata, validdata)
        else:
            p.depth(tree, traindata, validdata)
        newAcc = tf.accuracy(tree, testdata)
        print("The new accuracy of the tree after pruning is ", newAcc*100, "%.", sep = '')