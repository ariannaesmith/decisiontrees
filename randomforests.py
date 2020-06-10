#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ariannasmith
"""

import numpy
from sklearn.ensemble import RandomForestClassifier


def randomForests(fileset):
    """ Runs random forests algorithm on datasets."""
    traindata = numpy.genfromtxt(fileset[0], delimiter = ",")
    testdata = numpy.genfromtxt(fileset[2], delimiter = ",")
    
    # Create X and Y
    trainX = numpy.delete(traindata, (traindata.shape[1] - 1), axis = 1)
    testX = numpy.delete(testdata, (testdata.shape[1] - 1), axis = 1)    
    trainY = traindata[:, (traindata.shape[1] - 1)]
    testY = testdata[:, (testdata.shape[1] - 1)]

    clf = RandomForestClassifier()
    clf = clf.fit(trainX, trainY)
    
    currentscore = clf.score(testX, testY)
    
    print("Random forest score: ", currentscore*100, "%.\n", sep = '')
        