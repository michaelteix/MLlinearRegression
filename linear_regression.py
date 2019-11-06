# Michael Teixeira
# 1001375188

import sys
import numpy as np
from numpy import genfromtxt
import math

assert len(sys.argv) >= 5, "Not enough command line arguments!"

def fi_x(x, degree) :
    result = [1]

    for i in x :
        for j in range(1, degree + 1) :
            result.append(float(i)**float(j))

    result = np.array(result)
    return result

def linear_regression(trainingFile, degree, _lambda, testFile) :
    assert(degree >= 0 and degree <= 10)
    assert(_lambda >= 0)

    ### Traning Phase ###

    tdata = genfromtxt(trainingFile)
    fi = []
    t = []

    for i in tdata :
        fi.append( fi_x( i[:-1], degree ) )
        t.append(i[-1])

    fi = np.array(fi)
    t = np.array([t]).T
    w = np.linalg.pinv( _lambda * np.identity( len(fi[0]) ) + fi.T @ fi ) @ fi.T @ t
    
    for index, i in enumerate(w) :
        print( "w%d=%.4f" % ( index, i ) )

    ### Test Phase ###

    testData = genfromtxt(testFile)
    _fi = []
    true = []
    correct = 0

    print()
    for index, i in enumerate(testData) :
        _fi.append( fi_x( i[:-1], degree ) )
        true.append(i[-1])
        tPrime = w.T @ _fi[index]
        squaredError = ( true[index] - tPrime )**2
        print( "ID=%5d, output=%14.4f, target value = %10.4f, squared error = %.4f" % ( index + 1, tPrime, true[index], squaredError ) )

        if (squaredError < 1) :
            correct +=1
    
    print( "Accuracy = %.2f" % (correct / len(testData)) )
    return

### MAIN ###
linear_regression(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])