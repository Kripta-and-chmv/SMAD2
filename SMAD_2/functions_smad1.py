import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math

def Func(X, Y):
    return 1 + X - sp.exp(-X ** 2) + Y ** 2

def FindMean(x, y, U):
    N=len(x)
    mean = .0
    for i in range(N):
        U[i] = Func(x[i], y[i])
        mean += U[i]
    mean = mean / N
    return mean

def Graph(x, y):
    p1 = plt.plot(x, y, 'ro')
    plt.show()

def WritingInFile(names, sequences, fileName):
    with open(fileName, 'w') as f:
        for i in range(len(names)):
            f.write(names[i] + ':\t\t\t')
        f.write('\n')
        for j in range(len(sequences[0])):
            for i in range(len(names)):
                f.write(str(sequences[i][j]) + '\t\t\t')
            f.write('\n')

def FindResponds(x1, x2, outputFile, N):
    p = 0.08
    w2 = .0
    dispers = .0
    U = np.zeros(N)
    y = np.zeros(N)
    tr = .0

    mean = FindMean(x1, x2, U)
    
    for i in range(N):        
        tr = U[i] - mean
        w2 += tr ** 2

    w2 = w2 / (N - 1)
    dispers = math.sqrt(p * w2)
    ej = np.random.normal(0, dispers, N)

    for i in range(N):
        y[i] = U[i] + ej[i]
    
    WritingInFile(['sigma'], [[w2]], 'sigma'+outputFile[6:])
    WritingInFile(['U', 'ej', 'y'], [U, ej, y], outputFile)

    return y