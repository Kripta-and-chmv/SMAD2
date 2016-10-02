import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math


def func_2(x):
    return x

def func_3(x):
    return sp.exp(-x ** 2)

def func_4(x):
    return x**2

def create_X_matr(x1, x2, N):
    X = []
    for i in range(N):
        X.append([])
        X[i].append(1.)
        X[i].append(func_2(x1[i]))
        X[i].append(func_3(x1[i]))
        X[i].append(func_4(x2[i]))
    return np.array(X)

def parameter_estimation_tetta(matr_X, Y):
    XtX = np.matmul(matr_X.T, matr_X)
    XtX_1 = np.linalg.inv(XtX)
    XtX_1_Xt = np.matmul(XtX_1, matr_X.T)
    est_tetta = np.matmul(XtX_1_Xt, Y)
    return est_tetta

def parameter_estimation_y(est_tetta, matr_X):
    est_y = np.matmul(matr_X, est_tetta)
    return est_y

def parameter_estimation_error(Y, est_y):
    est_e = Y - est_y
    return est_e

def parameter_estimation_sigma_2(est_e, N):
    est_sigma_2 = np.matmul(est_e.T, est_e)/(N - 4)
    return est_sigma_2

def check_adequacy_of_the_model(sigma, est_sigma_2):
    Ft = 1.8117
    F = est_sigma_2 / sigma ** 2
    if Ft >= F:
        return True
    else:
        return False
#################################
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
            f.write(names[i] + ':\n')
            for j in range(len(sequences[i])):
                f.write('\t' + str(sequences[i][j]) + '\n')

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
    
    WritingInFile(['U', 'ej', 'y'], [U, ej, y], outputFile)

    return y, dispers