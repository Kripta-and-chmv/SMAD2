import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import math
import scipy.stats as st

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
    Ft = 1.5705
    F = est_sigma_2 / sigma ** 2
    ##########
    #���� ������
    #alpha = 0.05
    #p_value = st.f.cdf(F, 21, float('Inf'))
    #if p_value > alpha:
    #    return False
    #else:
    #    return True
        
    if Ft >= F:
        return True
    else:
        return False

def get_x1_x2(fname):
    str_file = []
    x1 = []
    x2 = []
    with open(fname, 'r') as f:
        for line in f:
            str_file.append(line)
    for i in range(1, len(str_file)):
        s = str_file[i].expandtabs(1).rstrip()
        x1_el, x2_el = s.split('   ')
        x1.append(float(x1_el))
        x2.append(float(x2_el))
    return x1, x2

def get_y(fname):
    str_file = []
    y = []
    with open(fname, 'r') as f:
        for line in f:
            str_file.append(line)
    for i in range(1, len(str_file)):
        s = str_file[i].expandtabs(1).rstrip()
        u, y_el, ej = s.split('   ')
        y.append(float(y_el))
    return y

def get_sigma(fname):
    str_file = []
    with open(fname, 'r') as f:
        for line in f:
            str_file.append(line)
    sigma = float(str_file[1].expandtabs(1).rstrip())
    return sigma