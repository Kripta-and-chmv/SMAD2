import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import functions_smad1 as f


N = 25
x1 = np.random.uniform(-1, 1, N)
x2 = np.random.uniform(-1, 1, N)

f.WritingInFile(['x1', 'x2'], [x1, x2], 'x1x2.txt')

#for i in range(N):
#    x2[i] = 0.3

zeros = np.zeros(N)

y_0_x2 = f.FindResponds(zeros, x2, 'u_y_ej_0_x2.txt', N)

y_x1_0 = f.FindResponds(x1, zeros, 'u_y_ej_x1_0.txt', N)

f.FindResponds(x1, x2, 'u_y_ej_x1_x2.txt', N)

f.Graph(x1, y_x1_0)
f.Graph(x2, y_0_x2)