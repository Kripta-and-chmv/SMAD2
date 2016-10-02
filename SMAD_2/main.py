import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import func as f


N = 25
x1 = np.random.uniform(-1, 1, N)
x2 = np.random.uniform(-1, 1, N)

#f.WritingInFile(['x1', 'x2'], [x1, x2], 'x1x2.txt')

##for i in range(N):
##    x2[i] = 0.3

#zeros = np.zeros(N)

#y_0_x2 = f.FindResponds(zeros, x2, 'u_y_ej_0_x2.txt', N)

#y_x1_0 = f.FindResponds(x1, zeros, 'u_y_ej_x1_0.txt', N)
matr_X = f.create_X_matr(x1, x2, N)
Y, sigma = f.FindResponds(x1, x2, 'u_y_ej_x1_x2.txt', N)
est_tetta = f.parameter_estimation_tetta(np.float32(matr_X), np.float32(Y))
est_y = f.parameter_estimation_y(est_tetta, np.float32(matr_X))
est_e = f.parameter_estimation_error(np.float32(Y), est_y)
est_sigma_2 = f.parameter_estimation_sigma_2(est_e, N)
check = f.check_adequacy_of_the_model(sigma, est_sigma_2)
print(check)
#f.Graph(x1, y_x1_0)
#f.Graph(x2, y_0_x2)