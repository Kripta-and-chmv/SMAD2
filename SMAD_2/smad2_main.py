import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import functions_smad2 as f


N = 25
x1, x2 = f.get_x1_x2('x1x2.txt')
Y = f.get_y('u_y_ej_x1_x2.txt')
sigma = f.get_sigma('sigma_x1_x2.txt')


matr_X = f.create_X_matr(x1, x2, N)

est_tetta = f.parameter_estimation_tetta(np.float32(matr_X), np.float32(Y))
est_y = f.parameter_estimation_y(est_tetta, np.float32(matr_X))
est_e = f.parameter_estimation_error(np.float32(Y), est_y)
est_sigma_2 = f.parameter_estimation_sigma_2(est_e, N)
check, F, Ft = f.check_adequacy_of_the_model(sigma, est_sigma_2)
print(check)
f.write_in_file(Y, est_y, est_e, sigma, est_sigma_2, est_tetta, F, Ft)
#f.Graph(x1, y_x1_0)
#f.Graph(x2, y_0_x2)