import numpy as np
from functions import *
from joblib import Parallel, delayed
import os

x_ini = [-np.pi, -np.pi, -1.5, -1.5, 0.8, 0.83]
x_end = [np.pi, np.pi, 1.5, 1.5, 1.6, 1.47]
y_ini = [-np.pi, -np.pi, -2.5, -2.5, 1.8, -0.6]
y_end = [np.pi, np.pi, 2.5, 2.5, 3.0, 0.6]
L = 1000
x = np.linspace(x_ini, x_end, L)
y = np.linspace(y_ini, y_end, L)
ks = np.array([0.9, 1.5, 3.63, 4.0, 5.3, 6.908745])
N = int(5000)

if os.getlogin() == "matheus":
    path = "/home/matheus/Doutorado/Tese/Dados/"
elif os.getlogin() == "jdanilo":
    path = "/home/jdanilo/Matheus/Doutorado/Tese/Dados/"
elif os.getlogin() == "jdsjunior":
    path = "/home/jdsjunior/Matheus/Doutorado/Tese/Dados/"
for l in range(len(ks)):
    k = ks[l]
    x = np.linspace(x_ini[l], x_end[l], L)
    y = np.linspace(y_ini[l], y_end[l], L)
    x, y = np.meshgrid(x, y)
    rte = Parallel(n_jobs=-1)(delayed(RTE)(x[i, j], y[i, j], k, N) for i in range(L) for j in range(L))
    rte = np.array(rte).reshape((L, L))
    with open(path + "RTE_grid_L=%i_k=%.6f.dat" % (L, k), "w") as df:
        for i in range(L):
            for j in range(L):
                df.write("%.16f %.16f %.16f\n" % (x[i, j], y[i, j], rte[i, j]))
            df.write("\n")