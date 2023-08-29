import numpy as np
from functions import *
from joblib import Parallel, delayed
import os

x_ini = 1.45
x_end = 2.2
y_ini = -0.2
y_end = 1.7
L = 1000

x = np.linspace(x_ini, x_end, L)
y = np.linspace(y_ini, y_end, L)
x, y = np.meshgrid(x, y)
k = 1.5
N = int(5000)

if os.getlogin() == "matheus":
    path = "/home/matheus/Doutorado/Tese/Dados/"
elif os.getlogin() == "jdanilo":
    path = "/home/jdanilo/Matheus/Doutorado/Tese/Dados/"
elif os.getlogin() == "jdsjunior":
    path = "/home/jdsjunior/Matheus/Doutorado/Tese/Dados/"

rte = Parallel(n_jobs=-1)(delayed(RTE)(x[i, j], y[i, j], k, N) for i in range(L) for j in range(L))
rte = np.array(rte).reshape((L, L))
with open(path + "RTE_grid_L=%i_k=%.6f_zoom.dat" % (L, k), "w") as df:
    for i in range(L):
        for j in range(L):
            df.write("%.16f %.16f %.16f\n" % (x[i, j], y[i, j], rte[i, j]))
        df.write("\n")