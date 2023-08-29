import numpy as np
from functions import *
from joblib import Parallel, delayed
import os

x0 = 0.0
y_ini = -np.pi
y_end = np.pi
k_ini = 0.0
k_end = 5.0
L = 1000

k = np.linspace(k_ini, k_end, L)
y = np.linspace(y_ini, y_end, L)
k, y = np.meshgrid(k, y)
N = int(5000)

if os.getlogin() == "matheus":
    path = "/home/matheus/Doutorado/Tese/Dados/"
elif os.getlogin() == "jdanilo":
    path = "/home/jdanilo/Matheus/Doutorado/Tese/Dados/"
elif os.getlogin() == "jdsjunior":
    path = "/home/jdsjunior/Matheus/Doutorado/Tese/Dados/"

rte = Parallel(n_jobs=-1)(delayed(RTE)(x0, y[i, j], k[i, j], N) for i in range(L) for j in range(L))
rte = np.array(rte).reshape((L, L))
with open(path + "RTE_CGBD_L=%i_k=%.6f.dat" % (L, k), "w") as df:
    for i in range(L):
        for j in range(L):
            df.write("%.16f %.16f %.16f\n" % (k[i, j], y[i, j], rte[i, j]))
        df.write("\n")