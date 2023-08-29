from billiard_functions import *
import numpy as np
from joblib import Parallel, delayed
import os

L = 1000
N = int(5e3)
exponent = int(np.log10(N))
base = int(N/10**exponent)
gamma = [1, 2, 3]
eps = [0.665, 0.836, 0.43]
if os.getlogin() == "jdanilo":
    path = "/home/jdanilo/Matheus/Doutorado/Tese/Dados/"
elif os.getlogin() == "matheus":
    path = "/home/matheus/Doutorado/Tese/Dados/"
elif os.getlogin() == "jdsjunior":
    path = "/home/jdsjunior/Matheus/Doutorado/Tese/Dados/"
else:
    print("Invalid user!")
    import sys
    sys.exit()

for j in range(len(gamma)):
    datafile = path + "billiard_grid_rte_gamma=%i_eps=%.3f_N=%ie%i_L=%i.dat" % (gamma[j], eps[j], base, exponent, L)
    theta0 = np.linspace(0, 2*np.pi, L, endpoint=True)
    alpha0 = np.linspace(0, np.pi, L, endpoint=True)
    s0 = np.zeros(L)
    p0 = np.cos(alpha0)
    Lmax = arc_length(gamma[j], eps[j])
    for i in range(L):
        s0[i] = arc_length(gamma[j], eps[j], theta_end=theta0[i])
    s0 = s0/Lmax
    s0, p0 = np.meshgrid(s0, p0)
    theta0, alpha0 = np.meshgrid(theta0, alpha0)
    rte = Parallel(n_jobs=-1)(delayed(RTE)(theta0[ii, jj], alpha0[ii, jj], gamma[j], eps[j], N) for ii in range(L) for jj in range(L))
    rte = np.array(rte).reshape((L, L))
    with open(datafile, "w") as df:
        for ii in range(L):
            for jj in range(L):
                df.write("%.16f %.16f %.16f\n" % (s0[ii, jj], p0[ii, jj], rte[ii, jj]))
            df.write("\n")