import numpy as np
from functions import *

k = 1.5
x0 = -3.0
y0 = 0.0

N = int(1e10)
n = 200
exponent = int(np.log10(N))
base = int(N/10**exponent)
ftle = FTLE(x0, y0, k, n, N)

path = "/home/matheus/Doutorado/Tese/Dados/"
datafile = path + 'FTLE_Ntot=%ie%i_n=%i.dat' % (base, exponent, n)
np.savetxt(datafile, ftle)