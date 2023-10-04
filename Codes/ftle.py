import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import numpy as np
import pandas as pd
from functions import *
from string import ascii_lowercase
from joblib import Parallel, delayed
from pyunicorn.timeseries import RecurrencePlot as RP
from numba import njit
from scipy.interpolate import interp1d

k = 1.5
x0 = 3.0
y0 = 0.0

N = int(1e10)
n = 200
exponent = int(np.log10(N))
base = int(N/10**exponent)
ftle = FTLE(x0, y0, k, n, N)

path = "/home/matheus/Doutorado/Tese/Dados/"
datafile = path + 'FTLE_Ntot=%ie%i_n=%i.dat' % (base, exponent, n)
np.savetxt(datafile, ftle)