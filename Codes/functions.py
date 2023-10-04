import numpy as np # NumPy module
from numba import vectorize, njit # Numba module to create fast functions
from pyunicorn.timeseries import RecurrencePlot as RP # Pyunicorn module to create the recurrence plots
# Plotting modules
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

@njit
def time_series_stdmap(x0: float, y0: float, k: float, N: int) -> np.ndarray:
    """
    Generates a time series of points using the standard map model. This function uses Numba's `njit` decorator for performance optimization.

    Parameters
    ----------
    x0 : float
        The initial x-coordinate of the map.
    y0 : float
        The initial y-coordinate of the map.
    k : float
        The nonlinearity parameter of the map.
    N : int
        The number of iterations of the map.

    Returns
    -------
    out : ndarray
        The time series, with shape (N + 1, 2)

    Examples
    --------
    >>> x0 = 0.1
    >>> y0 = 0.1
    >>> k = 0.1
    >>> N = 10000
    >>> u = time_series_stdmap(x0, y0, k, N)

    Notes
    -----
    The standard map is a two-dimensional, nonlinear, area-preserving map that is often used as a simple model for Hamiltonian systems.
    In this function, the map is iterated `N` times, with the result stored in the `map_iterates` array. The values of x and y are
    limited to the interval [0, 2pi].

    """
    u = np.zeros((N + 1, 2))
    u[0, 0] = x0
    u[0, 1] = y0

    for i in range(1, N + 1):
        u[i, 1] = (u[i - 1, 1] - k*np.sin(u[i - 1, 0])) % (2*np.pi)
        u[i, 0] = (u[i - 1, 0] + u[i, 1]) % (2*np.pi)

        if u[i, 0] < 0: u[i, 0] += 2*np.pi
        if u[i, 0] > np.pi: u[i, 0] -= 2*np.pi
        if u[i, 1] < 0: u[i, 1] += 2*np.pi
        if u[i, 1] > np.pi: u[i, 1] -= 2*np.pi
        
    return u

@njit
def time_series_stdmap_cylinder(x0: float, y0: float, k: float, N: int, n: int) -> np.ndarray:
    """
    Generates a time series of points using the standard map model. It samples the time series every `n` iterations.
    
    This function uses Numba's `njit` decorator for performance optimization.

    Parameters
    ----------
    x0 : float
        The initial x-coordinate of the map.
    y0 : float
        The initial y-coordinate of the map.
    k : float
        The nonlinearity parameter of the map.
    N : int
        The number of iterations of the map.

    Returns
    -------
    out : ndarray
        The time series, with shape (N + 1, 2)

    Examples
    --------
    >>> x0 = 0.1
    >>> y0 = 0.1
    >>> k = 0.1
    >>> N = 10000
    >>> u = time_series_stdmap(x0, y0, k, N)

    Notes
    -----
    The standard map is a two-dimensional, nonlinear, area-preserving map that is often used as a simple model for Hamiltonian systems.
    In this function, the map is iterated `N` times, with the result stored in the `map_iterates` array. The values of x are limited
    to the interval [0, 2pi].

    """
    m = round(N/n)
    u = np.zeros((m, 3))
    x = x0
    y = y0
    count = 0
    for i in range(1, N + 1):
        y = y - k*np.sin(x)
        x = (x + y) % (2*np.pi)
        if i % n == 0:
            u[count, 0] = i
            u[count, 1] = x
            u[count, 2] = y
            count += 1
        
    return u

@njit
def lyapunov(x0: float, y0: float, k: float, N: int) -> np.ndarray:
    """
    Calculate the Lyapunov exponents of the standard map.

    This function uses Numba's `njit` decorator for performance optimization.

    Parameters
    ----------
    x0 : float
        The initial x-coordinate of the map.
    y0 : float
        The initial y-coordinate of the map.
    k : float
        The constant parameter of the map.
    N : int
        The number of iterations of the map.

    Returns
    -------
    out : ndarray
        The Lyapunov exponent of the map and the last position.

    Notes
    -----
    The Lyapunov exponent is a measure of the exponential rate of separation of nearby trajectories in a dynamical system. In this function, the standard map is iterated `N` times, and the Lyapunov exponent is calculated as the average of the logarithm of the magnitude of the upper triangular matrix (Jacobian matrix) element 1,1 divided by `N` (Eckmann-Ruelle method [1]).

    References
    ----------

    [1] J.-P. Eckmann and D. Ruelle, “Ergodic theory of chaos and strange attractors,” Reviews Modern Physics 57, 617–656 (1985).

    """
    x = x0
    y = y0
    J = np.ones((2, 2), dtype=np.float64)
    beta0 = 0
    sumT11 = 0
    sumT22 = 0
    lypnv = np.zeros(2)
    for i in range(N):
        # Iterates the map
        y = (y - k*np.sin(x)) % (2*np.pi)
        x = (x + y) % (2*np.pi)
        # Jacobian matrix
        J[0, 0] = 1.0 - k*np.cos(x)
        J[1, 0] = -k*np.cos(x)
        # Rotation angle
        beta = np.arctan((-J[1,0]*np.cos(beta0) + J[1,1]*np.sin(beta0))/(J[0,0]*np.cos(beta0) - J[0,1]*np.sin(beta0)))
        # Diagonal elements of the upper triangular matrix
        T11 = np.cos(beta0)*(J[0,0]*np.cos(beta) - J[1,0]*np.sin(beta)) - np.sin(beta0)*(J[0,1]*np.cos(beta) - J[1,1]*np.sin(beta))
        T22 = np.sin(beta0)*(J[0,0]*np.sin(beta) + J[1,0]*np.cos(beta)) + np.cos(beta0)*(J[0,1]*np.sin(beta) + J[1,1]*np.cos(beta))
        sumT11 += np.log(abs(T11))/np.log(2)
        sumT22 += np.log(abs(T22))/np.log(2)
        # Update the rotation angle
        beta0 = beta
    
    lypnv[0] = sumT11/N
    lypnv[1] = sumT22/N

    return lypnv[0], lypnv[1], x, y

@njit
def lyapunov_vs_n(x0: float, y0: float, k: float, N: int) -> np.ndarray:
    """
    Calculate the Lyapunov exponent of the standard map.

    This function uses a vectorized implementation to optimize performance.

    Parameters
    ----------
    x0 : float
        The initial x-coordinate of the map.
    y0 : float
        The initial y-coordinate of the map.
    k : float
        The constant parameter of the map.
    N : int
        The number of iterations of the map.

    Returns
    -------
    out : float
        The Lyapunov exponent of the map.


    Notes
    -----
    The Lyapunov exponent is a measure of the exponential rate of separation of nearby trajectories in a dynamical system. In this function, the standard map is iterated `N` times, and the Lyapunov exponent is calculated as the average of the logarithm of the magnitude of the upper triangular matrix (Jacobian matrix) element 1,1 divided by `N` (Eckmann-Ruelle method [1]).

    References
    ----------

    [1] J.-P. Eckmann and D. Ruelle, “Ergodic theory of chaos and strange attractors,” Reviews Modern Physics 57, 617–656 (1985).

    """
    x = x0
    y = y0
    J = np.ones((2, 2), dtype=np.float64)
    beta0 = 0
    sumT = np.zeros(2)
    aux_lypnv = np.zeros(2)
    lypnv_vs_n = np.zeros((N, 2))

    for i in range(1, N + 1):
        # Iterates the map
        y = (y - k*np.sin(x)) % (2*np.pi)
        x = (x + y) % (2*np.pi)
        # Jacobian matrix
        J[0, 1] = -k*np.cos(x)
        J[1, 1] = 1.0 - k*np.cos(x)
        # Rotation angle
        beta = np.arctan((-J[1,0]*np.cos(beta0) + J[1,1]*np.sin(beta0))/(J[0,0]*np.cos(beta0) - J[0,1]*np.sin(beta0)))
        # Diagonal elements of the upper triangular matrix
        T11 = np.cos(beta0)*(J[0,0]*np.cos(beta) - J[1,0]*np.sin(beta)) - np.sin(beta0)*(J[0,1]*np.cos(beta) - J[1,1]*np.sin(beta))
        T22 = np.sin(beta0)*(J[0,0]*np.sin(beta) + J[1,0]*np.cos(beta)) + np.cos(beta0)*(J[0,1]*np.sin(beta) + J[1,1]*np.cos(beta))
        sumT[0] = np.log(abs(T11))
        sumT[1] = np.log(abs(T22))
        # Update the rotation angle
        beta0 = beta
        aux_lypnv += sumT/np.log(2)
        lypnv_vs_n[i - 1, :] = aux_lypnv/i

    return lypnv_vs_n
   
#@njit
def FTLE(x0:float, y0: float, k:float, n: int, Ntot: int) -> np.ndarray:
    """
    Compute the Finite-Time Lyapunov Exponents (FTLE) for a given set of parameters.

    This function uses Numba's `njit` decorator for performance optimization.

    Parameters
    ----------
    x0 : float
        The initial x-coordinate of the map.
    y0 : float
        The initial y-coordinate of the map.
    k : float
        The constant parameter of the map.
    n : int
        The size of the finite-time windows
    Ntot : int
        The number of iterations of the map.

    Returns
    -------
    np.ndarray: Array of FTLE values computed at each measurement point. The array has shape (N, 2),
                    where N is the total number of measurements, and each row represents the FTLE values,
                    with (:, 0) being the largest Lyapunov exponent.
    """
    N = round(Ntot/n)
    ftle = np.zeros((N, 2))
    x = x0
    y = y0
    for i in range(N):
        ftle[i, 0], ftle[i, 1], x, y = lyapunov(x, y, k, n)
    
    return ftle


@vectorize(['f8(f8, f8, f8, i8)', 'f4(f4, f4, f4, i4)'],
           target='parallel',
           nopython=True)
def dig(x0, y0, k, N):
    """
    Compute the dig measure for a given set of parameters.

    This function uses a vectorized implementation to optimize performance.

    Parameters
    ----------
    x0 : float
        The initial x-coordinate of the map.
    y0 : float
        The initial y-coordinate of the map.
    k : float
        The constant parameter of the map.
    N : int
        The number of iterations of the map.

    Returns
    -------
    
    float: The computed dig measure.
    """
    u = np.arange(1, N)/N
    S = sum(np.exp(-1/(u*(1 - u))))

    # WB for the first N iterates
    ts = time_series_stdmap(x0, y0, k, N)
    w = np.exp(-1/(u*(1 - u)))/S
    WB0 = sum(w*np.cos(ts[1:-1, 0]))
    # WB for the second N iterates
    x0 = ts[-1, 0]
    y0 = ts[-1, 1]
    ts = time_series_stdmap(x0, y0, k, N)
    w = np.exp(-1/(u*(1 - u)))/S
    WB1 = sum(w*np.cos(ts[1:-1, 0]))

    return -np.log10(abs(WB0 - WB1))

@njit
def diffusion_coef(x_lims: np.ndarray, y_lims: np.ndarray, nx: int, ny: int, k: float, N: int) -> float:
    """
    Calculate the diffusion coefficient for a set of initial conditions uniformly distributed
    in phase space given the limits `x_lims` and `y_lims`.

    This function uses Numba's `njit` decorator for performance optimization.


    Parameters
    ----------
    x_lims : np.ndarray
        Array containing the lower and upper limits of the x values
    y_lims : np.ndarray
        Array containing the lower and upper limits of the y values
    nx : int
        Number of points the x-axis
    ny : int
        Number of points in the y-axis
    k : float
        The constant parameter of the map.
    N : int
        The number of iterations of the map.

    Returns
    --------
    float: The computed diffusion coefficient.

    Notes
    -----
    The diffusion coefficient measures how much a quantity spreads out over time due to random processes.
    In this context, it quantifies the spreading of y-values over N iterations in a 2D grid of (x, y) points.
    """
    D = 0
    for i in range(nx):
        x = x_lims[0] + (x_lims[1] - x_lims[0])*i/(nx - 1)
        for j in range(ny):
            y0 = y_lims[0] + (y_lims[1] - y_lims[0])*j/(ny - 1)
            y = y0
            for l in range(N):
                y = y - k*np.sin(x)
                x = (x + y) % (2*np.pi)
            D += (y - y0)**2
    
    D /= int(nx*ny)
    D /= N

    return D

@njit
def ymean2_vs_k(k: float, N: int, n: int, n_icx: int, n_icy: int, xlims: np.ndarray, ylims: np.ndarray) -> np.ndarray:
    """
    Compute the mean squared displacement of the y-coordinate over different initial conditions as a function of time.

    Parameters
    -----------
    k : float
        The constant parameter of the map.
    N : int
        The number of iterations of the map.
    n : int
        The window size in which the mean squared displacement will be computed. n = 1 means that
        the function will return the mean squared displacement for each time step.
    n_icx : int
        Number of initial conditions along the x-axis.
    n_icy : int
        Number of initial conditions along the y-axis.
    xlims : np.ndarray
        Array containing the lower and upper limits of the x values.
    ylims : np.ndarray
        Array containing the lower and upper limits of the y values.

    Returns
    -------
    np.ndarray: Array of mean squared displacement values computed over different initial conditions.

    Notes
    -----
    This function computes the mean squared displacement of the y-coordinate using a specified number
    of initial conditions and parameters. It involves iterating the system and accumulating the squared
    differences between the y-coordinates and their initial values.
    """
    x_ini, x_end = xlims
    y_ini, y_end = ylims
    ymean2 = np.zeros(round(N/n))
    for j in range(n_icx):
        x0 = x_ini + (x_end - x_ini)*j/n_icx
        for l in range(n_icy):
            y0 = y_ini + (y_end - y_ini)*l/n_icy
            ts = time_series_stdmap_cylinder(x0, y0, k, N, n)
            ymean2[:] += (ts[:, 2] - y0)**2
    ymean2[:] /= (n_icx*n_icy)
    return ymean2

def unc_frac_PS(eps: np.ndarray, num_ic: int, k: float, N: int, x0: float = -np.pi, x1: float = np.pi, y0: float = -np.pi, y1: float = np.pi, threshold: float = 11.25, return_UIC: bool = False) -> np.ndarray:
    """
    Calculate the uncertainty fraction for an uncertainty of eps on the phase space.

    Parameters
    ------------
    eps : np.ndarray
        Array containing the uncertainty in the initial condition.
    num_ic : int
        Number of initial condtions used in the calculation of the uncertainty fraction.
    k : float
        Standard map's nonlinearity parameter.
    N : int
        Number of iterations.
    x0 : float, optional
        Lower limit for the x variable (default is -pi).
    x1 : float, optional
        Upper limit for the x variable (default is pi).
    y0 : float, optional
        Lower limit for the y variable (default is -pi).
    y1 : float, optional
        Upper limit for the y variable (default is pi).
    threshold : float, optional
        Cutoff value for dig to distinguish between chaotic and regular orbits (default is 11.25)
    return_UIC : boolean, optional
        If True, the function will also return the uncertain initial conditions as two arrays,
        one for the x positions e another with the y positions. UIC.shape = (eps.shape[0], num_ic) (default is False).
    
    Returns
    ------------
    out : np.ndarray
        The uncertainty fraction as a function of eps.
    """
    if return_UIC == False:
        f = np.zeros(eps.shape[0])
        for i in range(eps.shape[0]):
            # Generates the random initial conditions
            x = np.random.uniform(x0, x1, num_ic)
            y = np.random.uniform(y0, y1, num_ic)
            """ Initial condition #1 """
            xi = x
            yi = y
            fs1 = dig(xi, yi, k, N)
            fs1 = np.where(fs1 >= threshold, 1, 0) # If dig > threshold (regular), fs = 1, else (chaotic) fs = 0.
            """ Initial condition #2 """
            xi = x + eps[i]
            yi = y
            fs2 = dig(xi, yi, k, N)
            fs2 = np.where(fs2 >= threshold, 1, 0) # If dig > threshold (regular), fs = 1, else (chaotic) fs = 0.
            """ Initial condition #3 """
            xi = x - eps[i]
            yi = y
            fs3 = dig(xi, yi, k, N)
            fs3 = np.where(fs3 >= threshold, 1, 0) # If dig > threshold (regular), fs = 1, else (chaotic) fs = 0.
            # Obtains the uncertainty fraction
            f[i] = calc_f(fs1, fs2, fs3)
        return f
    else:
        f = np.zeros(eps.shape[0])
        UIC_x = np.zeros((eps.shape[0], num_ic))
        UIC_y = np.zeros((eps.shape[0], num_ic))
        for i in range(eps.shape[0]):
            # Generates the random initial conditions
            x = np.random.uniform(x0, x1, num_ic)
            y = np.random.uniform(y0, y1, num_ic)
            UIC_x[i,:] = x
            UIC_y[i,:] = y
            """ Initial condition #1 """
            xi = x
            yi = y
            fs1 = dig(xi, yi, k, N)
            fs1 = np.where(fs1 >= threshold, 1, 0) # If dig > threshold (regular), fs = 1, else (chaotic) fs = 0.
            """ Initial condition #2 """
            xi = x + eps[i]
            yi = y
            fs2 = dig(xi, yi, k, N)
            fs2 = np.where(fs2 >= threshold, 1, 0) # If dig > threshold (regular), fs = 1, else (chaotic) fs = 0.
            """ Initial condition #3 """
            xi = x - eps[i]
            yi = y
            fs3 = dig(xi, yi, k, N)
            fs3 = np.where(fs3 >= threshold, 1, 0) # If dig > threshold (regular), fs = 1, else (chaotic) fs = 0.
            # Obtains the uncertainty fraction and the uncertain initial conditions
            f[i] = calc_f(fs1, fs2, fs3, UIC_x[i], UIC_y[i])
        return f, UIC_x, UIC_y
    
@njit(parallel=True)
def calc_f(fs1: np.ndarray, fs2: np.ndarray, fs3: np.ndarray, x: np.ndarray = None, y: np.ndarray = None):
    """
    Calculate the uncertainty fraction given the result of the original initial condition and the
    other two perturbed initial contions.

    Parameters
    ------------
    fs1, fs2, fs3 : np.ndarray
        Final state of the original IC and the perturbed ICs
    x : float, optional
        x coordinate of the uncertain initial condition. If given, the positions where the IC is not uncertain will be overwrite with the value -100.
    y : float, optional
        y coordinate of the uncertain initial condition. If given, the positions where the IC is not uncertain will be overwrite with the value -100.
    Returns
    ------------
    out : float
        The uncertainty fraction.
    """
    num_ic = len(fs1)
    f = 0
    if x == None and y == None:
        for j in range(num_ic):
            if fs1[j] != fs2[j] or fs1[j] != fs3[j]:
                f += 1
    else:
        for j in range(num_ic):
            if fs1[j] != fs2[j] or fs1[j] != fs3[j]:
                f += 1
            else:
                x[j] = -100
                y[j] = -100

    return f/num_ic

@njit
def white_vertline_distr(recmat: np.ndarray) -> np.ndarray:
    """
    Calculate the distribution of the lengths of white vertical lines in a binary matrix.

    Parameters
    ----------
    recmat : np.ndarray
        A 2-dimensional binary numpy array (recurrence matrix).

    Returns
    -------
    out: np.ndarray
        An array containing the count of white vertical lines for each length.
        
    Examples
    --------
    >>> recmat = np.array([[0, 0, 0, 1, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0]])
    >>> white_vertline_distr(recmat)
    array([3., 2., 1.])

    Notes
    -----
    The input binary matrix is assumed to be a square matrix of size N x N and the lines on the border are excluded [1].

    References
    ----------
    [1] K. H. Kraemer and N. Marwan, Border effect corrections for diagonal line based recurrence quantification analysis measures, Physics Letters A 383, 125977 (2019)
    """
    N = recmat.shape[0]
    P = np.zeros(N)
    for i in range(N):
        k = 0 # Counts the length of the white lines
        l = 0 # Checks if the white line is not on the border
        for j in range(N):
            if recmat[i, j] == 0 and l != 0:
                k += 1
            else:
                if k != 0:
                    P[k] += 1
                    k = 0
                elif recmat[i, j] == 1:
                    l = 1
    
    return P

def RTE(x0: float, y0: float, k: float, T: int, metric: str ='supremum', lmin: int = 1, eps: float = 10/100, threshold_eps: bool = True, approach: str = "maximum", return_last_pos: bool = False) -> any:
    """
    Return the recurrence time entropy (RTE) [1-3] given an initial condition (x0, y0) considering border effects [4] when evaluating the distribution of white vertical lines.
    The standard deviation is calculated using the selected approach described in the Appendix A (default is maximum).

    Parameters
    ----------
    x0 : float
        The initial value of the x-coordinate.
    y0 : float
        The initial value of the y-coordinate.
    k : float
        The non-linearity parameter of the map.
    T : int
        The number of iterations (length of the orbit).
    metric : str, optional
        The metric for measuring distances in phase space. Possible values are 'supremum', 'manhattan', 'euclidean' (default='supremum').
    lmin : int, optional
        Minimal length of white vertical lines used in the RTE computation (default=1).
    eps : float, optional
        Threshold for the recurrence plot in units of time series standard deviation if threshold_std=True. It is the threshold if threshold_std=False (default=10/100).
    threshold_eps : bool, optional
        If True, generates the recurrence plot using a fixed threshold in units of the time series standard deviation (default=True)
    approach : str, optional
        The approach for calculating the standard deviation. Possible values are 'concatenation', 'maximum', 'euclidean' (default='maximum')
    return_last_pos : bool, optional
        If True, also return the last position of the orbit. (default=False).

    Returns
    -------
    out : tuple or float
        If `return_last_pos=False` (default), returns the RTE. 
        If `return_last_pos=True`, returns a tuple (RTE, x, y), where `x` and `y` are the last position in phase space.

    References
    ----------
    [1] http://www.pik-potsdam.de/~donges/pyunicorn/api/timeseries/recurrence_plot.html
    [2] M. A. Little et al., Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection, BioMedical Engineering OnLine 6, 23 (2007)
    [3] K. H. Kraemer et al., Recurrence threshold selection for obtaining robust recurrence characteristics in different embedding dimensions, Chaos 28, 085720 (2018)
    [4] K. H. Kraemer and N. Marwan, Border effect corrections for diagonal line based recurrence quantification analysis measures, Physics Letters A 383, 125977 (2019)
    """
    time_series = time_series_stdmap(x0, y0, k, T)

    if threshold_eps:
        if approach == "concatenation":
            eps = time_series.std()*eps
        elif approach == "maximum":
            eps = max(np.std(time_series[:, 0]), np.std(time_series[:, 1]))*eps
        elif approach == "euclidean":
            eps = np.sqrt(np.std(time_series[:, 0])**2 + np.std(time_series[:, 1])**2)*eps
        else:
            print("Invalid approach!")
            sys.exit()
    rp = RP(time_series, metric=metric, normalize=False, threshold=eps, silence_level=2)
    recmat = rp.recurrence_matrix()
    p = white_vertline_distr(recmat)
    p = p[lmin:]
    p = np.extract(p != 0, p)

    p_normed = p/p.sum()

    if return_last_pos:
        return (- (p_normed*np.log(p_normed)).sum(), time_series[-1, 0], time_series[-1, 1])
    else:
        return - (p_normed*np.log(p_normed)).sum()


def FTRTE(x0: float, y0: float, k: float, n: int, Ntot: int, metric: str = 'supremum', lmin: int = 1, eps: float = 10/100, threshold_eps: bool = True, approach: str = "maximum") -> np.ndarray:
    """
    Calculates the finite-time recurrence time entropy (FTRTE) [1-3] considering border effects [4] on the distribution of white vertical lines.
    The standard deviation is calculated using the selected approach described in the Appendix A (default is maximum).

    Parameters
    ----------
    x0 : float
        The initial value of the x-coordinate.
    y0 : float
        The initial value of the y-coordinate.
    k : float
        The non-linearity parameter of the map.
    n : int
        The number of iterations (length of the orbit - finite-time).
    Ntot : int
        The total number of iterations.
    metric : str, optional
        The metric for measuring distances in phase space. Possible values are 'supremum', 'manhattan', 'euclidean' (default='supremum').
    lmin : int, optional
        Minimal length of white vertical lines used in the RTE computation (default=1).
    eps : float, optional
        Threshold for the recurrence plot in unit of time series standard deviation (default=10/100).
    threshold_eps : bool, optional
        If True, generates the recurrence plot using a fixed threshold in units of the time series standard deviation (default=True)
    approach : str, optional
        The approach for calculating the standard deviation. Possible values are 'concatenation', 'maximum', 'euclidean' (default='maximum')

    Returns
    -------
    out : ndarray
        Array with the FTRTE distribution.

    References
    ----------
    [1] http://www.pik-potsdam.de/~donges/pyunicorn/api/timeseries/recurrence_plot.html
    [2] M. A. Little et al., Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection, BioMedical Engineering OnLine 6, 23 (2007)
    [3] K. H. Kraemer et al., Recurrence threshold selection for obtaining robust recurrence characteristics in different embedding dimensions, Chaos 28, 085720 (2018)
    [4] K. H. Kraemer and N. Marwan, Border effect corrections for diagonal line based recurrence quantification analysis measures, Physics Letters A 383, 125977 (2019)
    """
    N = round(Ntot/n)
    ftrte = np.zeros(N)
    x = x0
    y = y0
    for i in range(N):
        ftrte[i], x, y = RTE(x, y, k, n, metric=metric, lmin=lmin, eps=eps, return_last_pos=True,  threshold_eps=threshold_eps, approach=approach)

    return ftrte

def corr_coef(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient between two arrays x and y.

    Parameters
    ----------
    x : ndarray
        The first input array. Must have the same number of elements as `y`.
    y : ndarray
        The second input array. Must have the same number of elements as `x`.

    Returns
    -------
    correlation_coefficient : float
        The Pearson correlation coefficient between `x` and `y`.

    Examples
    --------
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2, 4, 6, 8, 10]
    >>> corr_coef(x, y)
    1.0

    >>> x = [1, 2, 3, 4, 5]
    >>> y = [10, 8, 6, 4, 2]
    >>> corr_coef(x, y)
    -1.0

    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2, 4, 7, 8, 10]
    >>> corr_coef(x, y)
    0.957...

    Notes
    -----
    The correlation coefficient is a value between -1 and 1, indicating the strength and direction of the linear relationship between `x` and `y`. A value of -1 indicates a perfect negative linear relationship, a value of 1 indicates a perfect positive linear relationship, and a value of 0 indicates no linear relationship.

    """
    std_x = np.std(x)
    std_y = np.std(y)
    covxy = np.cov(x, y)[0][1]
    cc = covxy/(std_x*std_y)

    return cc

def plot_params(fontsize: int = 20, tick_labelsize: int = 17, legend_fontsize: int = 14) -> None:
    """
    Update the parameters of the plot.

    Returns
    -------
    cmap : string
        The color map used in the colored plots.
    """
    plt.clf()
    plt.rc('font', size=fontsize)
    plt.rc('xtick', labelsize=tick_labelsize)
    plt.rc('ytick', labelsize=tick_labelsize)
    plt.rc('legend', fontsize=legend_fontsize)
    font = {'family' : 'stix'}
    plt.rc('font', **font)
    plt.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams['axes.linewidth'] = 1.3 #set the value globally