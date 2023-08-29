import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import vectorize, njit
from pyunicorn.timeseries import RecurrencePlot as RP # Pyunicorn module to 
from scipy.integrate import simps
import sys

@vectorize(['f8(i8, f8, f8)'],
            nopython=True)
def R(gamma: int, eps: float, theta: float) -> float:
    """
    Calculate the first positive root of the cubic equation that gives the radius R of the billiard
    using the Newton-Raphson method.
    
    Parameters
    ------------
    gamma : int
        Defines the shape of the billiard.
    eps   : float
        Modifies the shape of the billiard.
    theta : float or np.ndarray
        Polar angle measured counter-clockwise from the x-axis.
                  
    Returns
    ------------
    R : float or np.ndarray
        The radius of the billiard for a given angle theta. This is a scalar if theta is a scalar.
    """
    limit = 1e9
    b = 2*np.sqrt(3*eps)*np.cos(gamma*theta)/9

    if abs(b) < 1e-11:
        xn = 3.0
    else:
        xs = -2/(3*b)
        xi = xs/2.0
        if xs <= 0:
            xn = 3.0
        else:
            if xi > 3.0:
                xn = 3.0
            else:
                xn = xi
    
    e = 1/limit
    i = 1
    while True:
        if i > limit:
            print('No convergence for theta!!!')
            #print('theta = %g\ngamma = %i\neps = %g' % (theta*180/np.pi, gamma, eps))
        FR = xn**2 + b*xn**3 - 1
        dFdR = xn*(3*b*xn + 2)
        xn_new = xn - FR/dFdR
        d = xn_new - xn
        if abs(d) < e:
            return xn_new
        xn = xn_new
        i = i + 1
    
    return -1

def F_R(R: float, b: float) -> float:
    """
    Calculate the value of F(R) = R² + bR³ - 1.

    Parameters
    ----------
    R : float
        The value of R.
    b : float
        b = 2*np.sqrt(3*eps)/9 * cos(theta)
    
    Returns
    -------
    float : the derivative of R.
    """

    return R**2 + b*R**3 - 1

def Fl_R(R : float, b : float) -> float:
    """
    Calculate the value of F'(R) = R(3bR + 2).

    Parameters
    ----------
    R : float
        The value of R.
    b : float
        b = 2*np.sqrt(3*eps)/9 * cos(theta)
    
    Returns
    -------
    float : the derivative of R.
    """
    return R*(3*b*R + 2)

@vectorize(['f8(f8, i8, f8, f8)'],
          nopython=True)
def dRdtheta(R: float, gamma: int, eps: float, theta: float) -> float:
    """
    Calculate the first derivative of R(theta), d R/d theta.

    Parameters
    ------------
    R : float or np.ndarray
        The radius of the billiard for a given angle theta. Must be of same size as theta.
    gamma : int
        Defines the shape of the billiard.
    eps : float
        Modifies the shape of the billiard.
    theta : float or np.ndarray
        Polar angle measured counter-clockwise from the x-axis. Must be of same size as R.
    
    Returns
    ------------
    dRdtheta : float or np.ndarray
        The first derivative of R(theta). This is a scalar if R and theta are scalars.
    """
    a = 2*np.sqrt(3*eps)/9

    cima = a*gamma*np.sin(gamma*theta)*R**2
    baixo = 2 + 3*a*np.cos(gamma*theta)*R

    return cima/baixo

@vectorize(['f8(f8, f8, i8, f8, f8)'],
          nopython=True)
def d2Rdtheta2(dRdtheta: float, R: float, gamma: int, eps: float, theta: float) -> float:
    """
    Calculate the second derivative of R(theta), d2 R/d theta2.

    Parameters
    ------------
    dR/dtheta : float or np.ndarray
        The first derivative of R for a given angle theta. Must be of same size as theta and R.
    R : float or np.ndarray
        The radius of the billiard for a given angle theta. Must be of same size as theta and dRdtheta.
    gamma : int
        Defines the shape of the billiard.
    eps : float
        Modifies the shape of the billiard.
    theta : float or np.ndarray
        Polar angle measured counter-clockwise from the x-axis. Must be of same size as R and dR/dtheta.
    
    Returns
    ------------
    dR2dtheta2 : float or np.ndarray
        The second derivative of R(theta). This is a scalar if R, dRdtheta and theta are scalars.
    """

    a = 2*np.sqrt(3*eps)/9

    AA = (2*R*dRdtheta*a*gamma*np.sin(gamma*theta) + a*gamma*gamma*R*R*np.cos(gamma*theta))*(2 + 3*a*R*np.cos(gamma*theta))
    BB = (a*gamma*R*R*np.sin(gamma*theta))*(3*a*dRdtheta*np.cos(gamma*theta) - 3*a*gamma*R*np.sin(gamma*theta))
    CC = (2 + 3*a*R*np.cos(gamma*theta))**2

    return (AA - BB)/CC

@njit
def collision_point(x0: float, y0: float, Rmax: float, gamma: int, eps: float, mu: float) -> float:
    """
    Calculate the next coliision point given the initial point (x0, y0) and the angle mu.

    This function uses Numba's `njit` decorator for performance optimization.

    Parameters
    ------------
    x0 : float
        Initial x-position on the boundary.
    y0 : float
        Initial y-position on the boundary.
    Rmax : float
        Maximum radius of the billiard. It is the value of R for theta = pi/gamma.
    gamma : int
        Defines the shape of the billiard.
    eps : float
        Modifies the shape of the billiard.
    mu : float
        Direction of the particle's velocity measured counter-clockwise from the x-axis.

    Returns
    -----------
    theta : float
        Polar angle measured counter-clockwise from the x-axis.
           
    """
    tol = 1e-11
    b = 2*(x0*np.cos(mu) + y0*np.sin(mu))
    c = x0**2 + y0**2 - Rmax**2
    dte = (-b + np.sqrt(b**2 - 4*c))/2
    j = 1
    while True:
        if(j > 200):
            print('No solution for theta!!')
            break
        xe = x0 + np.cos(mu)*dte
        ye = y0 + np.sin(mu)*dte
        re = np.sqrt(xe**2 + ye**2)
        thetaa = np.arctan2(ye, xe) % (2*np.pi)
        Ra = R(gamma, eps, thetaa)
        xa = Ra*np.cos(thetaa)
        ya = Ra*np.sin(thetaa)
        if abs(re - Ra) < tol and abs(xe - xa) < tol and abs(ye - ya) < tol:
            return thetaa
        # Update the positions for a new test
        Rla = dRdtheta(Ra, gamma, eps, thetaa)
        xla = Rla*np.cos(thetaa) - ya
        yla = Rla*np.sin(thetaa) + xa
        dte = (ya - y0 + (yla/xla)*(x0 - xa))/(np.sin(mu) - (yla/xla)*np.cos(mu))
        j += 1
    
    return -1

@njit
def time_series(theta0: float, alpha0: float, gamma: int, eps: float, num_coll: int) -> np.ndarray:
    """
    Calculate the (x, y) time series given an initial condition.

    This function uses Numba's `njit` decorator for performance optimization.

    Parameters
    ------------
    theta0   : float
        Initial polar angle on the boundary measured counter-clockwise from the x-axis.
    alpha0   : float
        Initial velocity's direction measured from the tangent line on theta0.
    gamma    : int
        Defines the shape of the billiard.
    eps      : float
        Modifies the shape of the billiard.
    num_coll : int
        Number of collisions.

    Returns
    ------------
    ts : np.ndarray
        The x and y time series. t[:, 0] = x and t[:, 1] = y.
    """
    # Return array
    ts = np.zeros((num_coll + 1, 2))
    # Initial quantities
    Rmax = R(gamma, eps, (np.pi/gamma))
    R0 = R(gamma, eps, theta0)
    Rl = dRdtheta(R0, gamma, eps, theta0)
    x0 = R0*np.cos(theta0)
    y0 = R0*np.sin(theta0)
    xl = Rl*np.cos(theta0) - y0
    yl = Rl*np.sin(theta0) + x0
    phi = np.arctan2(yl, xl) % (2*np.pi)
    mu = (alpha0 + phi) % (2*np.pi)
    # Initial condition
    ts[0, 0] = x0
    ts[0, 1] = y0
    
    for i in range(num_coll):
        theta = collision_point(ts[i, 0], ts[i, 1], Rmax, gamma, eps, mu)
        R0 = R(gamma, eps, theta)
        Rl = dRdtheta(R0, gamma, eps, theta)
        ts[i + 1, 0] = R0*np.cos(theta)
        ts[i + 1, 1] = R0*np.sin(theta)
        xl = Rl*np.cos(theta) - ts[i + 1, 1]
        yl = Rl*np.sin(theta) + ts[i + 1, 0]
        phi = np.arctan2(yl, xl) % (2*np.pi)
        alpha = (phi - mu) % (np.pi)
        mu = (alpha + phi) % (2*np.pi)
        
    return ts

@njit
def phase_space(theta: float, alpha: float, gamma: int, eps: float, num_coll: int) -> np.ndarray:
    """
    Calculate the (theta, alpha) time series given an initial condition.

    This function uses Numba's `njit` decorator for performance optimization.

    Parameters
    ------------
    theta0 : float
        Initial polar angle on the boundary measured counter-clockwise from the x-axis.
    alpha0 : float
        Initial velocity's direction measured from the tangent line on theta0.
    gamma : int
        Defines the shape of the billiard.
    eps : float
        Modifies the shape of the billiard.
    num_coll : int
        Number of collisions.

    Returns
    ------------
    ts : np.ndarray
        The theta and alpha time series. t[:, 0] = theta and t[:, 1] = alpha.
    """
    ps = np.zeros((num_coll + 1, 2))
    ps[0, 0] = theta
    ps[0, 1] = alpha
    # Initial quantities
    Rmax = R(gamma, eps, (np.pi/gamma))
    Ra = R(gamma, eps, theta)
    Rl = dRdtheta(Ra, gamma, eps, theta)
    x = Ra*np.cos(theta)
    y = Ra*np.sin(theta)
    xl = Rl*np.cos(theta) - Ra*np.sin(theta)
    yl = Rl*np.sin(theta) + Ra*np.cos(theta)
    phi = np.arctan2(yl, xl) % (2*np.pi)
    mu = (alpha + phi) % (2*np.pi)
    for i in range(1, num_coll + 1):
        theta = collision_point(x, y, Rmax, gamma, eps, mu)
        Ra = R(gamma, eps, theta)
        Rl = dRdtheta(Ra, gamma, eps, theta)
        x = Ra*np.cos(theta)
        y = Ra*np.sin(theta)
        xl = Rl*np.cos(theta) - Ra*np.sin(theta)
        yl = Rl*np.sin(theta) + Ra*np.cos(theta)
        phi = np.arctan2(yl, xl) % (2*np.pi)
        alpha = (phi - mu) % np.pi
        mu = (alpha + phi) % (2*np.pi)
        #
        ps[i, 0] = theta
        ps[i, 1] = alpha

    return ps

@vectorize(['f8(f8, f8, i8, f8, i8)'],
            target='parallel',
            nopython=True)
def lyapunov(theta: float, alpha: float, gamma: int, eps: float, num_coll: int) -> float:
    """
    Calculate the largest Lyapunov exponent for the billiard system.

    This function uses a vectorized implementation to optimize performance.

    Parameters
    ------------
    theta0 : float
        Initial polar angle on the boundary measured counter-clockwise from the x-axis.
    alpha0 : float
        Initial velocity's direction measured from the tangent line on theta0.
    gamma : int
        Defines the shape of the billiard.
    eps : float
        Modifies the shape of the billiard.
    num_coll : int
        Number of collisions.

    Returns
    -------
    out : float
        The largest Lyapunov exponent.
    """
    theta0 = theta
    alpha0 = alpha
    # Initial quantities
    Rmax = R(gamma, eps, (np.pi/gamma))
    R0 = R(gamma, eps, theta0)
    Rl0 = dRdtheta(R0, gamma, eps, theta0)
    Rll0 = d2Rdtheta2(Rl0, R0, gamma, eps, theta0)
    x0 = R0*np.cos(theta0)
    y0 = R0*np.sin(theta0)
    xl0 = Rl0*np.cos(theta0) - R0*np.sin(theta0)
    yl0 = Rl0*np.sin(theta0) + R0*np.cos(theta0)
    xll0 = Rll0*np.cos(theta0) - 2*Rl0*np.sin(theta0) - R0*np.cos(theta0)
    yll0 = Rll0*np.sin(theta0) + 2*Rl0*np.cos(theta0) - R0*np.sin(theta0)
    phi = np.arctan2(yl0, xl0) % (2*np.pi)
    mu = (alpha0 + phi) % (2*np.pi)
    beta0 = 0
    somaT11 = 0
    for i in range(num_coll):
        theta1 = collision_point(x0, y0, Rmax, gamma, eps, mu)
        R1 = R(gamma, eps, theta1)
        Rl1 = dRdtheta(R1, gamma, eps, theta1)
        Rll1 = d2Rdtheta2(Rl1, R1, gamma, eps, theta1)
        x1 = R1*np.cos(theta1)
        y1 = R1*np.sin(theta1)
        xl1 = Rl1*np.cos(theta1) - R1*np.sin(theta1)
        yl1 = Rl1*np.sin(theta1) + R1*np.cos(theta1)
        xll1 = Rll1*np.cos(theta1) - 2*Rl1*np.sin(theta1) - R1*np.cos(theta1)
        yll1 = Rll1*np.sin(theta1) + 2*Rl1*np.cos(theta1) - R1*np.sin(theta1)
        phi = np.arctan2(yl1, xl1) % (2*np.pi)
        alpha1 = (phi - mu) % np.pi
        deltax = x1 - x0
        dphidt0 = (xl0*yll0 - xll0*yl0)/(xl0**2 + yl0**2)
        dphidt1 = (xl1*yll1 - xll1*yl1)/(xl1**2 + yl1**2)
        chi = Rl1*(np.sin(theta1) - np.tan(mu)*np.cos(theta1)) + R1*(np.cos(theta1) + np.tan(mu)*np.sin(theta1))
        # Jacobian matrix in terms of (theta, alpha)
        J11 = ((1 + np.tan(mu)**2)*dphidt0*deltax + yl0 - np.tan(mu)*xl0)/chi
        J12 = deltax*(1 + np.tan(mu)**2)/chi
        J21 = dphidt1*J11 - dphidt0
        J22 = dphidt1*J12 - 1
        # Jacobian matrix in terms of the Birkhoff coordinates (s, p)
        J11 = np.sqrt((R1**2 + Rl1**2)/(R0**2 + Rl0**2))*J11
        J12 = (-np.sqrt(R1**2 + Rl1**2)/np.sin(alpha0))*J12
        J21 = (-np.sin(alpha1)/np.sqrt(R0**2 + Rl0**2))*J21
        J22 = (np.sin(alpha1)/np.sin(alpha0))*J22
        # Obtain the new angle of rotation for the Jacobian matrix triangularization
        beta = np.arctan((-J21*np.cos(beta0) + J22*np.sin(beta0))/(J11*np.cos(beta0) - J12*np.sin(beta0)))
        # Evaluate the diagonal elements of T = O^{-1} * J, which are its eigenvalues
        T11 = np.cos(beta0)*(J11*np.cos(beta) - J21*np.sin(beta)) - np.sin(beta0)*(J12*np.cos(beta) - J22*np.sin(beta))
        somaT11 = somaT11 + np.log(abs(T11))/np.log(2.0)
        # Update the rotation angle
        beta0 = beta
        # Update the particle's direction
        mu = (alpha1 + phi) % (2*np.pi)
        # Update the old variables
        theta0 = theta1
        alpha0 = alpha1
        x0 = x1
        y0 = y1
        xl0 = xl1
        yl0 = yl1
        xll0 = xll1
        yll0 = yll1
        R0 = R1
        Rl0 = Rl1
        Rll0 = Rll1

    lypnv = somaT11/num_coll

    return lypnv

@vectorize(['f8(f8, f8, i8, f8, i8)'],
            target='parallel',
            nopython=True)
def dig(theta0: float, alpha0: float, gamma: int, eps: float, num_coll: int):
    """
    Compute the dig measure for a given set of parameters.

    This function uses a vectorized implementation to optimize performance.

    Parameters
    ------------
    theta0 : float
        Initial polar angle on the boundary measured counter-clockwise from the x-axis.
    alpha0 : float
        Initial velocity's direction measured from the tangent line on theta0.
    gamma : int
        Defines the shape of the billiard.
    eps : float
        Modifies the shape of the billiard.
    num_coll : int
        Number of collisions.

    Returns
    -------
    out : float
        The dig measure.
    """
    u = np.arange(1, num_coll)/num_coll
    S = sum(np.exp(-1/(u*(1 - u))))
    # Evaluate WB(h) for the first num_coll iterates
    WB0 = 0
    ts = phase_space(theta0, alpha0, gamma, eps, num_coll)
    w = np.exp(-1/(u*(1 - u)))/S
    WB0 = sum(w*np.cos(ts[1:-1, 0]))
    theta = ts[-1, 0]
    alpha = ts[-1, 1]
    ts = phase_space(theta, alpha, gamma, eps, num_coll)
    w = np.exp(-1/(u*(1 - u)))/S
    WB1 = sum(w*np.cos(ts[1:-1, 0]))

    return -np.log10(abs(WB0 - WB1))

@njit
def white_vertline_distr(recmat: np.ndarray) -> np.ndarray:
    """
    Calculate the distribution of the lengths of white vertical lines in a binary matrix.

    Parameters
    ----------
    recmat : ndarray
        A 2-dimensional binary numpy array (recurrence matrix).

    Returns
    -------
    out: ndarray
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

def RTE(theta0: float, alpha0: float, gamma: int, eps: float, T: int, metric: str = 'supremum', lmin: int = 1, threshold: float = 10/100, threshold_std_eps: bool = True, approach: str = "maximum", return_last_pos: bool = False) -> float:
    """
    Return the recurrence time entropy [1] given an initial condition (`x0`, `y0`).
    
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

    Return
    ------
    out : tuple
        The white vertical entropy
        If return_last_pos = True, the second and third elements are last position in phase space (x, y).

    References
        ----------
    [1] http://www.pik-potsdam.de/~donges/pyunicorn/api/timeseries/recurrence_plot.html
    [2] M. A. Little et al., Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection, BioMedical Engineering OnLine 6, 23 (2007)
    [3] K. H. Kraemer et al., Recurrence threshold selection for obtaining robust recurrence characteristics in different embedding dimensions, Chaos 28, 085720 (2018)
    [4] K. H. Kraemer and N. Marwan, Border effect corrections for diagonal line based recurrence quantification analysis measures, Physics Letters A 383, 125977 (2019)
    """
    time_series = phase_space(theta0, alpha0, gamma, eps, T)
    sp_time_series = np.zeros_like(time_series)
    sp_time_series[:, 1] = np.cos(time_series[:, 1])
    for i in range(len(time_series[:, 0])):
        sp_time_series[i, 0] = arc_length(gamma, eps, theta_end=time_series[i, 0])
    if threshold_std_eps:
        if approach == "concatenation":
            threshold = sp_time_series.std()*threshold
        elif approach == "maximum":
            threshold = max(np.std(sp_time_series[:, 0]), np.std(sp_time_series[:, 1]))*threshold
        elif approach == "euclidean":
            threshold = np.sqrt(np.std(sp_time_series[:, 0])**2 + np.std(sp_time_series[:, 1])**2)*threshold
        else:
            print("Invalid approach!")
            sys.exit()
    else:
        threshold = threshold
    rp = RP(sp_time_series, metric=metric, normalize=False, threshold=threshold, silence_level=2)
    recmat = rp.recurrence_matrix()
    p = white_vertline_distr(recmat)
    p = p[lmin:]
    p = np.extract(p != 0, p)
    p_normed = p/p.sum()

    if return_last_pos:
        return (- (p_normed*np.log(p_normed)).sum(), time_series[-1, 0], time_series[-1, 1])
    else:
        return - (p_normed*np.log(p_normed)).sum()
    
def arc_length(gamma: int, eps: float, theta_ini: float = 0.0, theta_end: float = 2*np.pi):
    """
    Calculate the arc length of a parametric curve defined by functions R(gamma, eps, theta)
    and dRdtheta(radius, gamma, eps, theta) using Simpson's rule integration.

    Parameters
    ----------
    gamma : int
        Defines the shape of the billiard.
    eps : float
        Modifies the shape of the billiard.
    theta_ini : float, optional
        Starting angle for integration in radians. Default is 0.0.
    theta_end : float, optional
        Ending angle for integration in radians. Default is 2*pi.

    Returns
    -------
    out : float 
        Arc length of the parametric curve.
    """
    theta = np.linspace(theta_ini, theta_end, 500, endpoint=True)
    radius = R(gamma, eps, theta)
    radius_prime = dRdtheta(radius, gamma, eps, theta)
    f = np.sqrt(radius**2 + radius_prime**2)
    return simps(f, theta)


def plot_params(fontsize=20, tick_labelsize=17, legend_fontsize=14):
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