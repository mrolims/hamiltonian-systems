import numpy as np # NumPy module
from numba import njit # Numba module to create fast functions
# Plotting modules

@njit
def lyapunov_vs_n(x0: float, y0: float, k: float, N: int) -> np.ndarray:
    """
    Calculate the Lyapunov exponent of the standard map map.

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

@njit
def lyapunov(x0: float, y0: float, k: float, N: int) -> np.ndarray:
    """
    Calculate the Lyapunov exponents of the standard map map.

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