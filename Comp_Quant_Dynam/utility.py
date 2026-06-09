import numpy as np   # standard numerics library
from collections.abc import Iterable, Sequence
from numpy import pi, sin, cos, tan, arcsin, arccos, arctan, sqrt, exp
from scipy.special import factorial, binom


def example_func(x):
    """
    Example function to demonstrate the repository structure.
    Returns the ground state wavefunction of the quantum harmonic oscillator at position 'x' in numerical units.
    """

    return 1 / pi ** (1 / 4) * exp(-x ** 2 / 2)


#################### Solution sheet 2 ####################


def create_xvals(L, npoints, endpoint=True):
    """
    Creates a grid of 'npoints' evenly spaced values between -L/2 and L/2.
    The 'endpoint' parameter determines whether the endpoint L/2 is included in the grid.
    Returns the grid of x values and the grid spacing dx.
    """
    xvals = np.linspace(-L / 2, L / 2, npoints, endpoint=endpoint)
    dx = xvals[1] - xvals[0]
    return xvals, dx


#################### Solution sheet 3 ####################


def FT(psi, x, k):
    """
    Computes the discrete Fourier transform of the wavefunction `psi` defined on the grid `x` to the momentum space grid `k`.
    """
    npoints = len(x)
    assert len(k) == npoints, "Length of k must match length of x"
    return np.sum(psi * exp(-1j * np.outer(k, x)), axis=1)

def iFT(phi, x, k):
    """
    Computes the inverse discrete Fourier transform of the momentum space wavefunction `phi` defined on the grid `k` to the position space grid `x`.
    """
    npoints = len(x)
    assert len(k) == npoints, "Length of k must match length of x"
    return 1 / npoints * np.sum(phi * exp(1j * np.outer(x, k)), axis=1)

def gaussian_wave_packet(x, x0, sigma, p0):
    """
    Creates a Gaussian wave packet centered at position `x0` with width `sigma` and momentum `p0`.
    """
    norm = 1 / np.sqrt(np.sqrt(2 * np.pi) * sigma)
    # p0 * x0 is a global phase factor that we can ignore, so we can omit it
    #  in the expression for the wave packet.
    return norm * exp(-(x - x0) ** 2 / (4 * sigma ** 2) + 1j * p0 * x)

def create_tvecs(tsteps, dt):
    """
    Creates a time vector for steps'' time steps with time step size 'dt'.
    Returns the time vector of length tsteps+1, starting from 0 to tsteps*dt.
    """
    return np.linspace(0, tsteps * dt, tsteps + 1) # will have length tsteps + 1


##################### Exercise sheet 4 ###################

def idx2state(N1, N2, i):
    """
    Converts a single index `i` to a 'state' in the product Hilbert space |n1, n2> of dimension N1 x N2.
    """
    assert i >= 0 and i < N1 * N2, "Index out of bounds"
    n1 = i // N2
    n2 = i % N2
    state = [n1, n2]
    return state 
 
def state2idx(N1, N2, state):
    """
    Converts a `state` |n1, n2> from the product Hilbert space of dimension `N1 x N2` to a single index `i`.
    """
    n1 = state[0]
    n2 = state[1]
    if n1 < 0 or n1 >= N1 or n2 < 0 or n2 >= N2:
        i = -1 # return -1 if the state is out of bounds
    else:
        i = n1 * N2 + n2
    return i
    

##################### Solution sheet 4 ###################

def create_coherent_state(N, alpha):
    """
    Creates a coherent state |alpha> in the Fock basis of dimension `N` with complex amplitude `alpha`.
    The coherent state is defined as:
    |alpha> = exp(-|alpha|^2/2) sum_{n=0}^{N-1} (alpha^n / sqrt(n!)) |n>
    """

    nvec = np.arange(N)
    state = exp(-np.abs(alpha) ** 2 / 2) * np.power(alpha, nvec) / sqrt(factorial(nvec))
    return state

def expectation_value(state, operator):
    """
    Computes the expectation value of an `operator` in a given `state`.
    The `operator` argument can be either a single operator or an iterable of operators.
    If it is an iterable, the function returns a vector of expectation values for each operator.
    """
    n_obsv, operator = _check_if_sized(operator)
    if n_obsv > 1:
        return np.array([expectation_value(state, op) for op in operator])

    return np.vdot(state, operator @ state)

def _check_if_sized(obsv_vec):
    """
    Helper function to check if the input `obsv_vec` is an iterable of operators or a single operator, and to determine the number of observables.
    If `obsv_vec` is a single operator or a Sequence containing a single operator, it returns (1, obsv_vec).
    If `obsv_vec` is an iterable of operators, it returns (n_obsv, obsv_vec) where n_obsv is the number of observables.
    """
    if isinstance(obsv_vec, Iterable) and not isinstance(obsv_vec, (str, bytes)) and getattr(obsv_vec, "ndim", None) != 2:
        if isinstance(obsv_vec, Sequence):
            n_obsv = len(obsv_vec)
        else:
            # e.g. generator: materialize once so length is defined
            obsv_vec = tuple(obsv_vec)
            n_obsv = len(obsv_vec)
        if n_obsv == 1:
            obsv_vec = obsv_vec[0] # if there is only one observable, return it as a single operator instead of a list
    else:
        n_obsv = 1

    return n_obsv, obsv_vec


##################### Exercise sheet 7 ###################


def CSS(N, theta, phi):
    """
    Returns the coefficients of the coherent spin state (CSS) |theta, phi> in the Dicke basis of dimension N+1.
    The CSS is defined as:
    
    |theta, phi> = sum_{k=0}^N sqrt(binomial(N,k)) * (cos(theta/2)^k * sin(theta/2)^(N-k) * exp(i * k * phi)) |k>
    
    where |k> is the Dicke state with k 0-spins (spin up) and N-k 1-spins (spin down).
    Note that the CSS is a superposition of Dicke states with different numbers of excitations, and the coefficients depend on the angles theta and phi.
    The CSS is a generalization of the coherent state for spin systems, and it can be used to describe states that are localized around a specific point on the Bloch sphere.
    """
    
    # exceptions to avoid 0^0
    if theta == np.pi:
        return np.eye(1, N + 1, 0)[0]
    elif theta == 0:
        return np.eye(1, N + 1 , N)[0]
    else:
        kvec = np.arange(0, N + 1)
        trigonometric_part = cos(theta / 2) ** kvec * sin(theta / 2) ** (N - kvec) * exp(1j * kvec * phi)
        return trigonometric_part * sqrt(binom(N, kvec))

def proj_CSS(psi, N, theta, phi):
    """
    Computes the projection of a state `psi` onto a coherent spin state (CSS) defined by angles `theta` and `phi` for a system of `N` spins.
    """
    
    css_state = CSS(N, theta, phi)
    return np.abs(psi.conj().T @ css_state) ** 2

def Husimi_th_ph(N, psi, nth, nph):
    """
    Computes the Husimi distribution of a state `psi` on a grid of angles `theta` and `phi` for a system of `N` spins.
    The grid is defined by `nth` points in the theta direction ([0, pi]) and `nph` points in the phi direction ([0, 2*pi)).
    Returns the grid of theta and phi values and the corresponding Husimi distribution values.
    """
    Theta = np.linspace(0, pi, nth, endpoint=True)
    Phi = np.linspace(0, 2 * pi, nph, endpoint=False)
    # container for Husimi function
    H = np.zeros((nth, nph))
    # calculate H on the grid
    for x_idx, theta in enumerate(Theta):
        for y_idx, phi in enumerate(Phi):
            H[x_idx, y_idx] = proj_CSS(psi, N, theta, phi)
    return Theta, Phi, H


def Husimi_z_phi(N, psi, nz, nph):
    """
    Computes the Husimi distribution of a state `psi` on a grid of `z` and `phi` for a system of `N` spins.
    The grid is defined by `nz` points in the z direction ([-1, 1]) and `nph` points in the phi direction ([0, 2*pi)).
    Returns the grid of z and phi values and the corresponding Husimi distribution values.
    """

    Z = np.linspace(-1, 1, nz, endpoint=True)
    th = arccos(Z)
    Phi = np.linspace(0, 2 * pi, nph, endpoint=False)
    H = np.zeros((nz, nph))
    for x_idx, theta in enumerate(th):
        for y_idx, phi in enumerate(Phi):
            H[x_idx, y_idx] = proj_CSS(psi, N, theta, phi)
    return Z, Phi, H
    

def Husimi_front(N, psi, nz, ny):
    """
    Computes the Husimi distribution of a state `psi` on a grid of `z` and `y` for a system of `N` spins, looking from the +x direction.
    The grid is defined by `nz` points in the z direction ([-1, 1]) and `ny` points in the y direction ([-1, 1]).
    Returns the grid of z and y values and the corresponding Husimi distribution values.
    """

    Z = np.linspace(-1, 1, nz, endpoint=True)
    Y = np.linspace(-1, 1, ny, endpoint=True)
    H = np.zeros((nz, ny))
    mask = np.zeros_like(H, dtype=bool) # make pixels outside of the circle white
    for idx_z, z in enumerate(Z):
        for idx_y, y in enumerate(Y):
            r2 = z ** 2 + y ** 2
            if r2 >= 1 + 1e-10: # outside allowed region
                H[idx_z, idx_y] = 0
                mask[idx_z, idx_y] = True
            else:
                theta = arccos(z)
                if theta == 0 or theta == np.pi: # in this case phi is undetermined
                    phi = 0
                else:
                    sin_phi = y / sin(theta)
                    if abs(sin_phi) > 1:
                        sin_phi = int(sin_phi / abs(sin_phi)) # set to 1 or -1
                    phi = arcsin(sin_phi) # corresponds to positive x
                H[idx_z, idx_y] = proj_CSS(psi, N, theta, phi)
                mask[idx_z, idx_y] = False
    H = np.ma.array(H, mask=mask)
    return Z, Y, H

def Husimi_back(N, psi, nz, ny):
    """
    Computes the Husimi distribution of a state `psi` on a grid of `z` and `y` for a system of `N` spins, looking from the -x direction.
    The grid is defined by `nz` points in the z direction ([-1, 1]) and `ny` points in the y direction ([-1, 1]).
    Returns the grid of z and y values and the corresponding Husimi distribution values.
    """
    
    Z = np.linspace(-1, 1, nz, endpoint=True)
    Y = np.linspace(-1, 1, ny, endpoint=True)
    H = np.zeros((nz, ny))
    mask = np.zeros_like(H, dtype=bool)
    for idx_z, z in enumerate(Z):
        for idx_y, y in enumerate(Y):
            r2 = z ** 2 + y ** 2
            if r2 >= 1 + 1e-10: # outside allowed region
                H[idx_z, idx_y] = 0
                mask[idx_z, idx_y] = True
            else:
                theta = arccos(z)
                if theta == 0 or theta == pi:
                    phi = 0
                else:
                    sin_phi = y / sin(theta)
                    if abs(sin_phi) > 1:
                        sin_phi = int(sin_phi / abs(sin_phi)) # set to 1 or -1
                    phi = pi - arcsin(sin_phi) # corresponds to negative x
                H[idx_z, idx_y] = proj_CSS(psi, N, theta, phi)
                mask[idx_z, idx_y] = False
    H = np.ma.array(H, mask = mask)
    return Z, Y, H

def Husimi_top(N, psi, nx, ny):
    """
    Computes the Husimi distribution of a state `psi` on a grid of `x` and `y` for a system of `N` spins, looking from the +z direction.
    The grid is defined by `nx` points in the x direction ([-1, 1]) and `ny` points in the y direction ([-1, 1]).
    Returns the grid of x and y values and the corresponding Husimi distribution values.
    """
    
    X = np.linspace(-1 , 1, nx, endpoint=True)
    Y = np.linspace(-1 , 1, ny, endpoint=True)
    H = np.zeros((nx, ny))
    mask = np.zeros_like(H, dtype=bool)
    for idx_x, x in enumerate(X):
        for idx_y, y in enumerate(Y):
            r2 = x ** 2 + y ** 2
            if r2 >= 1 + 1e-10: # outside allowed region
                H[idx_x, idx_y] = 0
                mask[idx_x, idx_y] = True
            else:
                if r2 > 1: # numerical issues close to the boundary
                    r2 = 1
                z = np.sqrt(1 - r2)
                theta = np.arccos(z)
                # avoid dividing by 0; Gets a bit tricky.
                if np.isclose(x, 0):
                    if y >= 0:
                        phi = pi / 2
                    else:
                        phi = 3 * pi / 2
                elif x > 0:
                    phi = arctan(y / x)
                else:
                    phi = pi + arctan(y / x)
                H[idx_x, idx_y] = proj_CSS(psi, N, theta, phi)
                mask[idx_x, idx_y] = False
    H = np.ma.array(H, mask=mask)
    return X, Y, H